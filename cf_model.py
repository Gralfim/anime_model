"""
cf_model.py — User-based Collaborative Filtering přes AniList API

Pipeline:
  1. Výběr seed titulů (niche anime s vysokým tvým skóre)
  2. Pro každý seed titul: stáhni uživatele kteří ho dokončili
     z AniList (endpoint Page { mediaList }) — vrací všechny uživatele,
     ne jen ty s nedávnou aktivitou jako Jikan
  3. Stáhni kompletní animlisty těchto uživatelů (MediaListCollection)
  4. Výpočet Pearsonovy korelace na diferenciálních skóre
     diff = user_score - anilist_avg_score
     → odstraní bias "každý hodnotí populární anime vysoko"
  5. Weighted average hodnocení podobných uživatelů → doporučení

Datové poznámky:
  - AniList skóre jsou 0–100, normalizujeme na 0–10 (dělení 10)
  - MAL ↔ AniList ID mapping přes pole idMal (AniList vrací MAL ID)
  - Uživatelé jsou identifikováni AniList user ID (int), ne username
"""

import json
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tabulate import tabulate

log = logging.getLogger(__name__)

# ── GraphQL dotazy ─────────────────────────────────────────────────────────────

# Uživatelé kteří dokončili daný titul (podle AniList media ID)
QUERY_MEDIA_WATCHERS = """
query ($mediaId: Int, $page: Int) {
  Page(page: $page, perPage: 50) {
    pageInfo { hasNextPage }
    mediaList(mediaId: $mediaId, status: COMPLETED, sort: SCORE_DESC) {
      score
      user { id name }
    }
  }
}
"""

# Přeložení MAL ID na AniList media ID
QUERY_MAL_TO_AL = """
query ($idMal: Int) {
  Media(idMal: $idMal, type: ANIME) {
    id
    idMal
    title { romaji }
    averageScore
    popularity
  }
}
"""

# Kompletní animelist uživatele (jen dokončené s hodnocením)
QUERY_USER_LIST = """
query ($userId: Int, $page: Int) {
  Page(page: $page, perPage: 50) {
    pageInfo { hasNextPage }
    mediaList(userId: $userId, type: ANIME, status: COMPLETED,
              sort: SCORE_DESC) {
      score
      user { mediaListOptions { scoreFormat } }
      media {
        id
        idMal
        averageScore
        title { romaji }
      }
    }
  }
}
"""


# ── Datové struktury ──────────────────────────────────────────────────────────

@dataclass
class CFConfig:
    """Konfigurace CF modelu — načítá se z config.yaml sekce cf."""

    # Seed tituly
    seed_min_score:      int   = 8      # min. tvoje MAL skóre
    seed_max_popularity: int   = 50000  # max. AniList popularity (niche filtr)
    seed_titles_count:   int   = 20     # kolik seed titulů použít
    watchers_per_seed:   int   = 100    # max. uživatelů na seed titul

    # Filtrování podobných uživatelů
    min_overlap:         int   = 15     # min. společných titulů
    min_correlation:     float = 0.15   # min. Pearsonova korelace
    top_k_users:         int   = 50     # počet nejpodobnějších uživatelů

    # Predikce
    min_cf_users:        int   = 3      # min. podobných uživatelů hodnotitelů
    show_top:            int   = 25     # počet výsledků
    min_avg_score:       float = 65.0   # min. AniList průměr (0–100 škála)
    request_delay:       float = 0.8    # sekundy mezi AniList requesty

    # Technické
    cache_dir:           str   = "cache"

    @classmethod
    def from_config(cls, cfg: dict) -> "CFConfig":
        c = cfg.get("cf", {})
        return cls(
            seed_min_score=      c.get("seed_min_score",      8),
            seed_max_popularity= c.get("seed_max_popularity",  50000),
            seed_titles_count=   c.get("seed_titles_count",    20),
            watchers_per_seed=   c.get("watchers_per_seed",    100),
            min_overlap=         c.get("min_overlap",          15),
            min_correlation=     c.get("min_correlation",      0.15),
            top_k_users=         c.get("top_k_users",          50),
            min_cf_users=        c.get("min_cf_users",         3),
            show_top=            c.get("show_top",             25),
            min_avg_score=       c.get("min_avg_score",        65.0),
            request_delay=       c.get("request_delay",        0.8),
            cache_dir=           cfg.get("cache_dir",          "cache"),
        )


@dataclass
class SimilarUser:
    user_id:     int
    username:    str
    correlation: float
    overlap:     int
    avg_diff:    float   # průměrný diferenciál tohoto uživatele


@dataclass
class CFRecommendation:
    al_id:       int              # AniList media ID
    mal_id:      int | None       # MAL ID (může být None)
    title:       str
    cf_score:    float            # predikované skóre na MAL škále (1–10)
    avg_score:   float            # AniList průměr normalizovaný na 0–10
    n_users:     int              # počet podobných uživatelů kteří hodnotili
    avg_sim:     float            # průměrná korelace hodnotitelů
    top_raters:  list[str]        # jména nejpodobnějších hodnotitelů


# ── AniList HTTP klient (CF-specifický, s vlastní cache) ─────────────────────

class AniListCFClient:
    """
    Jednoduchý AniList GraphQL klient pro CF pipeline.
    Vlastní cache oddělená od content-based AniList cache.
    Sdílí rate-limiting logiku s anilist_client.py.
    """

    GRAPHQL_URL  = "https://graphql.anilist.co"
    RETRY_DELAYS = [5, 15, 60]

    def __init__(self, cache_dir: str, request_delay: float = 0.8):
        self._cache     = Path(cache_dir) / "cf_al"
        self._cache.mkdir(parents=True, exist_ok=True)
        self._delay     = request_delay
        self._last_req  = 0.0

        import requests
        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
            "Accept":       "application/json",
            "User-Agent":   "anime-taste-model/1.0",
        })

    # ── Cache ──────────────────────────────────────────────────────────────────

    def _cf(self, key: str) -> Path:
        safe = key.replace("/", "_").replace("?", "_").replace(":", "_")
        return self._cache / f"{safe}.json"

    def _load(self, key: str):
        f = self._cf(key)
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None

    def _save(self, key: str, data) -> None:
        self._cf(key).write_text(
            json.dumps(data, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8"
        )

    # ── HTTP ───────────────────────────────────────────────────────────────────

    def _post(self, query: str, variables: dict) -> dict | None:
        elapsed = time.time() - self._last_req
        if elapsed < self._delay:
            time.sleep(self._delay - elapsed)

        import requests as req_mod
        for attempt, wait in enumerate(self.RETRY_DELAYS + [None]):
            try:
                resp = self._session.post(
                    self.GRAPHQL_URL,
                    json={"query": query, "variables": variables},
                    timeout=20,
                )
                self._last_req = time.time()

                if resp.status_code == 429:
                    w = int(resp.headers.get("Retry-After", wait or 60))
                    log.warning(f"AniList rate limit, čekám {w}s…")
                    time.sleep(w)
                    continue

                if resp.status_code == 404:
                    return None

                resp.raise_for_status()
                data = resp.json()

                if "errors" in data:
                    for e in data["errors"]:
                        log.debug(f"GraphQL error: {e.get('message')}")
                    return None

                return data

            except req_mod.RequestException as e:
                if attempt < len(self.RETRY_DELAYS) - 1:
                    log.warning(f"Request chyba ({e}), retry za {wait}s…")
                    time.sleep(wait)
                else:
                    log.error(f"AniList CF request selhal: {e}")
                    return None
        return None

    # ── Veřejné metody ─────────────────────────────────────────────────────────

    def mal_to_al_id(self, mal_id: int) -> dict | None:
        """
        Převede MAL ID na AniList media ID a stáhne základní metadata.
        Výsledek: {"id": int, "idMal": int, "title": str,
                   "averageScore": int, "popularity": int}
        """
        key = f"mal2al_{mal_id}"
        cached = self._load(key)
        if cached is not None:
            return cached if cached else None

        data = self._post(QUERY_MAL_TO_AL, {"idMal": mal_id})
        media = (data or {}).get("data", {}).get("Media")
        if media:
            self._save(key, media)
        else:
            self._save(key, {})  # sentinel: nenalezeno
        return media

    def get_media_watchers(
        self,
        al_id:    int,
        max_users: int = 100,
    ) -> list[dict]:
        """
        Vrátí uživatele kteří dokončili daný titul.
        Výsledek: [{"user_id": int, "username": str, "score": float}, ...]
        score je normalizovaný na 0–10.
        """
        key = f"watchers_{al_id}_n{max_users}"
        cached = self._load(key)
        if cached is not None:
            return cached

        results = []
        page    = 1
        pages_needed = math.ceil(max_users / 50)

        while page <= pages_needed:
            data = self._post(QUERY_MEDIA_WATCHERS, {"mediaId": al_id, "page": page})
            page_data = (data or {}).get("data", {}).get("Page", {})
            if not page_data:
                break

            for entry in page_data.get("mediaList", []):
                score = (entry.get("score") or 0) / 10.0  # 0–100 → 0–10
                user  = entry.get("user") or {}
                uid   = user.get("id")
                uname = user.get("name")
                if uid and uname and score > 0:
                    results.append({
                        "user_id":  uid,
                        "username": uname,
                        "score":    score,
                    })

            if not page_data.get("pageInfo", {}).get("hasNextPage"):
                break
            page += 1
            time.sleep(self._delay)

        self._save(key, results)
        return results

    def get_user_animelist(self, user_id: int) -> list[dict]:
        """
        Vrátí kompletní dokončené hodnocení uživatele.
        Výsledek: [{"al_id": int, "mal_id": int|None,
                    "score": float,       # 0–10
                    "avg_score": float,   # 0–10 (AniList průměr)
                    "title": str}, ...]
        """
        key = f"userlist_{user_id}"
        cached = self._load(key)
        if cached is not None:
            return cached

        results = []
        page    = 1

        while True:
            data = self._post(QUERY_USER_LIST, {"userId": user_id, "page": page})
            page_data = (data or {}).get("data", {}).get("Page", {})
            if not page_data:
                break

            for entry in page_data.get("mediaList", []):
                raw     = entry.get("score") or 0
                fmt     = (entry.get("user") or {}).get(
                             "mediaListOptions", {}).get("scoreFormat", "POINT_10")
                divisor = {"POINT_100": 10.0, "POINT_5": 2.0,
                           "POINT_3": 10.0/3}.get(fmt, 1.0)
                score   = raw / divisor
                if score <= 0:
                    continue
                media     = entry.get("media") or {}
                al_id     = media.get("id")
                mal_id    = media.get("idMal")
                avg_raw   = media.get("averageScore") or 0
                avg_score = avg_raw / 10.0
                title     = (media.get("title") or {}).get("romaji", "")
                if al_id:
                    results.append({
                        "al_id":     al_id,
                        "mal_id":    mal_id,
                        "score":     score,
                        "avg_score": avg_score,
                        "title":     title,
                    })

            if not page_data.get("pageInfo", {}).get("hasNextPage"):
                break
            page += 1
            time.sleep(self._delay)

        self._save(key, results)
        return results

    def clear_watcher_cache(self) -> int:
        """Smaže cache seed uživatelů (watcher soubory). Vrátí počet smazaných."""
        deleted = 0
        for f in self._cache.glob("watchers_*.json"):
            f.unlink()
            deleted += 1
        return deleted


# ── CF Engine ─────────────────────────────────────────────────────────────────

class CollaborativeFilter:
    """User-based collaborative filtering přes AniList API."""

    def __init__(self, cfg: CFConfig):
        self.cfg    = cfg
        self.client = AniListCFClient(cfg.cache_dir, cfg.request_delay)

    # ── Pomocné funkce ─────────────────────────────────────────────────────────

    def _pearson_diff(
        self,
        my_diffs:    dict[int, float],   # {al_id: diff}
        other_diffs: dict[int, float],   # {al_id: diff}
    ) -> tuple[float, int] | None:
        """
        Pearsonova korelace na průnicích al_id.
        Vrací (korelace, overlap) nebo None pokud nedostatečný překryv.
        """
        common = sorted(set(my_diffs) & set(other_diffs))
        if len(common) < self.cfg.min_overlap:
            return None

        a = np.array([my_diffs[k]    for k in common])
        b = np.array([other_diffs[k] for k in common])

        a_c   = a - a.mean()
        b_c   = b - b.mean()
        denom = np.sqrt((a_c**2).sum() * (b_c**2).sum())
        if denom < 1e-9:
            return None

        r = float(np.dot(a_c, b_c) / denom)
        return (r, len(common)) if r >= self.cfg.min_correlation else None

    # ── Výběr seed titulů ──────────────────────────────────────────────────────

    def select_seed_titles(
        self,
        scored_entries: list,     # list[MalEntry]
        jikan_data:     dict,     # {mal_id: jikan_data} — pro members/popularity
    ) -> list[tuple[int, int, str, int]]:
        """
        Vybere niche tituly pro seed.
        Vrací [(mal_id, al_id, title, popularity), ...] seřazené dle popularity.
        """
        cfg        = self.cfg
        candidates = []

        for entry in scored_entries:
            if entry.score < cfg.seed_min_score:
                continue

            # Zkus AniList metadata (popularity je spolehlivější než MAL members)
            al_meta = self.client.mal_to_al_id(entry.mal_id)
            if not al_meta:
                continue

            popularity = al_meta.get("popularity") or 0
            if popularity == 0 or popularity > cfg.seed_max_popularity:
                continue

            al_id = al_meta["id"]
            title = (al_meta.get("title") or {}).get("romaji") or entry.title
            candidates.append((entry.mal_id, al_id, title, popularity))

        # Nejnišovější první
        candidates.sort(key=lambda x: x[3])
        return candidates[:cfg.seed_titles_count]

    # ── Sběr kandidátů ────────────────────────────────────────────────────────

    def collect_candidate_users(
        self,
        seed_titles: list[tuple[int, int, str, int]],
    ) -> dict[int, str]:
        """
        Pro každý seed titul stáhne uživatele kteří ho dokončili.
        Vrací {user_id: username} — deduplikovaný slovník.
        """
        all_users: dict[int, str] = {}

        for mal_id, al_id, title, popularity in seed_titles:
            print(
                f"  Seed: {title[:42]:<42} pop={popularity:>6,}  ",
                end="\r"
            )
            watchers = self.client.get_media_watchers(al_id, self.cfg.watchers_per_seed)
            for w in watchers:
                all_users[w["user_id"]] = w["username"]

        print(f"\n  Kandidátů: {len(all_users)} unikátních uživatelů z "
              f"{len(seed_titles)} seed titulů")
        return all_users

    # ── Výpočet podobnosti ────────────────────────────────────────────────────

    def find_similar_users(
        self,
        my_al_diffs: dict[int, float],    # {al_id: diff} — tvůj diff vektor
        candidate_users: dict[int, str],  # {user_id: username}
    ) -> list[SimilarUser]:
        """
        Stáhne animlisty kandidátů a spočítá Pearsonovu korelaci.
        """
        cfg     = self.cfg
        similar = []
        total   = len(candidate_users)

        for i, (user_id, username) in enumerate(candidate_users.items()):
            if i % 25 == 0:
                print(
                    f"  Podobnost: {i}/{total}  "
                    f"nalezeno: {len(similar)}…",
                    end="\r"
                )
            try:
                entries = self.client.get_user_animelist(user_id)
            except Exception as e:
                log.debug(f"Chyba pro uživatele {username}: {e}")
                continue

            if not entries:
                continue

            # Sestavení diff vektoru pro tohoto uživatele
            other_diffs: dict[int, float] = {}
            for e in entries:
                al_id     = e.get("al_id")
                score     = e.get("score", 0)
                avg_score = e.get("avg_score", 0)
                if al_id and score > 0 and avg_score > 0:
                    other_diffs[al_id] = score - avg_score

            result = self._pearson_diff(my_al_diffs, other_diffs)
            if result is None:
                continue

            corr, overlap = result
            avg_diff = float(np.mean(list(other_diffs.values()))) if other_diffs else 0.0

            similar.append(SimilarUser(
                user_id     = user_id,
                username    = username,
                correlation = corr,
                overlap     = overlap,
                avg_diff    = avg_diff,
            ))

        similar.sort(key=lambda u: -u.correlation)
        print(
            f"\n  Podobní uživatelé: {len(similar)} "
            f"(práh r≥{cfg.min_correlation}, overlap≥{cfg.min_overlap})"
        )
        return similar[:cfg.top_k_users]

    # ── Agregace doporučení ───────────────────────────────────────────────────

    def aggregate_recommendations(
        self,
        similar_users: list[SimilarUser],
        existing_al_ids: set[int],
    ) -> list[CFRecommendation]:
        """
        Agreguje hodnocení podobných uživatelů na neviděné tituly.
        cf_score = avg_score + weighted_avg(diff) → normalizováno na 1–10.
        """
        cfg = self.cfg

        # {al_id: [(diff, weight, username, avg_score, mal_id, title)]}
        item_data: dict[int, list] = defaultdict(list)

        for user in similar_users:
            try:
                entries = self.client.get_user_animelist(user.user_id)
            except Exception:
                continue

            for e in entries:
                al_id     = e.get("al_id")
                score     = e.get("score", 0)
                avg_score = e.get("avg_score", 0)
                mal_id    = e.get("mal_id")
                title     = e.get("title", "")

                if not al_id or score <= 0 or avg_score < cfg.min_avg_score / 10:
                    continue
                if al_id in existing_al_ids:
                    continue

                diff = score - avg_score
                item_data[al_id].append((
                    diff, user.correlation, user.username,
                    avg_score, mal_id, title,
                ))

        recommendations = []
        for al_id, entries_list in item_data.items():
            if len(entries_list) < cfg.min_cf_users:
                continue

            total_w      = sum(w for _, w, *_ in entries_list)
            if total_w < 1e-9:
                continue

            weighted_diff = sum(d * w for d, w, *_ in entries_list) / total_w
            avg_score_    = float(np.mean([a for _, _, _, a, *_ in entries_list]))
            cf_score      = float(np.clip(avg_score_ + weighted_diff, 0.1, 10.0))
            avg_sim       = float(np.mean([w for _, w, *_ in entries_list]))

            # Metadata z prvního záznamu (jsou shodné pro daný al_id)
            _, _, _, _, mal_id, title = entries_list[0]
            top_raters = [
                u for _, _, u, *_ in sorted(entries_list, key=lambda x: -x[1])[:5]
            ]

            recommendations.append(CFRecommendation(
                al_id      = al_id,
                mal_id     = mal_id,
                title      = title,
                cf_score   = cf_score,
                avg_score  = avg_score_,
                n_users    = len(entries_list),
                avg_sim    = avg_sim,
                top_raters = top_raters,
            ))

        recommendations.sort(key=lambda r: -r.cf_score)
        return recommendations

    # ── Hlavní pipeline ────────────────────────────────────────────────────────

    def run(
        self,
        scored_entries:  list,        # list[MalEntry] — tvoje hodnocené tituly
        jikan_data:      dict,        # {mal_id: jikan_data}
        existing_mal_ids: set[int],   # všechny tvoje MAL tituly
        titles_map:      dict[int, str],
    ) -> tuple[list[CFRecommendation], list[SimilarUser]]:
        """
        Spustí celou CF pipeline.
        Vrací (doporučení, seznam podobných uživatelů).
        """
        # ── Příprava tvého diff vektoru (na AniList škále) ─────────────────
        print("\n  [CF 1/5] Překlad MAL ID na AniList ID a příprava diff vektoru…")
        my_al_diffs:    dict[int, float] = {}
        mal_to_al_map:  dict[int, int]   = {}
        existing_al_ids: set[int]        = set()
        translated = 0

        for entry in scored_entries:
            al_meta = self.client.mal_to_al_id(entry.mal_id)
            if not al_meta:
                continue
            al_id     = al_meta["id"]
            avg_raw   = al_meta.get("averageScore") or 0
            avg_score = avg_raw / 10.0
            mal_to_al_map[entry.mal_id] = al_id
            existing_al_ids.add(al_id)
            if entry.score > 0 and avg_score > 0:
                my_al_diffs[al_id] = (entry.score - avg_score)
                translated += 1

        # Zahrni i nehodnocené tituly do "existujících" pro filtrování výsledků
        for mal_id in existing_mal_ids:
            al_meta = self.client.mal_to_al_id(mal_id)
            if al_meta:
                existing_al_ids.add(al_meta["id"])

        print(f"  Přeloženo: {translated} titulů s diff vektorem "
              f"(průměr diff: {np.mean(list(my_al_diffs.values())):.2f})")

        # ── Výběr seed titulů ──────────────────────────────────────────────
        print(f"\n  [CF 2/5] Výběr niche seed titulů "
              f"(skóre ≥{self.cfg.seed_min_score}, "
              f"popularity ≤{self.cfg.seed_max_popularity:,})…")
        seed_titles = self.select_seed_titles(scored_entries, jikan_data)

        if not seed_titles:
            log.warning("Žádné seed tituly — zkus uvolnit seed_max_popularity.")
            return [], []

        print(f"\n  Vybraných seed titulů: {len(seed_titles)}")
        for mal_id, al_id, title, pop in seed_titles[:6]:
            print(f"    {title[:45]:<45} pop={pop:>6,}")

        # ── Sběr kandidátních uživatelů ────────────────────────────────────
        print(f"\n  [CF 3/5] Sběr uživatelů kteří seed tituly dokončili…")
        candidate_users = self.collect_candidate_users(seed_titles)

        if not candidate_users:
            log.warning("Žádní kandidátní uživatelé.")
            return [], []

        # ── Výpočet podobnosti ─────────────────────────────────────────────
        print(f"\n  [CF 4/5] Výpočet podobnosti ({len(candidate_users)} uživatelů)…")
        similar_users = self.find_similar_users(my_al_diffs, candidate_users)

        if not similar_users:
            log.warning("Žádní podobní uživatelé — zkus snížit min_correlation.")
            return [], []

        # ── Agregace doporučení ────────────────────────────────────────────
        print(f"\n  [CF 5/5] Agregace doporučení z {len(similar_users)} uživatelů…")
        recs = self.aggregate_recommendations(similar_users, existing_al_ids)
        log.info(f"CF doporučení: {len(recs)} titulů")

        return recs[:self.cfg.show_top * 3], similar_users


# ── Výpis výsledků ────────────────────────────────────────────────────────────

def print_cf_report(
    recommendations: list[CFRecommendation],
    similar_users:   list[SimilarUser] | None = None,
    show_top:        int = 25,
    show_users:      int = 10,
) -> None:
    W = 74
    print(f"\n{'═'*W}")
    print(f"  COLLABORATIVE FILTERING (AniList) — TOP {show_top} DOPORUČENÍ")
    print(f"{'═'*W}")
    print(f"  cf_score = AniList průměr (0–10) + vážený průměr diff podobných uživatelů")
    print(f"  Δ = odchylka od průměru komunity")
    print()

    table = []
    for rec in recommendations[:show_top]:
        delta   = rec.cf_score - rec.avg_score
        mal_str = str(rec.mal_id) if rec.mal_id else "—"
        table.append([
            f"{rec.cf_score:.2f}",
            rec.title[:42],
            f"{rec.avg_score:.2f}",
            f"{delta:+.2f}",
            rec.n_users,
            f"{rec.avg_sim:.2f}",
            mal_str,
        ])

    print(tabulate(
        table,
        headers=["CF skóre", "titul", "AL průměr", "Δ",
                 "uživatelů", "avg r", "MAL ID"],
        tablefmt="simple",
        colalign=("center","left","center","right","center","center","right"),
    ))

    if similar_users and show_users > 0:
        print(f"\n{'─'*W}")
        print(f"  TOP {show_users} PODOBNÝCH UŽIVATELŮ")
        print(f"  (Pearsonova korelace diff vektorů na AniList škále)")
        print()
        rows = [
            [f"{u.correlation:.3f}", u.username, u.overlap, f"{u.avg_diff:+.2f}"]
            for u in similar_users[:show_users]
        ]
        print(tabulate(rows,
                       headers=["r", "uživatel", "společných", "avg Δ"],
                       tablefmt="simple"))

    print(f"\n{'═'*W}\n")


def export_cf_csv(
    recommendations: list[CFRecommendation],
    path: str = "cf_recommendations.csv",
) -> None:
    import pandas as pd
    rows = [{
        "al_id":     r.al_id,
        "mal_id":    r.mal_id or "",
        "title":     r.title,
        "cf_score":  round(r.cf_score,  2),
        "avg_score": round(r.avg_score, 2),
        "delta":     round(r.cf_score - r.avg_score, 2),
        "n_users":   r.n_users,
        "avg_sim":   round(r.avg_sim,   3),
        "top_raters": ", ".join(r.top_raters),
    } for r in recommendations]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    log.info(f"CF výsledky uloženy: {path}")
