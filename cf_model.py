"""
cf_model.py — User-based Collaborative Filtering

Pipeline:
  1. Výběr seed uživatelů přes niche tituly (long-tail seeding)
     → uživatelé kteří viděli tvá oblíbená niche anime mají podobný vkus
  2. Stažení jejich animlistů z Jikan API
  3. Výpočet podobnosti přes diferenciální skóre (Pearsonova korelace)
     → filtruje bias "každý hodnotí SAO vysoko"
  4. Weighted average hodnocení podobných uživatelů pro neviděné tituly
  5. Výstup: ranked list doporučení s metadaty

Diferenciální skóre:
    diff(user, anime) = user_score - mal_avg_score
    → +1.5 znamená "hodnotím výrazně výš než komunita"
    → -1.0 znamená "hodnotím níž než komunita"

Pearsonova korelace na diff vektorech zachytí uživatele kteří mají
podobné *odchylky* od průměru — ne jen ty kteří hodnotí vše vysoko.
"""

import logging
import time
import json
import math
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
from tabulate import tabulate

log = logging.getLogger(__name__)


# ── Datové struktury ──────────────────────────────────────────────────────────

@dataclass
class CFConfig:
    """Konfigurace CF modelu — načítá se z config.yaml sekce cf."""

    # Výběr seed uživatelů
    seed_min_score:      int   = 8     # min. tvoje skóre pro seed titul
    seed_max_popularity: int   = 50000 # max. MAL members (niche filtr)
    seed_titles_count:   int   = 20    # kolik niche titulů použít jako seed
    users_per_seed:      int   = 15    # kolik uživatelů na seed titul

    # Filtrování uživatelů
    min_overlap:         int   = 15    # min. společných titulů pro korelaci
    min_correlation:     float = 0.15  # min. Pearsonova korelace
    top_k_users:         int   = 50    # počet nejpodobnějších uživatelů

    # Predikce
    min_cf_users:        int   = 3     # min. podobných uživatelů kteří titul hodnotili
    show_top:            int   = 25    # počet výsledků
    min_mal_score:       float = 6.5   # min. MAL skóre kandidáta

    # Technické
    cache_dir:           str   = "cache"
    request_delay:       float = 0.5   # sekundy mezi Jikan requesty

    @classmethod
    def from_config(cls, cfg: dict) -> "CFConfig":
        c = cfg.get("cf", {})
        return cls(
            seed_min_score=      c.get("seed_min_score",      8),
            seed_max_popularity= c.get("seed_max_popularity",  50000),
            seed_titles_count=   c.get("seed_titles_count",    20),
            users_per_seed=      c.get("users_per_seed",       15),
            min_overlap=         c.get("min_overlap",          15),
            min_correlation=     c.get("min_correlation",      0.15),
            top_k_users=         c.get("top_k_users",          50),
            min_cf_users=        c.get("min_cf_users",         3),
            show_top=            c.get("show_top",             25),
            min_mal_score=       c.get("min_mal_score",        6.5),
            cache_dir=           cfg.get("cache_dir",          "cache"),
            request_delay=       c.get("request_delay",        0.5),
        )


@dataclass
class SimilarUser:
    username:    str
    correlation: float           # Pearsonova korelace diff vektorů
    overlap:     int             # počet společných titulů
    avg_diff:    float           # průměrný diferenciál (bias korekce)


@dataclass
class CFRecommendation:
    mal_id:       int
    title:        str
    cf_score:     float          # vážený průměr diferenciálů + MAL průměr
    mal_score:    float
    n_users:      int            # počet podobných uživatelů kteří hodnotili
    avg_sim:      float          # průměrná korelace hodnotitelů
    top_raters:   list[str]      # jména nejpodobnějších uživatelů kteří hodnotili


# ── CF Engine ─────────────────────────────────────────────────────────────────

class CollaborativeFilter:
    """
    User-based collaborative filtering nad Jikan API.

    Všechna síťová volání jsou cachována — opakované spuštění
    nestahuje znovu stejná data.
    """

    def __init__(self, cfg: CFConfig, jikan_client):
        self.cfg    = cfg
        self.jikan  = jikan_client
        self._cache = Path(cfg.cache_dir) / "cf"
        self._cache.mkdir(parents=True, exist_ok=True)

    # ── Cache ──────────────────────────────────────────────────────────────────

    def _cf_cache(self, key: str) -> Path:
        safe = key.replace("/", "_").replace("?", "_").replace("&", "_")
        return self._cache / f"{safe}.json"

    def _load(self, key: str):
        f = self._cf_cache(key)
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None

    def _save(self, key: str, data) -> None:
        self._cf_cache(key).write_text(
            json.dumps(data, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8"
        )

    # ── Jikan helpers ──────────────────────────────────────────────────────────

    def _jikan_get(self, endpoint: str) -> dict | None:
        """Cachované Jikan GET — sdílí rate limiting s hlavním JikanClient."""
        cached = self._load(endpoint)
        if cached is not None:
            return cached

        result = self.jikan._get(endpoint)
        if result:
            self._save(endpoint, result)
        return result

    def _get_user_animelist(self, username: str) -> list[dict]:
        """
        Stáhne kompletní animelist uživatele (všechny stránky).
        Vrací list {mal_id, score, status}.
        Výsledek je cachován.
        """
        cache_key = f"user_animelist_{username}"
        cached = self._load(cache_key)
        if cached is not None:
            return cached

        all_entries = []
        page = 1
        while True:
            data = self._jikan_get(
                f"users/{username}/animelist?status=7&limit=300&page={page}"
            )
            if not data or not data.get("data"):
                break

            for item in data["data"]:
                ls = item.get("list_status", {})
                score = ls.get("score", 0)
                if score and score > 0:
                    all_entries.append({
                        "mal_id": item.get("mal_id"),
                        "score":  score,
                        "status": ls.get("status", ""),
                    })

            if not data.get("pagination", {}).get("has_next_page"):
                break
            page += 1
            time.sleep(self.cfg.request_delay)

        self._save(cache_key, all_entries)
        return all_entries

    def _get_anime_userupdates(self, mal_id: int, pages: int = 2) -> list[str]:
        """
        Vrátí seznam uživatelských jmen kteří nedávno aktualizovali
        daný titul (endpoint /anime/{id}/userupdates).

        Jde o proxy pro "kdo tento titul sledoval/dokončil".
        """
        cache_key = f"userupdates_{mal_id}_p{pages}"
        cached = self._load(cache_key)
        if cached is not None:
            return cached

        usernames = []
        for page in range(1, pages + 1):
            data = self._jikan_get(f"anime/{mal_id}/userupdates?page={page}")
            if not data or not data.get("data"):
                break
            for entry in data["data"]:
                user = entry.get("user", {})
                name = user.get("username")
                if name:
                    usernames.append(name)
            if not data.get("pagination", {}).get("has_next_page"):
                break
            time.sleep(self.cfg.request_delay)

        self._save(cache_key, usernames)
        return usernames

    # ── Výběr seed uživatelů ──────────────────────────────────────────────────

    def select_seed_titles(
        self,
        scored_entries: list,      # list[MalEntry] — tvoje hodnocené tituly
        jikan_data:     dict,      # {mal_id: jikan_data} — pro members count
    ) -> list[tuple[int, str, int, int]]:
        """
        Vybere niche tituly pro seed výběr uživatelů.

        Kritéria (z CFConfig):
          - tvoje skóre >= seed_min_score
          - MAL members <= seed_max_popularity (niche filtr)

        Seřazeno: nejnišovější tituly s vysokým hodnocením první
        (maximalizuje specifičnost seedu).

        Vrací list (mal_id, title, user_score, members) sestupně dle specifičnosti.
        """
        cfg = self.cfg
        candidates = []

        for entry in scored_entries:
            if entry.score < cfg.seed_min_score:
                continue
            data     = jikan_data.get(entry.mal_id) or {}
            members  = data.get("members") or 0
            if members == 0 or members > cfg.seed_max_popularity:
                continue
            candidates.append((entry.mal_id, entry.title, entry.score, members))

        # Seřaď: nejnišovější první (nízký members = vysoká specifičnost)
        candidates.sort(key=lambda x: x[3])
        selected = candidates[:cfg.seed_titles_count]

        log.info(
            f"Seed tituly: {len(selected)} niche anime "
            f"(skóre ≥{cfg.seed_min_score}, members ≤{cfg.seed_max_popularity:,})"
        )
        return selected

    def collect_seed_users(
        self,
        seed_titles: list[tuple[int, str, int, int]],
    ) -> list[str]:
        """
        Shromáždí seznam candidate uživatelů z userupdates seed titulů.
        Deduplikuje a vrátí unikátní seznam.
        """
        cfg        = self.cfg
        all_users: set[str] = set()
        pages_needed = max(1, math.ceil(cfg.users_per_seed / 75))

        for mal_id, title, score, members in seed_titles:
            print(f"  Seed: {title[:45]} (skóre {score}, {members:,} members)…",
                  end="\r")
            users = self._get_anime_userupdates(mal_id, pages=pages_needed)
            all_users.update(users[:cfg.users_per_seed * 2])

        print(f"\n  Seed uživatelé: {len(all_users)} unikátních uživatelů nalezeno")
        return list(all_users)

    # ── Výpočet podobnosti ────────────────────────────────────────────────────

    def compute_similarity(
        self,
        my_ratings:      dict[int, int],    # {mal_id: score}
        my_mal_scores:   dict[int, float],  # {mal_id: mal_avg_score}
        other_ratings:   list[dict],        # [{mal_id, score}]
        other_mal_scores: dict[int, float], # {mal_id: mal_avg_score} (sdílený)
    ) -> tuple[float, int, float] | None:
        """
        Spočítá Pearsonovu korelaci diferenciálních skóre.

        Diferenciál: user_score - mal_avg_score
        → odstraní bias způsobený tím, že oblíbená anime dostávají vysoká
          skóre od všech bez ohledu na specifický vkus

        Vrací (korelace, overlap, avg_diff_other) nebo None pokud
        není dostatečný překryv.
        """
        cfg = self.cfg

        # Sestav slovník pro druhého uživatele
        other_dict = {
            e["mal_id"]: e["score"]
            for e in other_ratings
            if e.get("mal_id") and e.get("score", 0) > 0
        }

        # Najdi společné tituly
        common = set(my_ratings.keys()) & set(other_dict.keys())
        # Filtruj na tituly kde máme MAL průměr
        common = {mid for mid in common if my_mal_scores.get(mid, 0) > 0}

        if len(common) < cfg.min_overlap:
            return None

        # Diferenciální vektory
        my_diffs    = np.array([my_ratings[m]    - my_mal_scores[m]  for m in common])
        other_diffs = np.array([other_dict[m]    - my_mal_scores.get(m, other_dict[m]) for m in common])

        # Pearsonova korelace
        my_mean    = my_diffs.mean()
        other_mean = other_diffs.mean()
        my_c       = my_diffs    - my_mean
        other_c    = other_diffs - other_mean

        denom = np.sqrt((my_c**2).sum() * (other_c**2).sum())
        if denom < 1e-9:
            return None

        corr     = float(np.dot(my_c, other_c) / denom)
        avg_diff = float(other_diffs.mean())

        if corr < cfg.min_correlation:
            return None

        return corr, len(common), avg_diff

    # ── Hlavní pipeline ────────────────────────────────────────────────────────

    def run(
        self,
        my_entries:     list,       # list[MalEntry] — tvoje hodnocené tituly
        jikan_data:     dict,       # {mal_id: jikan_data} — pro members + MAL score
        existing_ids:   set[int],   # tituly které už máš na MAL
        titles_map:     dict[int, str],
    ) -> list[CFRecommendation]:
        """
        Spustí celou CF pipeline a vrátí seřazený list doporučení.
        """
        cfg = self.cfg

        # ── Příprava tvých dat ─────────────────────────────────────────────
        my_ratings: dict[int, int] = {
            e.mal_id: e.score
            for e in my_entries
            if e.score > 0
        }
        my_mal_scores: dict[int, float] = {
            mid: float((jikan_data.get(mid) or {}).get("score") or 0)
            for mid in my_ratings
        }

        # ── Výběr seed titulů ──────────────────────────────────────────────
        print("\n  [CF 1/5] Výběr niche seed titulů…")
        seed_titles = self.select_seed_titles(my_entries, jikan_data)

        if not seed_titles:
            log.warning("Žádné seed tituly nenalezeny — uvolni seed_max_popularity v config.")
            return []

        print(f"\n  Top seed tituly:")
        for mid, title, score, members in seed_titles[:8]:
            print(f"    {score}★  {members:>6,} members  {title[:45]}")

        # ── Sběr candidate uživatelů ───────────────────────────────────────
        print(f"\n  [CF 2/5] Sběr uživatelů z {len(seed_titles)} seed titulů…")
        candidate_users = self.collect_seed_users(seed_titles)

        if not candidate_users:
            log.warning("Žádní candidate uživatelé — zkus zvýšit users_per_seed.")
            return []

        # ── Výpočet podobnosti ─────────────────────────────────────────────
        print(f"\n  [CF 3/5] Výpočet podobnosti pro {len(candidate_users)} uživatelů…")
        similar_users: list[SimilarUser] = []
        processed = 0

        for username in candidate_users:
            processed += 1
            if processed % 20 == 0:
                print(f"  Zpracováno: {processed}/{len(candidate_users)}, "
                      f"podobných zatím: {len(similar_users)}…", end="\r")

            try:
                other_ratings = self._get_user_animelist(username)
            except Exception as e:
                log.debug(f"Chyba pro uživatele {username}: {e}")
                continue

            result = self.compute_similarity(
                my_ratings, my_mal_scores,
                other_ratings, my_mal_scores,
            )
            if result is None:
                continue

            corr, overlap, avg_diff = result
            similar_users.append(SimilarUser(
                username    = username,
                correlation = corr,
                overlap     = overlap,
                avg_diff    = avg_diff,
            ))

        # Seřaď a ořež na top_k
        similar_users.sort(key=lambda u: -u.correlation)
        similar_users = similar_users[:cfg.top_k_users]

        print(f"\n  Podobní uživatelé: {len(similar_users)} "
              f"(z {len(candidate_users)} kandidátů, "
              f"práh r≥{cfg.min_correlation}, overlap≥{cfg.min_overlap})")

        if not similar_users:
            log.warning("Žádní dostatečně podobní uživatelé.")
            return []

        # ── Agregace hodnocení ─────────────────────────────────────────────
        print(f"\n  [CF 4/5] Agregace hodnocení podobných uživatelů…")

        # Pro každé neviděné anime shromáždi (weighted_diff, weight, username)
        item_data: dict[int, list[tuple[float, float, str]]] = defaultdict(list)

        for user in similar_users:
            weight = user.correlation  # váha = korelační koeficient
            try:
                ratings = self._get_user_animelist(user.username)
            except Exception:
                continue

            for entry in ratings:
                mid   = entry.get("mal_id")
                score = entry.get("score", 0)
                if not mid or not score or mid in existing_ids:
                    continue
                mal_avg = float((jikan_data.get(mid) or {}).get("score") or 0)
                if mal_avg < cfg.min_mal_score:
                    continue
                diff = score - mal_avg if mal_avg > 0 else 0.0
                item_data[mid].append((diff, weight, user.username))

        # ── Sestavení doporučení ───────────────────────────────────────────
        print(f"\n  [CF 5/5] Sestavení doporučení…")
        recommendations: list[CFRecommendation] = []

        for mal_id, entries_cf in item_data.items():
            if len(entries_cf) < cfg.min_cf_users:
                continue

            mal_avg = float((jikan_data.get(mal_id) or {}).get("score") or 0)
            if mal_avg < cfg.min_mal_score:
                continue

            # Weighted average diferenciálu
            total_weight = sum(w for _, w, _ in entries_cf)
            if total_weight < 1e-9:
                continue

            weighted_diff = sum(d * w for d, w, _ in entries_cf) / total_weight
            cf_score      = float(np.clip(mal_avg + weighted_diff, 1, 10))
            avg_sim       = float(np.mean([w for _, w, _ in entries_cf]))

            # Top raters (nejpodobnější uživatelé kteří to hodnotili)
            top_raters = [
                u for _, _, u in sorted(entries_cf, key=lambda x: -x[1])[:5]
            ]

            title = (
                titles_map.get(mal_id)
                or (jikan_data.get(mal_id) or {}).get("title", f"ID:{mal_id}")
            )

            recommendations.append(CFRecommendation(
                mal_id     = mal_id,
                title      = title,
                cf_score   = cf_score,
                mal_score  = mal_avg,
                n_users    = len(entries_cf),
                avg_sim    = avg_sim,
                top_raters = top_raters,
            ))

        recommendations.sort(key=lambda r: -r.cf_score)
        log.info(f"CF doporučení: {len(recommendations)} titulů")
        return recommendations[:cfg.show_top * 3]  # rezerva pro filtrování


# ── Výpis výsledků ────────────────────────────────────────────────────────────

def print_cf_report(
    recommendations: list[CFRecommendation],
    similar_users:   list | None = None,
    show_top:        int = 25,
    show_users:      int = 10,
) -> None:
    """Vypíše CF report na stdout."""
    W = 72
    print(f"\n{'═'*W}")
    print(f"  COLLABORATIVE FILTERING — TOP {show_top} DOPORUČENÍ")
    print(f"{'═'*W}")
    print(f"  cf_score = MAL průměr + vážený průměr diferenciálů podobných uživatelů")
    print(f"  Δ = odchylka od MAL průměru (jak moc podobní uživatelé nad/podhodnocují)")
    print()

    table = []
    for rec in recommendations[:show_top]:
        delta = rec.cf_score - rec.mal_score
        table.append([
            f"{rec.cf_score:.2f}",
            rec.title[:42],
            f"{rec.mal_score:.2f}",
            f"{delta:+.2f}",
            rec.n_users,
            f"{rec.avg_sim:.2f}",
        ])

    print(tabulate(
        table,
        headers=["CF skóre", "titul", "MAL", "Δ", "uživatelů", "avg r"],
        tablefmt="simple",
        colalign=("center", "left", "center", "right", "center", "center"),
    ))

    if similar_users and show_users > 0:
        print(f"\n{'─'*W}")
        print(f"  TOP {show_users} PODOBNÝCH UŽIVATELŮ")
        print(f"  (Pearsonova korelace diferenciálních skóre)")
        print()
        u_rows = [
            [f"{u.correlation:.3f}", u.username, u.overlap, f"{u.avg_diff:+.2f}"]
            for u in similar_users[:show_users]
        ]
        print(tabulate(
            u_rows,
            headers=["r", "uživatel", "společných", "avg Δ"],
            tablefmt="simple",
        ))

    print(f"\n{'═'*W}\n")


def export_cf_csv(
    recommendations: list[CFRecommendation],
    path: str = "cf_recommendations.csv",
) -> None:
    """Uloží CF doporučení do CSV."""
    import pandas as pd
    rows = [
        {
            "mal_id":    r.mal_id,
            "title":     r.title,
            "cf_score":  round(r.cf_score,  2),
            "mal_score": round(r.mal_score, 2),
            "delta":     round(r.cf_score - r.mal_score, 2),
            "n_users":   r.n_users,
            "avg_sim":   round(r.avg_sim,   3),
            "top_raters": ", ".join(r.top_raters),
        }
        for r in recommendations
    ]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    log.info(f"CF výsledky uloženy: {path}")
