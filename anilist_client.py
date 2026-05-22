"""
anilist_client.py — AniList GraphQL klient s cachováním a rate limitingem

AniList API: https://anilist.gitbook.io/anilist-apiv2-docs/
GraphQL endpoint: https://graphql.anilist.co
Rate limit: 90 requestů/minutu (klient automaticky čeká při 429)

Klíčová výhoda oproti MAL: 500+ granulárních tagů včetně archetypů
postav (Tsundere, Kuudere), narativních vzorů (Love Triangle, Slow
Romance) a tematických kategorií (Tearjerker, Philosophy).
Každý tag má rank 0–100 (jak dominantní je v daném titulu).
"""

import json
import time
import logging
from pathlib import Path

import requests

log = logging.getLogger(__name__)

GRAPHQL_URL   = "https://graphql.anilist.co"
REQUEST_DELAY = 0.7          # sekundy mezi requesty (konzervativní)
RETRY_DELAYS  = [5, 15, 60]  # backoff při 429

# GraphQL dotaz — stáhne tagy, studia a základní metadata přes MAL ID
QUERY_BY_MAL_ID = """
query ($idMal: Int) {
  Media(idMal: $idMal, type: ANIME) {
    id
    idMal
    title { romaji english }
    tags {
      name
      rank
      isAdult
      isGeneralSpoiler
      isMediaSpoiler
      category
    }
    studios {
      nodes { name isAnimationStudio }
    }
    format
    source
    averageScore
    popularity
    favourites
  }
}
"""

# GraphQL dotaz pro hromadné stažení (až 50 ID najednou)
# Anilist nemá batch endpoint pro arbitrary IDs, ale
# můžeme stránkovat přes idMal_in
QUERY_BATCH = """
query ($ids: [Int]) {
  Page(perPage: 50) {
    media(idMal_in: $ids, type: ANIME) {
      id
      idMal
      title { romaji english }
      tags {
        name
        rank
        isAdult
        isGeneralSpoiler
        isMediaSpoiler
        category
      }
      studios {
        nodes { name isAnimationStudio }
      }
      format
      source
      averageScore
      popularity
      favourites
    }
  }
}
"""


class AniListClient:
    def __init__(self, cache_dir: str = "cache"):
        self.cache_path = Path(cache_dir) / "anilist"
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self._last_request = 0.0
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept":       "application/json",
            "User-Agent":   "anime-taste-model/1.0",
        })

    # ── Cache helpers ──────────────────────────────────────────────────────────

    def _cache_file(self, mal_id: int) -> Path:
        return self.cache_path / f"mal_{mal_id}.json"

    def _load_cache(self, mal_id: int):
        f = self._cache_file(mal_id)
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))
        return None        # None = not cached yet; {} = cached but not found

    def _save_cache(self, mal_id: int, data) -> None:
        f = self._cache_file(mal_id)
        f.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── HTTP helpers ───────────────────────────────────────────────────────────

    def _post(self, query: str, variables: dict) -> dict | None:
        """Provede GraphQL POST request s rate limitingem a retry logikou."""
        elapsed = time.time() - self._last_request
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)

        payload = {"query": query, "variables": variables}

        for attempt, delay in enumerate(RETRY_DELAYS + [None]):
            try:
                resp = self.session.post(GRAPHQL_URL, json=payload, timeout=20)
                self._last_request = time.time()

                if resp.status_code == 429:
                    # AniList vrátí Retry-After header
                    wait = int(resp.headers.get("Retry-After", delay or 60))
                    log.warning(f"Rate limit 429, čekám {wait}s…")
                    time.sleep(wait)
                    continue

                if resp.status_code == 404:
                    return None

                resp.raise_for_status()
                data = resp.json()

                # GraphQL vrátí errors pole i při HTTP 200
                if "errors" in data:
                    for err in data["errors"]:
                        if err.get("status") == 404:
                            return None
                        log.warning(f"GraphQL error: {err.get('message')}")
                    return None

                return data

            except requests.RequestException as e:
                if attempt < len(RETRY_DELAYS) - 1:
                    log.warning(f"Request chyba ({e}), retry za {delay}s…")
                    time.sleep(delay)
                else:
                    log.error(f"AniList request selhal: {e}")
                    return None

        return None

    # ── Veřejné metody ─────────────────────────────────────────────────────────

    def get_anime(self, mal_id: int) -> dict | None:
        """
        Vrátí AniList data pro anime dle MAL ID.
        Výsledek je cachován — opakované volání je okamžité.

        Vrátí None pokud anime na AniList neexistuje.
        Vrátí {} (prázdný dict) pokud byl uložen jako nenalezený.
        """
        cached = self._load_cache(mal_id)
        if cached is not None:
            return cached if cached else None   # {} → None

        result = self._post(QUERY_BY_MAL_ID, {"idMal": mal_id})
        if result and result.get("data", {}).get("Media"):
            data = result["data"]["Media"]
            self._save_cache(mal_id, data)
            return data
        else:
            # Nenalezeno — ulož prázdný dict jako sentinel
            self._save_cache(mal_id, {})
            log.debug(f"AniList: MAL ID {mal_id} nenalezeno")
            return None

    def get_anime_batch(
        self,
        mal_ids: list[int],
        show_progress: bool = True,
    ) -> dict[int, dict]:
        """
        Stáhne AniList data pro seznam MAL ID.

        Nejprve zkusí cache, nekešované stáhne po dávkách 50 kusů
        (AniList Page query je efektivnější než jednotlivé requesty).

        Vrací dict {mal_id: anilist_data}.
        """
        results   = {}
        uncached  = []

        # 1. Načti z cache
        for mal_id in mal_ids:
            cached = self._load_cache(mal_id)
            if cached is not None:
                if cached:   # neprázdný = platná data
                    results[mal_id] = cached
                # prázdný = nenalezeno, přeskočíme
            else:
                uncached.append(mal_id)

        if show_progress and uncached:
            print(f"  AniList cache: {len(results)} hit, {len(uncached)} ke stažení…")

        if not uncached:
            return results

        # 2. Stáhni nekešované po dávkách 50
        batch_size = 50
        batches    = [uncached[i:i+batch_size] for i in range(0, len(uncached), batch_size)]

        for b_idx, batch in enumerate(batches):
            if show_progress:
                done = b_idx * batch_size
                print(
                    f"  AniList stahování: {done}/{len(uncached)}…",
                    end="\r"
                )

            result = self._post(QUERY_BATCH, {"ids": batch})
            if not result:
                # Fallback: stáhni jeden po jednom
                log.warning("Batch query selhala, přepínám na jednotlivé requesty…")
                for mal_id in batch:
                    data = self.get_anime(mal_id)
                    if data:
                        results[mal_id] = data
                continue

            media_list = result.get("data", {}).get("Page", {}).get("media", [])

            # Ulož do cache i results
            fetched_mal_ids = set()
            for media in media_list:
                mid = media.get("idMal")
                if mid:
                    self._save_cache(mid, media)
                    results[mid] = media
                    fetched_mal_ids.add(mid)

            # Ulož sentinel pro nenalezená
            for mal_id in batch:
                if mal_id not in fetched_mal_ids:
                    self._save_cache(mal_id, {})

        if show_progress and uncached:
            print(f"  AniList staženo: {len(results)}/{len(mal_ids)} titulů.          ")

        return results

    # ── Utilitní metody ────────────────────────────────────────────────────────

    @staticmethod
    def extract_tags(
        anilist_data: dict,
        exclude_adult:   bool = True,
        exclude_spoiler: bool = True,
        min_rank:        int  = 0,
    ) -> dict[str, float]:
        """
        Extrahuje tagy z AniList dat jako dict {tag_name: rank_0_to_1}.

        Parametry:
            exclude_adult   — vynech adult tagy (isAdult=True)
            exclude_spoiler — vynech spoilerové tagy
            min_rank        — minimální rank pro zahrnutí (0–100)

        Vrací rank normalizovaný na 0–1.
        """
        tags = {}
        for tag in anilist_data.get("tags", []):
            if exclude_adult   and tag.get("isAdult"):
                continue
            if exclude_spoiler and (
                tag.get("isGeneralSpoiler") or tag.get("isMediaSpoiler")
            ):
                continue
            rank = tag.get("rank", 0) or 0
            if rank >= min_rank:
                tags[tag["name"]] = rank / 100.0
        return tags

    @staticmethod
    def extract_animation_studios(anilist_data: dict) -> list[str]:
        """Vrátí seznam animačních studií (isAnimationStudio=True)."""
        studios = []
        for node in anilist_data.get("studios", {}).get("nodes", []):
            if node.get("isAnimationStudio"):
                studios.append(node["name"])
        return studios

    def list_all_studios(self, mal_ids: list[int]) -> dict[str, int]:
        """
        Vrátí frekvenční slovník animačních studií pro zadaná MAL ID.
        Počítá počet unikátních titulů (ne výskytů) — ochrana před duplicitami.
        """
        from collections import Counter
        counter: Counter = Counter()

        data = self.get_anime_batch(mal_ids, show_progress=False)
        for anilist_data in data.values():
            seen: set[str] = set()
            for node in anilist_data.get("studios", {}).get("nodes", []):
                if node.get("isAnimationStudio"):
                    name = node["name"]
                    if name not in seen:
                        counter[name] += 1
                        seen.add(name)

        return dict(counter)

    def list_all_tags(self, mal_ids: list[int]) -> dict[str, int]:
        """
        Projde data pro zadaná MAL ID a vrátí frekvenční slovník
        všech tagů. Počítá počet *unikátních titulů* s daným tagem
        (ne surový počet výskytů — chrání před duplikáty v API datech).

        Použití:
            tags = client.list_all_tags(scored_mal_ids)
            for tag, count in sorted(tags.items(), key=lambda x: -x[1])[:50]:
                print(f"{count:4d}  {tag}")
        """
        from collections import Counter
        counter: Counter = Counter()

        data = self.get_anime_batch(mal_ids, show_progress=False)
        for anilist_data in data.values():
            # Deduplikuj tagy v rámci jednoho titulu před počítáním
            seen_in_this_anime: set[str] = set()
            for tag in anilist_data.get("tags", []):
                if tag.get("isAdult") or tag.get("isGeneralSpoiler"):
                    continue
                name = tag["name"]
                if name not in seen_in_this_anime:
                    counter[name] += 1
                    seen_in_this_anime.add(name)

        return dict(counter)
