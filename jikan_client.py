"""
jikan_client.py — Jikan API v4 klient s cachováním a rate limitingem

Jikan je neoficiální REST API pro MyAnimeList.
Dokumentace: https://docs.api.jikan.moe/
Rate limit: ~3 requesty/sekundu (klient automaticky čeká)
"""

import json
import time
import logging
from pathlib import Path
import requests

log = logging.getLogger(__name__)

BASE_URL = "https://api.jikan.moe/v4"
REQUEST_DELAY = 0.4          # sekundy mezi requesty (bezpečný interval)
MAX_RETRIES   = 4
RETRY_DELAYS  = [2, 5, 10, 30]  # exponenciální backoff při 429


class JikanClient:
    def __init__(self, cache_dir: str = "cache"):
        self.cache_path = Path(cache_dir)
        self.cache_path.mkdir(exist_ok=True)
        self._last_request = 0.0
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "anime-taste-model/1.0"

    # ── Interní helpers ────────────────────────────────────────────

    def _cache_file(self, key: str) -> Path:
        safe = key.replace("/", "_").replace("?", "_")
        return self.cache_path / f"{safe}.json"

    def _load_cache(self, key: str):
        f = self._cache_file(key)
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))
        return None

    def _save_cache(self, key: str, data) -> None:
        f = self._cache_file(key)
        f.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _get(self, endpoint: str) -> dict:
        """Provede GET request s cachováním, rate limitingem a retry logikou."""
        cached = self._load_cache(endpoint)
        if cached is not None:
            return cached

        # Rate limiting
        elapsed = time.time() - self._last_request
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)

        url = f"{BASE_URL}/{endpoint}"
        for attempt, delay in enumerate(RETRY_DELAYS + [None]):
            try:
                resp = self.session.get(url, timeout=15)
                self._last_request = time.time()

                if resp.status_code == 429:
                    wait = RETRY_DELAYS[attempt] if attempt < len(RETRY_DELAYS) else 60
                    log.warning(f"Rate limit 429, čekám {wait}s…")
                    time.sleep(wait)
                    continue

                if resp.status_code == 404:
                    # Anime neexistuje nebo je NSFW — uložíme None do cache
                    self._save_cache(endpoint, None)
                    return None

                resp.raise_for_status()
                data = resp.json()
                self._save_cache(endpoint, data)
                return data

            except requests.RequestException as e:
                if attempt < len(RETRY_DELAYS) - 1:
                    log.warning(f"Chyba {e}, retry za {RETRY_DELAYS[attempt]}s…")
                    time.sleep(RETRY_DELAYS[attempt])
                else:
                    log.error(f"Selhalo po {MAX_RETRIES} pokusech: {url}")
                    return None

        return None

    # ── Veřejné metody ─────────────────────────────────────────────

    def get_anime(self, mal_id: int) -> dict | None:
        """
        Vrátí detailní informace o anime dle MAL ID.

        Vrací klíčové pole 'data' s atributy:
            title, type, source, episodes, score, year,
            genres, themes, demographics, studios
        """
        result = self._get(f"anime/{mal_id}/full")
        if result and "data" in result:
            return result["data"]
        return None

    def get_anime_staff(self, mal_id: int) -> list[dict]:
        """
        Vrátí staff pro anime dle MAL ID (samostatný endpoint /anime/{id}/staff).

        Vrací list objektů:
            [{"person": {"mal_id": ..., "name": ...}, "positions": [...]}, ...]

        Cachováno odděleně od /full dat pod klíčem "staff_{mal_id}".
        """
        result = self._get(f"anime/{mal_id}/staff")
        if result and "data" in result:
            return result["data"]
        return []

    def get_staff_batch(
        self,
        mal_ids: list[int],
        show_progress: bool = True,
    ) -> dict[int, list[dict]]:
        """
        Stáhne staff data pro seznam MAL ID.
        Vrací dict {mal_id: [staff_entries]}.
        """
        results = {}
        total   = len(mal_ids)

        for i, mal_id in enumerate(mal_ids):
            if show_progress and i % 10 == 0:
                print(f"  Staff data: {i}/{total}…", end="\r")
            staff = self.get_anime_staff(int(mal_id))
            results[mal_id] = staff  # může být prázdný list

        if show_progress:
            non_empty = sum(1 for v in results.values() if v)
            print(f"  Staff stažen: {non_empty}/{total} titulů s daty.          ")

        return results

    def get_top_anime(self, limit: int = 100, min_score: float = 7.0) -> list[dict]:
        """
        Stáhne top anime z MAL seřazené podle skóre.
        Filtruje na min_score. Vrací list anime data objektů.
        """
        results = []
        page = 1
        per_page = 25  # Jikan maximum

        while len(results) < limit:
            data = self._get(f"top/anime?page={page}&type=tv")
            if not data or "data" not in data:
                break

            for item in data["data"]:
                if item.get("score", 0) < min_score:
                    # Top anime jsou seřazené — pod min_score už nic nepřijde
                    return results
                results.append(item)
                if len(results) >= limit:
                    break

            if not data.get("pagination", {}).get("has_next_page"):
                break
            page += 1
            time.sleep(REQUEST_DELAY)

        return results

    def list_all_staff(
        self,
        mal_ids: list[int],
        show_progress: bool = False,
    ) -> dict[str, list[tuple[int, str, str, int]]]:
        """
        Projde Jikan data pro zadaná MAL ID a vrátí frekvenční přehled
        režisérů a scenáristů ve formátu vhodném pro config.yaml.

        Vrací:
            {
              "directors": [(mal_id, name, position, count), ...],
              "writers":   [(mal_id, name, position, count), ...],
            }
        Seřazeno sestupně dle počtu titulů.
        """
        from collections import defaultdict

        DIRECTOR_POSITIONS = {"director", "series director"}
        WRITER_POSITIONS   = {"script", "series composition", "screenplay",
                              "original creator", "original story"}

        directors: dict[int, list] = defaultdict(lambda: ["", "", 0])
        writers:   dict[int, list] = defaultdict(lambda: ["", "", 0])

        staff_data = self.get_staff_batch(mal_ids, show_progress=show_progress)
        for staff_list in staff_data.values():
            for entry in staff_list:
                person    = entry.get("person") or {}
                person_id = person.get("mal_id")
                name      = person.get("name", "")
                if not person_id:
                    continue
                positions = {p.lower() for p in (entry.get("positions") or [])}
                if positions & DIRECTOR_POSITIONS:
                    pos_str = next(iter(positions & DIRECTOR_POSITIONS))
                    directors[person_id][0] = name
                    directors[person_id][1] = pos_str
                    directors[person_id][2] += 1
                if positions & WRITER_POSITIONS:
                    pos_str = next(iter(positions & WRITER_POSITIONS))
                    writers[person_id][0] = name
                    writers[person_id][1] = pos_str
                    writers[person_id][2] += 1

        def to_list(d):
            return sorted(
                [(pid, info[0], info[1], info[2]) for pid, info in d.items()],
                key=lambda x: -x[3]
            )

        return {"directors": to_list(directors), "writers": to_list(writers)}

    def list_mal_features(
        self,
        mal_ids: list[int],
    ) -> dict[str, dict]:
        """
        Projde Jikan data pro zadaná MAL ID a vrátí frekvenční přehledy
        pro genres, themes, demographics, sources a types.

        Vrací:
            {
              "genres":       {(mal_id, name): count},
              "themes":       {(mal_id, name): count},
              "demographics": {name: count},
              "sources":      {source: count},
              "types":        {type: count},
            }
        """
        from collections import Counter

        genres       = Counter()  # (mal_id, name) → count
        themes       = Counter()
        demographics = Counter()  # name → count
        sources      = Counter()
        types        = Counter()

        data = self.get_anime_batch(mal_ids, show_progress=False)
        for anime in data.values():
            for g in anime.get("genres", []):
                genres[(g["mal_id"], g["name"])] += 1
            for t in anime.get("themes", []):
                themes[(t["mal_id"], t["name"])] += 1
            for d in anime.get("demographics", []):
                demographics[d["name"]] += 1
            src = (anime.get("source") or "").strip()
            if src:
                sources[src] += 1
            typ = (anime.get("type") or "").strip()
            if typ:
                types[typ] += 1

        return {
            "genres":       dict(genres),
            "themes":       dict(themes),
            "demographics": dict(demographics),
            "sources":      dict(sources),
            "types":        dict(types),
        }

    def get_anime_batch(
        self,
        mal_ids: list[int],
        show_progress: bool = True
    ) -> dict[int, dict]:
        """
        Stáhne informace pro seznam MAL ID.
        Vrací dict {mal_id: anime_data}.
        Přeskočí ID, která nelze stáhnout.
        """
        results = {}
        total = len(mal_ids)

        for i, mal_id in enumerate(mal_ids):
            if show_progress and i % 10 == 0:
                print(f"  Stahuji data: {i}/{total} ({i/total*100:.0f}%)…", end="\r")

            data = self.get_anime(int(mal_id))
            if data:
                results[mal_id] = data

        if show_progress:
            print(f"  Staženo: {len(results)}/{total} titulů.          ")

        return results
