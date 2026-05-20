"""
series_aggregator.py — Agregace sérií přes sequel/prequel vazby

Problém: MAL a Jikan vidí každou řadu série jako samostatný titul.
Uživatel ale hodnotí sérii jako celek — a hodnocení jednotlivých řad
koreluje silně navzájem (umělá inflace trénovacích dat).

Řešení: Union-Find (disjoint-set) přes Sequel/Prequel vazby z Jikan.
Každá connected component = jedna série → zachováme záznam s MAX skóre.

Příklad:
  SAO S1 (score 9) + SAO S2 (score 7) + SAO Alicization (score 9)
  → jedna série s max skóre 9, příznaky z S1 (jako zástupce)

Zástupce série = entry s nejvyšším skóre (tie-break: více epizod).
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)


# ── Union-Find ────────────────────────────────────────────────────────────────

class UnionFind:
    """Jednoduchá union-find datová struktura s path compressionem."""

    def __init__(self):
        self._parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        if x not in self._parent:
            self._parent[x] = x
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])  # path compression
        return self._parent[x]

    def union(self, x: int, y: int) -> None:
        px, py = self.find(x), self.find(y)
        if px != py:
            self._parent[px] = py

    def components(self, ids: list[int]) -> dict[int, list[int]]:
        """Vrátí dict {root: [members]} pro zadaná ID."""
        result: dict[int, list[int]] = {}
        for id_ in ids:
            root = self.find(id_)
            result.setdefault(root, []).append(id_)
        return result


# ── Hlavní funkce ─────────────────────────────────────────────────────────────

# Typy vazeb z Jikan, které spojujeme do série
SERIES_RELATION_TYPES = {
    "sequel",
    "prequel",
    "alternative version",
    "side story",        # volitelné — některé side story jsou samostatné
}

# Typy vazeb které NESPOJUJEME (jiný příběh/vesmír)
SKIP_RELATION_TYPES = {
    "adaptation",
    "character",
    "other",
    "summary",
}


def build_series_groups(
    mal_ids:    list[int],
    jikan_data: dict,          # {mal_id: jikan_full_data}
    relation_types: set[str] = SERIES_RELATION_TYPES,
) -> dict[int, list[int]]:
    """
    Sestaví skupiny sérií přes Sequel/Prequel vazby.

    Vstup:
        mal_ids       — seznam MAL ID k seskupení
        jikan_data    — Jikan /anime/{id}/full data (musí obsahovat 'relations')
        relation_types — typy vazeb které spojují do série

    Výstup:
        dict {root_mal_id: [member_mal_ids]}
        Každá skupina = jedna série. Singleton = standalone titul.
    """
    id_set = set(mal_ids)
    uf     = UnionFind()

    # Inicializuj všechna ID
    for mid in mal_ids:
        uf.find(mid)

    # Projdi relace a spoj příbuzné tituly
    linked = 0
    for mal_id in mal_ids:
        data = jikan_data.get(mal_id)
        if not data:
            continue

        for relation in data.get("relations", []):
            rel_type = (relation.get("relation") or "").lower()
            if rel_type not in relation_types:
                continue

            for entry in relation.get("entry", []):
                if entry.get("type") != "anime":
                    continue
                related_id = entry.get("mal_id")
                # Spoj pouze pokud je related_id také v našem datasetu
                if related_id and related_id in id_set:
                    if uf.find(mal_id) != uf.find(related_id):
                        uf.union(mal_id, related_id)
                        linked += 1

    groups = uf.components(mal_ids)
    singletons = sum(1 for g in groups.values() if len(g) == 1)
    series     = len(groups) - singletons

    log.info(
        f"Série: {series} skupin z {len(mal_ids)} titulů "
        f"({linked} vazeb), {singletons} standalone titulů"
    )
    return groups


def aggregate_entries(
    entries:    list,           # list[MalEntry]
    jikan_data: dict,
    relation_types: set[str] = SERIES_RELATION_TYPES,
) -> list:
    """
    Kolapsuje trénovací záznamy na úrovni sérií.

    Pro každou skupinu (sérii):
      - score    = maximum přes všechny záznamy skupiny
      - zástupce = záznam s nejvyšším skóre (tie-break: více epizod)
        → příznaky se počítají z dat zástupce

    Vrací nový list MalEntry záznamů (jeden per série/standalone).
    """
    from mal_parser import MalEntry

    mal_ids   = [e.mal_id for e in entries]
    entry_map = {e.mal_id: e for e in entries}

    groups = build_series_groups(mal_ids, jikan_data)

    aggregated = []
    for root, members in groups.items():
        if len(members) == 1:
            aggregated.append(entry_map[members[0]])
            continue

        # Najdi zástupce: nejvyšší skóre, tie-break: více epizod
        group_entries = [entry_map[m] for m in members if m in entry_map]
        if not group_entries:
            continue

        max_score    = max(e.score for e in group_entries)
        # Zástupce = ten s nejvyšším skóre (a nejvíce epizodami při shodě)
        representative = max(
            group_entries,
            key=lambda e: (e.score, e.episodes)
        )

        # Vytvoř agregovaný záznam se zachovanými daty zástupce
        # ale score = maximum přes skupinu
        agg = MalEntry(
            mal_id=          representative.mal_id,
            title=           _series_title(group_entries, jikan_data),
            type=            representative.type,
            episodes=        sum(e.episodes for e in group_entries),
            watched_episodes=sum(e.watched_episodes for e in group_entries),
            score=           max_score,
            status=          representative.status,
            start_date=      representative.start_date,
            finish_date=     representative.finish_date,
            rewatched=       representative.rewatched,
        )
        aggregated.append(agg)

        titles = [e.title for e in group_entries]
        log.debug(
            f"Série [{', '.join(titles[:3])}{'…' if len(titles)>3 else ''}] "
            f"→ score {[e.score for e in group_entries]} → max {max_score}"
        )

    before = len(entries)
    after  = len(aggregated)
    log.info(
        f"Agregace sérií: {before} záznamů → {after} "
        f"(redukce o {before-after}, tj. {(before-after)/before*100:.0f}%)"
    )
    return aggregated


def _series_title(entries: list, jikan_data: dict) -> str:
    """Sestaví popis série pro výpisy (např. 'SAO [4 řady]')."""
    if len(entries) == 1:
        return entries[0].title

    # Najdi nejkratší společný prefix nebo použij první titul
    titles = [e.title for e in entries]
    base   = _common_prefix(titles)
    if len(base) > 4:
        return f"{base.rstrip(': ')} [{len(entries)} řady]"
    return f"{titles[0]} [{len(entries)} řady]"


def _common_prefix(strings: list[str]) -> str:
    if not strings:
        return ""
    prefix = strings[0]
    for s in strings[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix


def print_series_groups(
    entries:    list,
    jikan_data: dict,
    titles_map: dict[int, str],
) -> None:
    """Vypíše přehled nalezených sériových skupin (pro --analyze)."""
    mal_ids = [e.mal_id for e in entries]
    groups  = build_series_groups(mal_ids, jikan_data)
    entry_map = {e.mal_id: e for e in entries}

    multi = [(root, members) for root, members in groups.items() if len(members) > 1]
    multi.sort(key=lambda x: -len(x[1]))

    print(f"\n{'═'*60}")
    print(f"  SÉRIOVÉ SKUPINY ({len(multi)} sérií nalezeno)")
    print(f"{'═'*60}")

    for root, members in multi:
        scores = [entry_map[m].score for m in members if m in entry_map]
        ttls   = [titles_map.get(m, str(m))[:35] for m in members]
        print(f"\n  Serie ({len(members)} řad, scores: {scores}, max: {max(scores)}):")
        for t in ttls:
            print(f"    · {t}")
