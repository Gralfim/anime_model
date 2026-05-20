"""
mal_parser.py — Parser XML exportu z MyAnimeList

Jak exportovat: https://myanimelist.net/panel.php?go=export
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass


@dataclass
class MalEntry:
    mal_id: int
    title: str
    type: str
    episodes: int
    watched_episodes: int
    score: int           # 0 = neohodnoceno
    status: str          # Completed, Watching, Plan to Watch, …
    start_date: str
    finish_date: str
    rewatched: int


def parse_export(path: str) -> tuple[list[MalEntry], dict]:
    """
    Parsuje MAL XML export.

    Vrací:
        entries  — seznam MalEntry pro všechna anime
        userinfo — dict s informacemi o uživateli
    """
    tree = ET.parse(path)
    root = tree.getroot()

    # Informace o uživateli
    userinfo = {}
    myinfo = root.find("myinfo")
    if myinfo is not None:
        for child in myinfo:
            userinfo[child.tag] = child.text or ""

    # Anime záznamy
    entries = []
    for node in root.findall("anime"):
        d = {t.tag: (t.text or "").strip() for t in node}
        entries.append(MalEntry(
            mal_id=int(d.get("series_animedb_id", 0)),
            title=d.get("series_title", ""),
            type=d.get("series_type", ""),
            episodes=int(d.get("series_episodes") or 0),
            watched_episodes=int(d.get("my_watched_episodes") or 0),
            score=int(d.get("my_score") or 0),
            status=d.get("my_status", ""),
            start_date=d.get("my_start_date", ""),
            finish_date=d.get("my_finish_date", ""),
            rewatched=int(d.get("my_times_watched") or 0),
        ))

    return entries, userinfo


def split_by_status(entries: list[MalEntry]) -> dict[str, list[MalEntry]]:
    """Rozdělí záznamy podle statusu."""
    result: dict[str, list[MalEntry]] = {}
    for e in entries:
        result.setdefault(e.status, []).append(e)
    return result
