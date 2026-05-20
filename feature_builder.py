"""
feature_builder.py — Sestavení feature matrix z Jikan + AniList dat

Každý anime je reprezentován vektorem příznaků:
  ── Jikan (MAL) ──────────────────────────────────────────────────────────────
  - Binární:    žánr přítomen (1) nebo ne (0)
  - Binární:    téma přítomno (1) nebo ne (0)
  - One-hot:    demografie (Shounen / Seinen / Shoujo / Josei)
  - One-hot:    zdroj předlohy (Manga / LN / VN / Original / …)
  - One-hot:    typ média (TV / Movie / OVA / …)
  - Numerické:  MAL score, počet epizod (log), rok

  ── AniList ──────────────────────────────────────────────────────────────────
  - Spojitý:    tag rank 0–1 (např. anilist_Tsundere = 0.85)
                rank = jak dominantní je daný tag v titulu (dle AniList komunity)
  - Binární:    studio přítomno (1) nebo ne (0)

Výhoda AniList tagů: 500+ granulárních kategorií včetně archetypů postav
(Tsundere, Kuudere, Dandere), narativních vzorů (Love Triangle, Slow Romance,
Rivals to Lovers) a tematických kategorií (Tearjerker, Philosophy).

Všechny příznaky jsou normalizované StandardScalerem → koeficienty
jsou přímo srovnatelné napříč příznaky i zdroji dat.
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Mapy pro normalizaci hodnot z Jikan API ────────────────────────────────────

SOURCE_MAP = {
    "manga":         "Manga",
    "light novel":   "Light novel",
    "light_novel":   "Light novel",
    "visual novel":  "Visual novel",
    "visual_novel":  "Visual novel",
    "original":      "Original",
    "game":          "Game",
    "novel":         "Novel",
    "web manga":     "Manga",
    "4-koma manga":  "Manga",
    "doujinshi":     "Manga",
}

TYPE_MAP = {
    "tv":      "TV",
    "movie":   "Movie",
    "ova":     "OVA",
    "ona":     "ONA",
    "special": "Special",
    "music":   "Special",
}


@dataclass
class AniListConfig:
    """Konfigurace AniList příznaků."""
    # Tagy k zahrnutí: list názvů přesně dle AniList (case-sensitive)
    tags:            list[str] = field(default_factory=list)
    # Studia k zahrnutí jako binární příznaky
    studios:         list[str] = field(default_factory=list)
    # Minimální rank tagu pro zahrnutí jako nenulový příznak (0–100)
    min_rank:        int  = 0
    # Zda vyloučit adult tagy
    exclude_adult:   bool = True
    # Zda vyloučit spoilerové tagy
    exclude_spoiler: bool = True
    # Použít rank jako spojitý příznak (True) nebo binárně přítomen/nepřítomen (False)
    use_rank:        bool = True

    @classmethod
    def from_config(cls, cfg: dict) -> "AniListConfig":
        al = cfg.get("anilist", {})
        if not al.get("enabled", False):
            return cls(tags=[], studios=[])
        return cls(
            tags=            al.get("tags", []),
            studios=         al.get("studios", []),
            min_rank=        al.get("min_rank", 0),
            exclude_adult=   al.get("exclude_adult", True),
            exclude_spoiler= al.get("exclude_spoiler", True),
            use_rank=        al.get("use_rank", True),
        )

    @property
    def enabled(self) -> bool:
        return bool(self.tags or self.studios)


@dataclass
class FeatureConfig:
    """Konfigurace příznaků — načítá se z config.yaml."""
    genre_ids:           list[tuple[str, int]]       = field(default_factory=list)
    # (name, mal_id, skip_if_anilist)
    theme_ids:           list[tuple[str, int, bool]] = field(default_factory=list)
    demographics:        list[str]                   = field(default_factory=list)
    sources:             list[str]                   = field(default_factory=list)
    types:               list[str]                   = field(default_factory=list)
    use_mal_score:       bool = True
    use_composite_score: bool = False
    use_episodes:        bool = True
    use_year:            bool = True
    staff_directors:     list[tuple[str, int]] = field(default_factory=list)
    staff_writers:       list[tuple[str, int]] = field(default_factory=list)
    anilist:             AniListConfig         = field(default_factory=AniListConfig)

    @classmethod
    def from_config(cls, cfg: dict) -> "FeatureConfig":
        fc  = cfg.get("features", {})
        num = fc.get("numeric", {})
        st  = fc.get("staff", {})
        return cls(
            genre_ids=    [(g["name"], g["mal_id"]) for g in fc.get("genres", [])],
            theme_ids=    [
                (t["name"], t["mal_id"], bool(t.get("skip_if_anilist", False)))
                for t in fc.get("themes", [])
            ],
            demographics= fc.get("demographics", []),
            sources=      fc.get("sources", []),
            types=        fc.get("types", []),
            use_mal_score=       num.get("mal_score",       {}).get("include", True),
            use_composite_score= num.get("composite_score", {}).get("include", False),
            use_episodes=        num.get("episodes",        {}).get("include", True),
            use_year=            num.get("year",            {}).get("include", True),
            staff_directors= [(p["name"], p["mal_id"]) for p in st.get("directors", [])],
            staff_writers=   [(p["name"], p["mal_id"]) for p in st.get("writers",   [])],
            anilist=         AniListConfig.from_config(cfg),
        )


def _extract_id_set(items: list[dict]) -> set[int]:
    """Extrahuje množinu MAL ID z listu Jikan objektů."""
    return {item["mal_id"] for item in (items or []) if "mal_id" in item}


def anime_to_features(
    anime_data:    dict,
    fc:            FeatureConfig,
    anilist_data:  Optional[dict] = None,
) -> dict:
    """
    Převede Jikan anime_data (+ volitelně AniList data) na dict příznaků.

    Parametry:
        anime_data   — Jikan /anime/{id}/full data (povinné)
        fc           — konfigurace příznaků
        anilist_data — AniList Media data (volitelné, None = bez AniList příznaků)

    Vrací dict {feature_name: hodnota} nebo None pokud chybí Jikan data.
    """
    if anime_data is None:
        return None

    feats = {}

    # ── Žánry (z Jikan) ───────────────────────────────────────────────────────
    genre_ids = _extract_id_set(anime_data.get("genres", []))
    for name, mal_id in fc.genre_ids:
        feats[f"genre_{name}"] = 1.0 if mal_id in genre_ids else 0.0

    # ── Témata (z Jikan) ──────────────────────────────────────────────────────
    # Přeskočíme téma pokud: skip_if_anilist=True AND AniList je zapnutý
    # AND anilist_data je k dispozici (= máme AniList pokrytí pro tento titul)
    anilist_active = fc.anilist.enabled and anilist_data is not None
    theme_ids = _extract_id_set(anime_data.get("themes", []))
    for name, mal_id, skip_if_al in fc.theme_ids:
        if skip_if_al and anilist_active:
            continue   # AniList ekvivalent přebírá tento příznak
        feats[f"theme_{name}"] = 1.0 if mal_id in theme_ids else 0.0

    # ── Demografie (one-hot, z Jikan) ─────────────────────────────────────────
    demo_names = {d["name"] for d in (anime_data.get("demographics") or [])}
    for demo in fc.demographics:
        feats[f"demo_{demo}"] = 1.0 if demo in demo_names else 0.0

    # ── Zdroj předlohy (one-hot, z Jikan) ────────────────────────────────────
    raw_source        = (anime_data.get("source") or "").lower()
    normalized_source = SOURCE_MAP.get(raw_source, "Other")
    for src in fc.sources:
        feats[f"source_{src}"] = 1.0 if normalized_source == src else 0.0

    # ── Typ média (one-hot, z Jikan) ──────────────────────────────────────────
    raw_type        = (anime_data.get("type") or "").lower()
    normalized_type = TYPE_MAP.get(raw_type, "Other")
    for t in fc.types:
        feats[f"type_{t}"] = 1.0 if normalized_type == t else 0.0

    # ── Numerické příznaky ────────────────────────────────────────────────────
    mal_score_val = float(anime_data.get("score") or 0.0)

    if fc.use_mal_score:
        feats["mal_score"] = mal_score_val

    if fc.use_composite_score:
        # Composite score: vážený průměr MAL + AniList (0–10 škála)
        # AniList averageScore je 0–100, normalizujeme na 0–10
        al_raw = (anilist_data or {}).get("averageScore") or 0
        al_score = float(al_raw) / 10.0 if al_raw else 0.0
        if mal_score_val > 0 and al_score > 0:
            feats["composite_score"] = (mal_score_val + al_score) / 2.0
        elif mal_score_val > 0:
            feats["composite_score"] = mal_score_val
        elif al_score > 0:
            feats["composite_score"] = al_score
        else:
            feats["composite_score"] = 0.0

    if fc.use_episodes:
        eps = anime_data.get("episodes") or 1
        feats["log_episodes"] = math.log1p(float(eps))

    if fc.use_year:
        year = None
        aired = anime_data.get("aired", {})
        if aired and aired.get("from"):
            try:
                year = int(aired["from"][:4])
            except (ValueError, TypeError):
                pass
        feats["year"] = float(year) if year else 2010.0

    # ── Staff příznaky (z Jikan) ──────────────────────────────────────────────
    # Binární: 1 pokud daný režisér/scenárista pracoval na titulu.
    # Jikan /full vrací staff pole: [{person: {mal_id, name}, positions: [...]}]
    # Pozice jsou volné texty — hledáme klíčová slova pro direktora/scenáristu.
    DIRECTOR_POSITIONS  = {"director", "series director"}
    WRITER_POSITIONS    = {"script", "series composition", "screenplay",
                           "original creator", "original story"}

    staff_director_ids: set[int] = set()
    staff_writer_ids:   set[int] = set()
    for entry in anime_data.get("staff", []):
        person     = entry.get("person") or {}
        person_id  = person.get("mal_id")
        if not person_id:
            continue
        positions = {p.lower() for p in (entry.get("positions") or [])}
        if positions & DIRECTOR_POSITIONS:
            staff_director_ids.add(person_id)
        if positions & WRITER_POSITIONS:
            staff_writer_ids.add(person_id)

    for name, person_id in fc.staff_directors:
        feats[f"director_{name}"] = 1.0 if person_id in staff_director_ids else 0.0
    for name, person_id in fc.staff_writers:
        feats[f"writer_{name}"] = 1.0 if person_id in staff_writer_ids else 0.0

    # ── AniList tagy ──────────────────────────────────────────────────────────
    # Každý tag je příznak "anilist_{TagName}" s hodnotou:
    #   use_rank=True  → rank / 100  (spojitý 0–1)
    #   use_rank=False → 1.0 pokud rank >= min_rank, jinak 0.0
    al = fc.anilist
    if al.enabled and anilist_data:
        # Sestav slovník {tag_name: rank_0_to_1} z AniList dat.
        # max-wins: pokud AniList vrátí stejný tag vícekrát, zachováme
        # nejvyšší rank (konzervativnější než last-wins).
        al_tags: dict[str, float] = {}
        for tag in anilist_data.get("tags", []):
            if al.exclude_adult   and tag.get("isAdult"):
                continue
            if al.exclude_spoiler and (
                tag.get("isGeneralSpoiler") or tag.get("isMediaSpoiler")
            ):
                continue
            name     = tag["name"]
            rank_val = (tag.get("rank") or 0) / 100.0
            # max-wins: zachováme vyšší rank při duplicitě
            if name not in al_tags or rank_val > al_tags[name]:
                al_tags[name] = rank_val

        for tag_name in al.tags:
            key = f"anilist_{tag_name}"
            rank_val = al_tags.get(tag_name, 0.0)
            if al.use_rank:
                # Spojitý příznak: 0.0 pokud tag chybí nebo je pod min_rank
                feats[key] = rank_val if rank_val >= al.min_rank / 100.0 else 0.0
            else:
                # Binární příznak
                feats[key] = 1.0 if rank_val >= al.min_rank / 100.0 else 0.0

    elif al.enabled:
        # AniList data nejsou dostupná — vyplň nulami (nezkreslí model)
        for tag_name in al.tags:
            feats[f"anilist_{tag_name}"] = 0.0

    # ── AniList studia ────────────────────────────────────────────────────────
    if al.studios:
        al_studios: set[str] = set()
        if anilist_data:
            for node in anilist_data.get("studios", {}).get("nodes", []):
                if node.get("isAnimationStudio"):
                    al_studios.add(node["name"])
        for studio in al.studios:
            feats[f"studio_{studio}"] = 1.0 if studio in al_studios else 0.0

    return feats


def build_feature_matrix(
    entries:      list,           # list[MalEntry]
    jikan_data:   dict,           # {mal_id: jikan_data}
    fc:           FeatureConfig,
    anilist_data: dict | None = None,   # {mal_id: anilist_data}, volitelné
) -> tuple[pd.DataFrame, list[int], list[str]]:
    """
    Sestaví feature matrix pro trénovací data.

    Vstup:
        entries      — ohodnocené záznamy z MAL
        jikan_data   — Jikan data {mal_id: data}
        fc           — konfigurace příznaků
        anilist_data — AniList data {mal_id: data} (None = bez AniList)

    Výstup:
        X        — DataFrame příznaků
        scores   — list skutečných hodnocení
        mal_ids  — list MAL ID (stejné pořadí jako X a scores)
    """
    rows    = []
    scores  = []
    mal_ids = []
    skipped = 0

    for entry in entries:
        if entry.mal_id not in jikan_data:
            skipped += 1
            continue

        al_data = (anilist_data or {}).get(entry.mal_id)
        feats   = anime_to_features(jikan_data[entry.mal_id], fc, al_data)
        if feats is None:
            skipped += 1
            continue

        rows.append(feats)
        scores.append(entry.score)
        mal_ids.append(entry.mal_id)

    if skipped:
        log.info(f"Přeskočeno {skipped} titulů (chybějící Jikan data)")

    al_coverage = (
        sum(1 for mid in mal_ids if (anilist_data or {}).get(mid))
        if anilist_data else 0
    )
    if fc.anilist.enabled:
        log.info(
            f"AniList pokrytí: {al_coverage}/{len(mal_ids)} trénovacích titulů"
        )

    X = pd.DataFrame(rows).fillna(0.0)
    return X, scores, mal_ids


def build_prediction_matrix(
    jikan_data_list:  list[dict],      # list Jikan dat
    fc:               FeatureConfig,
    feature_columns:  list[str],       # musí odpovídat sloupcům z tréninku
    anilist_data:     dict | None = None,   # {mal_id: anilist_data}
) -> tuple[pd.DataFrame, list[int], list[str]]:
    """
    Sestaví feature matrix pro predikci (nová anime).

    Vstup:
        jikan_data_list — list Jikan dat pro predikci
        fc              — konfigurace příznaků
        feature_columns — sloupce z trénovacího X (zachovává pořadí)
        anilist_data    — AniList data {mal_id: data} (None = bez AniList)

    Výstup:
        X       — DataFrame příznaků (stejná struktura jako trénovací X)
        mal_ids — list MAL ID
        titles  — list názvů
    """
    rows    = []
    mal_ids = []
    titles  = []

    for data in jikan_data_list:
        if data is None:
            continue
        mal_id  = data.get("mal_id")
        al_data = (anilist_data or {}).get(mal_id) if mal_id else None
        feats   = anime_to_features(data, fc, al_data)
        if feats is None:
            continue
        rows.append(feats)
        mal_ids.append(mal_id)
        titles.append(data.get("title", "Unknown"))

    X = pd.DataFrame(rows).fillna(0.0)

    # Zarovnání sloupců — přidej chybějící jako 0, odstraň přebytečné
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feature_columns]

    return X, mal_ids, titles
