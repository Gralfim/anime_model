"""
cluster_analyzer.py — Shlukování trénovacích dat + doporučení per cluster

Pipeline:
  1. Diferenciální skóre: user_score - mal_avg_score
     → odstraní bias způsobený obecně dobrými tituly
  2. K-Means na feature matrix → skupiny titulů se společnými atributy
  3. Charakterizace clusterů: dominantní příznaky, průměrný diff, příklady
  4. Detekce synergií: párové kombinace příznaků → diff vs. samostatné
  5. Per-cluster doporučení z PTW/kandidátů

Výstup pro každý cluster:
  - pojmenování (automatické z dominantních příznaků)
  - průměrné skóre, MAL průměr, diferenciál
  - dominantní příznaky
  - synergie (kombinace příznaků které spolu fungují lépe)
  - tituly v clusteru (příklady)
  - doporučení: kandidáti přiřazení do clusteru s predikcí
"""

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

log = logging.getLogger(__name__)


# ── Datové struktury ──────────────────────────────────────────────────────────

@dataclass
class ClusterProfile:
    cluster_id:     int
    label:          str                    # automaticky generované jméno
    n:              int                    # počet titulů
    avg_score:      float                  # průměrné uživatelské skóre
    avg_mal_score:  float                  # průměrné MAL skóre
    avg_diff:       float                  # průměrný diferenciál
    std_diff:       float                  # rozptyl diferenciálu
    top_features:   list[tuple[str, float]]  # (feature, mean_value) sestupně
    synergies:      list[tuple[str, str, float, float, float]]
                                           # (f1, f2, diff_combo, diff_f1, diff_f2)
    member_ids:     list[int]              # MAL ID členů
    member_titles:  list[tuple[int, int, float]]  # (mal_id, score, diff)
    recommendations: list[tuple[int, str, float, float, float]]
                                           # (mal_id, title, pred_score, mal_score, similarity)


# ── Pomocné funkce ────────────────────────────────────────────────────────────

def _diff_score(user_score: int, mal_score: Optional[float]) -> float:
    """Diferenciální skóre: user - MAL průměr. None → 0 jako fallback."""
    if not mal_score:
        return 0.0
    return user_score - mal_score


def _auto_label(top_features: list[tuple[str, float]], avg_diff: float) -> str:
    """
    Automaticky pojmenuje cluster z dominantních příznaků.
    Vrátí popisný string, např. 'Emocionální romance drama (LN)'.
    """
    # Vyber top příznaky s hodnotou > 0.4
    dominant = [f for f, v in top_features[:8] if v > 0.4]

    # Slovník mapování příznak → čitelná etiketa
    LABEL_MAP = {
        "genre_Romance":          "Romance",
        "genre_Drama":            "Drama",
        "genre_Comedy":           "Komedie",
        "genre_Action":           "Akce",
        "genre_Ecchi":            "Ecchi",
        "genre_Sci-Fi":           "Sci-Fi",
        "genre_Supernatural":     "Supernatural",
        "genre_Mystery":          "Mystery",
        "genre_Sports":           "Sports",
        "genre_Slice of Life":    "Slice of Life",
        "genre_Fantasy":          "Fantasy",
        "theme_Harem":            "Harem",
        "theme_Reverse Harem":    "Rev. Harem",
        "anilist_Tsundere":       "Tsundere",
        "anilist_Kuudere":        "Kuudere",
        "anilist_Love Triangle":  "Love Triangle",
        "anilist_Tearjerker":     "Tearjerker",
        "anilist_Harem":          "Harem",
        "anilist_Isekai":         "Isekai",
        "anilist_School":         "Škola",
        "anilist_School Club":    "School Club",
        "anilist_Military":       "Military",
        "anilist_Music":          "Hudba",
        "anilist_Slow Romance":   "Slow Romance",
        "anilist_Childhood Friends": "Childhood Friends",
        "anilist_Psychological":  "Psychologické",
        "anilist_Tragedy":        "Tragédie",
        "anilist_Feel-good":      "Feel-good",
        "anilist_Bittersweet":    "Bittersweet",
        "anilist_Coming of Age":  "Coming of Age",
        "source_Light novel":     "LN",
        "source_Visual novel":    "VN",
        "source_Original":        "Originál",
        "demo_Seinen":            "Seinen",
        "demo_Shoujo":            "Shoujo",
        "demo_Josei":             "Josei",
        "type_Movie":             "Film",
        "type_OVA":               "OVA",
    }

    parts = [LABEL_MAP.get(f, f.split("_", 1)[-1]) for f in dominant[:4]]
    label = " + ".join(parts) if parts else "Smíšený"

    qual = " ✦" if avg_diff >= 1.0 else (" ↑" if avg_diff >= 0.3 else "")
    return label + qual


def _cluster_similarity(
    candidate_feats: pd.Series,
    cluster_center:  np.ndarray,
    scaler:          StandardScaler,
) -> float:
    """
    Kosínová podobnost mezi kandidátem a středem clusteru
    (v normalizovaném prostoru).
    """
    x = scaler.transform(candidate_feats.values.reshape(1, -1))[0]
    c = cluster_center
    norm_x = np.linalg.norm(x)
    norm_c = np.linalg.norm(c)
    if norm_x == 0 or norm_c == 0:
        return 0.0
    return float(np.dot(x, c) / (norm_x * norm_c))


# ── Analýza synergií ──────────────────────────────────────────────────────────

def detect_synergies(
    X:       pd.DataFrame,
    diffs:   np.ndarray,
    mask:    np.ndarray,           # boolean maska pro cluster
    top_features: list[str],
    min_n:   int = 5,              # min. titulů pro statistickou spolehlivost
    top_k:   int = 5,              # top K synergií
) -> list[tuple[str, str, float, float, float]]:
    """
    Detekuje párové synergie příznaků v rámci clusteru.

    Pro každý pár (f1, f2) spočítá:
      diff_combo  = průměrný diff titulů kde OBA příznaky jsou přítomny
      diff_f1     = průměrný diff titulů kde JEN f1 je přítomen
      diff_f2     = průměrný diff titulů kde JEN f2 je přítomen
      synergie    = diff_combo - max(diff_f1, diff_f2)

    Vrací top_k synergií seřazených sestupně.
    """
    Xc    = X[mask]
    dc    = diffs[mask]
    syngs = []

    # Pracuj jen s příznaky relevanntními pro cluster (top features)
    feats = [f for f in top_features if f in X.columns][:12]

    for f1, f2 in combinations(feats, 2):
        both  = (Xc[f1] > 0) & (Xc[f2] > 0)
        only1 = (Xc[f1] > 0) & (Xc[f2] == 0)
        only2 = (Xc[f1] == 0) & (Xc[f2] > 0)

        if both.sum() < min_n:
            continue

        diff_combo = float(dc[both.values].mean())
        diff_f1    = float(dc[only1.values].mean()) if only1.sum() >= 2 else diff_combo
        diff_f2    = float(dc[only2.values].mean()) if only2.sum() >= 2 else diff_combo
        synergie   = diff_combo - max(diff_f1, diff_f2)

        syngs.append((f1, f2, diff_combo, diff_f1, diff_f2, synergie))

    syngs.sort(key=lambda x: -x[5])
    return [(f1, f2, dc, d1, d2) for f1, f2, dc, d1, d2, _ in syngs[:top_k]]


# ── Hlavní funkce ─────────────────────────────────────────────────────────────

def run_clustering(
    X:           pd.DataFrame,
    scores:      list[int],
    mal_ids:     list[int],
    jikan_data:  dict,             # {mal_id: jikan_data} pro MAL skóre
    titles:      dict[int, str],
    k:           int = 6,
    random_seed: int = 42,
) -> tuple[list[ClusterProfile], np.ndarray, StandardScaler]:
    """
    Spustí K-Means clustering a vrátí profily clusterů.

    Vrací:
        profiles  — list ClusterProfile (jeden per cluster)
        labels    — numpy array přiřazení cluster label (délka = len(scores))
        scaler    — StandardScaler použitý při clusteringu
    """
    y_arr = np.array(scores, dtype=float)

    # Diferenciální skóre
    diffs = np.array([
        _diff_score(s, (jikan_data.get(mid) or {}).get("score"))
        for s, mid in zip(scores, mal_ids)
    ], dtype=float)

    # Normalizace pro clustering
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means
    km = KMeans(n_clusters=k, random_state=random_seed, n_init=20)
    labels = km.fit_predict(X_scaled)

    profiles = []
    for cid in range(k):
        mask    = labels == cid
        n       = int(mask.sum())
        if n == 0:
            continue

        # Základní statistiky
        avg_score    = float(y_arr[mask].mean())
        avg_diff     = float(diffs[mask].mean())
        std_diff     = float(diffs[mask].std())

        # MAL průměr pro tituly v clusteru
        mal_scores_c = [
            (jikan_data.get(mid) or {}).get("score") or 0.0
            for mid in np.array(mal_ids)[mask]
        ]
        avg_mal_score = float(np.mean([s for s in mal_scores_c if s > 0])) if mal_scores_c else 0.0

        # Dominantní příznaky: průměrná hodnota v clusteru vs. zbytek
        Xc    = X[mask]
        Xrest = X[~mask]
        feat_means = Xc.mean()
        feat_diffs = feat_means - X.mean()  # nadreprezentace vs. celkový průměr

        top_features = (
            feat_diffs
            .sort_values(ascending=False)
            .head(15)
            .items()
        )
        top_feats_list = [(f, float(v)) for f, v in top_features if v > 0]

        # Synergie
        top_feat_names = [f for f, _ in top_feats_list[:10]]
        synergies = detect_synergies(X, diffs, mask, top_feat_names)

        # Členové clusteru
        cluster_ids = list(np.array(mal_ids)[mask])
        cluster_scores = list(y_arr[mask].astype(int))
        cluster_diffs  = list(diffs[mask])

        member_titles = sorted(
            zip(cluster_ids, cluster_scores, cluster_diffs),
            key=lambda x: (-x[1], -x[2])
        )

        label = _auto_label(top_feats_list, avg_diff)

        profiles.append(ClusterProfile(
            cluster_id    = cid,
            label         = label,
            n             = n,
            avg_score     = avg_score,
            avg_mal_score = avg_mal_score,
            avg_diff      = avg_diff,
            std_diff      = std_diff,
            top_features  = top_feats_list,
            synergies     = synergies,
            member_ids    = cluster_ids,
            member_titles = list(member_titles),
            recommendations = [],
        ))

    # Seřaď clustery podle avg_diff (nejsilnější preference první)
    profiles.sort(key=lambda p: -p.avg_diff)

    return profiles, labels, scaler, km.cluster_centers_


def assign_candidates(
    profiles:       list[ClusterProfile],
    centers:        np.ndarray,
    scaler:         StandardScaler,
    Xp:             pd.DataFrame,          # feature matrix kandidátů
    p_ids:          list[int],
    p_titles_map:   dict[int, str],
    jikan_cands:    dict,                  # {mal_id: jikan_data}
    model_results,                         # ModelResults pro predikci skóre
    feature_cols:   list[str],             # sloupce z tréninku
    top_per_cluster: int = 8,
) -> None:
    """
    Přiřadí každého kandidáta do nejbližšího clusteru (kosínová podobnost)
    a naplní ClusterProfile.recommendations.

    Každé doporučení: (mal_id, title, predicted_score, mal_score, similarity)
    """
    from model import predict as model_predict
    import numpy as np

    if Xp.empty:
        return

    # Predikce skóre pro všechny kandidáty
    preds = np.clip(model_predict(model_results, Xp), 1, 10)

    # Normalizace kandidátů stejným scalerem jako clustering
    Xp_scaled = scaler.transform(Xp)

    # Středy clusterů → dict cid: center
    center_map = {p.cluster_id: centers[p.cluster_id] for p in profiles}

    # Pro každého kandidáta najdi nejbližší cluster
    candidate_assignments: dict[int, list] = {p.cluster_id: [] for p in profiles}

    for i, (mal_id, pred) in enumerate(zip(p_ids, preds)):
        x_norm = Xp_scaled[i]
        best_cid, best_sim = -1, -2.0
        for p in profiles:
            c    = center_map[p.cluster_id]
            n_x  = np.linalg.norm(x_norm)
            n_c  = np.linalg.norm(c)
            sim  = float(np.dot(x_norm, c) / (n_x * n_c)) if n_x > 0 and n_c > 0 else 0.0
            if sim > best_sim:
                best_sim, best_cid = sim, p.cluster_id

        title     = p_titles_map.get(mal_id, f"ID:{mal_id}")
        mal_score = (jikan_cands.get(mal_id) or {}).get("score") or 0.0
        candidate_assignments[best_cid].append(
            (mal_id, title, float(pred), float(mal_score), best_sim)
        )

    # Naplň recommendations — seřazeno dle predikované skóre
    for p in profiles:
        recs = sorted(
            candidate_assignments[p.cluster_id],
            key=lambda r: -r[2]
        )[:top_per_cluster]
        p.recommendations = recs


# ── Výpis výsledků ────────────────────────────────────────────────────────────

def print_cluster_report(
    profiles:  list[ClusterProfile],
    titles:    dict[int, str],
    examples:  int = 6,
) -> None:
    """Vypíše kompletní report clusterů na stdout."""

    LABEL_SHORT = {
        "genre_Romance":          "Romance",
        "genre_Drama":            "Drama",
        "genre_Comedy":           "Komedie",
        "genre_Action":           "Akce",
        "genre_Ecchi":            "Ecchi",
        "genre_Supernatural":     "Supernatural",
        "genre_Sci-Fi":           "Sci-Fi",
        "genre_Sports":           "Sports",
        "genre_Slice of Life":    "Slice of Life",
        "genre_Fantasy":          "Fantasy",
        "theme_Harem":            "Harem",
        "theme_Reverse Harem":    "Rev. Harem",
        "anilist_Tsundere":       "Tsundere",
        "anilist_Kuudere":        "Kuudere",
        "anilist_Yandere":        "Yandere",
        "anilist_Love Triangle":  "Love Triangle",
        "anilist_Tearjerker":     "Tearjerker",
        "anilist_Harem":          "Harem",
        "anilist_Isekai":         "Isekai",
        "anilist_School":         "Škola",
        "anilist_School Club":    "School Club",
        "anilist_Military":       "Military",
        "anilist_Music":          "Hudba",
        "anilist_Slow Romance":   "Slow Romance",
        "anilist_Childhood Friends": "Childhood Friends",
        "anilist_Psychological":  "Psychologické",
        "anilist_Tragedy":        "Tragédie",
        "anilist_Feel-good":      "Feel-good",
        "anilist_Bittersweet":    "Bittersweet",
        "anilist_Coming of Age":  "Coming of Age",
        "anilist_Fake Relationship": "Fake Relationship",
        "anilist_Rivals to Lovers": "Rivals to Lovers",
        "source_Light novel":     "LN",
        "source_Visual novel":    "VN",
        "source_Original":        "Originál",
        "source_Manga":           "Manga",
        "demo_Seinen":            "Seinen",
        "demo_Shoujo":            "Shoujo",
        "demo_Josei":             "Josei",
        "demo_Shounen":           "Shounen",
        "type_Movie":             "Film",
        "type_OVA":               "OVA",
    }

    def short(feat: str) -> str:
        return LABEL_SHORT.get(feat, feat.split("_", 1)[-1])

    W = 70
    print("\n" + "█" * W)
    print(f"  CLUSTER ANALÝZA — {len(profiles)} skupin vkusu")
    print("█" * W)

    for i, p in enumerate(profiles):
        # ── Hlavička clusteru ──────────────────────────────────────────────
        diff_str = f"{p.avg_diff:+.2f}"
        diff_bar = ("▲" * min(int(abs(p.avg_diff) * 3), 8)
                    if p.avg_diff >= 0
                    else "▼" * min(int(abs(p.avg_diff) * 3), 8))
        print(f"\n{'─'*W}")
        print(f"  CLUSTER {i+1}: {p.label}")
        print(f"{'─'*W}")
        print(f"  Titulů: {p.n}   |   "
              f"Tvůj průměr: {p.avg_score:.1f}   |   "
              f"MAL průměr: {p.avg_mal_score:.1f}   |   "
              f"Diferenciál: {diff_str} {diff_bar}")

        # ── Dominantní příznaky ────────────────────────────────────────────
        feat_tags = "  ".join(
            f"[{short(f)} {v:.0%}]"
            for f, v in p.top_features[:8]
            if v > 0.25
        )
        print(f"\n  Charakter: {feat_tags}")

        # ── Synergie ──────────────────────────────────────────────────────
        if p.synergies:
            print(f"\n  Synergie příznaků (kombinace která funguje lépe):")
            for f1, f2, dc, d1, d2 in p.synergies[:3]:
                synergie = dc - max(d1, d2)
                print(f"    {short(f1)} + {short(f2)}:  "
                      f"diff={dc:+.2f}  "
                      f"(samostatně: {d1:+.2f} / {d2:+.2f},  "
                      f"synergie: {synergie:+.2f})")

        # ── Příklady titulů ────────────────────────────────────────────────
        print(f"\n  Tvoje tituly v clusteru (top {examples}):")
        ex_rows = []
        for mal_id, score, diff in p.member_titles[:examples]:
            title = titles.get(mal_id, f"ID:{mal_id}")[:40]
            ex_rows.append([score, f"{diff:+.1f}", title])
        print(tabulate(ex_rows,
                       headers=["skóre", "diff", "titul"],
                       tablefmt="simple",
                       colalign=("center", "right", "left")))

        # ── Doporučení ─────────────────────────────────────────────────────
        if p.recommendations:
            print(f"\n  Doporučení pro tento cluster:")
            rec_rows = []
            for mal_id, title, pred, mal_sc, sim in p.recommendations:
                sim_pct = f"{sim*100:.0f}%"
                mal_str = f"{mal_sc:.2f}" if mal_sc > 0 else "—"
                rec_rows.append([
                    f"{pred:.1f}",
                    title[:42],
                    mal_str,
                    sim_pct,
                ])
            print(tabulate(
                rec_rows,
                headers=["predikce", "titul", "MAL", "shoda"],
                tablefmt="simple",
                colalign=("center", "left", "center", "right"),
            ))
        else:
            print("\n  [žádní kandidáti pro tento cluster]")

    print(f"\n{'█'*W}\n")


def export_cluster_csv(
    profiles: list[ClusterProfile],
    titles:   dict[int, str],
    path:     str = "clusters.csv",
) -> None:
    """Uloží přehled clusterů a doporučení do CSV."""
    rows = []

    # Členové
    for p in profiles:
        for mal_id, score, diff in p.member_titles:
            rows.append({
                "typ":        "člen",
                "cluster":    p.cluster_id,
                "label":      p.label,
                "mal_id":     mal_id,
                "title":      titles.get(mal_id, ""),
                "score":      score,
                "diff":       round(diff, 2),
                "pred_score": "",
                "mal_score":  "",
                "similarity": "",
            })

    # Doporučení
    for p in profiles:
        for mal_id, title, pred, mal_sc, sim in p.recommendations:
            rows.append({
                "typ":        "doporučení",
                "cluster":    p.cluster_id,
                "label":      p.label,
                "mal_id":     mal_id,
                "title":      title,
                "score":      "",
                "diff":       "",
                "pred_score": round(pred, 2),
                "mal_score":  round(mal_sc, 2) if mal_sc else "",
                "similarity": round(sim, 3),
            })

    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    log.info(f"Cluster report uložen: {path}")
