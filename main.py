#!/usr/bin/env python3
"""
main.py — Anime Taste Model
═══════════════════════════════════════════════════════════════════
Použití:
    python main.py                        # výchozí config.yaml
    python main.py --mode ptw             # predikuj PTW
    python main.py --mode ids --ids 5114 9253 32979
    python main.py --alpha 0.5            # jiná regularizace
    python main.py --explain 5114         # vysvětli predikci
    python main.py --analyze              # analýza příznaků
    python main.py --cluster              # cluster analýza + doporučení per cluster
    python main.py --cluster --k 5        # jiný počet clusterů (default: 6)
    python main.py --train-only           # jen trénink, bez predikce
    python main.py --list-tags            # průzkum AniList tagů + YAML výstup
    python main.py --cf                   # collaborative filtering (podobní uživatelé)
    python main.py --cf --cf-resample     # vynutí nový výběr uživatelů (ignoruje cache seedů)
    python main.py --list-tags --list-tags-min 5  # přísnější práh
    python main.py --list-staff           # průzkum režisérů/scenáristů + YAML
    python main.py --no-aggregate         # vypni agregaci sérií
═══════════════════════════════════════════════════════════════════
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
from tabulate import tabulate

from mal_parser        import parse_export, split_by_status
from jikan_client      import JikanClient
from anilist_client    import AniListClient
from feature_builder   import FeatureConfig, build_feature_matrix, build_prediction_matrix
from series_aggregator import aggregate_entries, print_series_groups
from cf_model          import (
    CFConfig, CollaborativeFilter,
    print_cf_report, export_cf_csv,
)
from cluster_analyzer  import (
    run_clustering, assign_candidates,
    print_cluster_report, export_cluster_csv,
)
from model             import (
    train, predict, explain_prediction,
    print_evaluation, print_coefficients,
    print_predicted_vs_actual, print_feature_analysis,
)

# ── Logging ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Hlavní pipeline ─────────────────────────────────────────────────────────────

def run(cfg: dict, args: argparse.Namespace) -> None:

    jikan   = JikanClient(cache_dir=cfg["cache_dir"])
    anilist = AniListClient(cache_dir=cfg["cache_dir"])
    fc      = FeatureConfig.from_config(cfg)

    # ── 1. Parsování MAL exportu ───────────────────────────────────────────────
    print("\n[1/6] Parsování MAL exportu…")
    entries, userinfo = parse_export(cfg["mal_export"])
    by_status = split_by_status(entries)

    username = userinfo.get("user_name", "?")
    print(f"  Uživatel:  {username}")
    print(f"  Celkem:    {len(entries)} titulů")
    for status, lst in sorted(by_status.items()):
        scored = sum(1 for e in lst if e.score > 0)
        print(f"    {status:<20} {len(lst):4d}  ({scored} ohodnocených)")

    # ── 2. Výběr trénovacích dat ───────────────────────────────────────────────
    train_cfg = cfg.get("training", {})
    statuses  = train_cfg.get("statuses", ["Completed"])
    min_score = train_cfg.get("min_score", 1)

    train_entries = [
        e for e in entries
        if e.status in statuses and e.score >= min_score
    ]
    print(f"\n[2/6] Trénovací data: {len(train_entries)} ohodnocených titulů")

    train_ids = [e.mal_id for e in train_entries]

    # ── Speciální příkaz: průzkum AniList tagů ─────────────────────────────────
    if args.list_tags:
        print("\n[ℹ] Průzkum AniList tagů pro trénovací tituly…")
        al_data    = anilist.get_anime_batch(train_ids)
        tag_counts = anilist.list_all_tags(train_ids)
        total_titles = len(al_data)
        top_n = getattr(args, "list_tags_n", None) or 80
        top_tags = sorted(tag_counts.items(), key=lambda x: -x[1])[:top_n]

        # Tabulkový výstup
        print(f"\n  Top {top_n} tagů (z {total_titles} titulů):\n")
        rows = [[cnt, f"{cnt/max(total_titles,1)*100:.0f}%", name]
                for name, cnt in top_tags]
        print(tabulate(rows, headers=["počet", "%", "tag"], tablefmt="simple"))

        # YAML ready-to-paste výstup
        min_count = getattr(args, "list_tags_min", None) or max(3, total_titles // 15)
        yaml_tags = [name for name, cnt in top_tags if cnt >= min_count]

        CATEGORIES = {
            "Archetypy postav": [
                "Tsundere","Kuudere","Dandere","Yandere","Genki Girl",
                "Ojou-sama","Tomboy","Loli","Bishounen",
            ],
            "Romance vzory": [
                "Love Triangle","Slow Romance","Childhood Friends",
                "Rivals to Lovers","Fake Relationship","Unrequited Love",
                "Age Gap","Forbidden Love","Sudden Girlfriend Appearance",
                "Harem","Reverse Harem","Polyamory",
            ],
            "Emocionální tón": [
                "Tearjerker","Feel-good","Tragedy","Bittersweet",
                "Slapstick","Parody",
            ],
            "Témata a prostředí": [
                "School","School Club","Workplace","Isekai","Military",
                "Music","Sports","Time Manipulation","Time Skip",
                "Coming of Age","Found Family","Philosophy","Psychological",
                "Non-linear Storytelling","Ensemble Cast",
            ],
        }

        print("\n  # ─── Zkopíruj do sekce anilist.tags v config.yaml ───────────")
        print(f"  # ─── (práh: {min_count}+ titulů z {total_titles}; zakomentuj co nechceš) ──")
        print()
        printed: set = set()
        for cat_name, cat_tags in CATEGORIES.items():
            cat_yaml = [t for t in cat_tags if t in set(yaml_tags)]
            if not cat_yaml:
                continue
            print(f"    # {cat_name}")
            for t in cat_yaml:
                cnt = tag_counts.get(t, 0)
                print(f"    - \"{t}\"  # {cnt} titulů ({cnt/max(total_titles,1)*100:.0f}%)")
                printed.add(t)
        uncategorized = [t for t in yaml_tags if t not in printed]
        if uncategorized:
            print("    # Ostatní")
            for t in uncategorized:
                cnt = tag_counts.get(t, 0)
                print(f"    - \"{t}\"  # {cnt} titulů ({cnt/max(total_titles,1)*100:.0f}%)")
        print()
        print(f"  # Celkem {len(yaml_tags)} tagů (práh {min_count}+); --list-tags-min N pro jiný práh")
        return

    # ── Speciální příkaz: průzkum staff ──────────────────────────────────────
    if args.list_staff:
        print("\n[ℹ] Průzkum režisérů a scenáristů pro trénovací tituly…")
        jikan_data_st = jikan.get_anime_batch(train_ids)
        staff = jikan.list_all_staff(train_ids)
        top_n = getattr(args, "list_tags_n", None) or 30
        min_count = getattr(args, "list_tags_min", None) or 2

        for role, key in [("Režiséři", "directors"), ("Scenáristé", "writers")]:
            entries = staff[key][:top_n]
            print(f"\n  Top {top_n} {role}:\n")
            rows = [[cnt, pid, name, pos] for pid, name, pos, cnt in entries]
            print(tabulate(rows,
                headers=["titulů", "MAL ID", "jméno", "pozice"],
                tablefmt="simple"))
            print(f"\n  # ─── Zkopíruj do features.staff.{key} v config.yaml ───")
            for pid, name, pos, cnt in entries:
                if cnt < min_count:
                    break
                safe = name.replace(",", "").replace(" ", "_")
                print(f"    - name: \"{safe}\"  # {name} ({cnt} titulů, {pos})")
                print(f"      mal_id: {pid}")
        return

    # ── Speciální příkaz: průzkum MAL příznaků ───────────────────────────────
    if args.list_mal:
        print("\n[ℹ] Průzkum MAL příznaků pro trénovací tituly…")
        jikan_data_mal = jikan.get_anime_batch(train_ids)
        mal_feats = jikan.list_mal_features(train_ids)
        total = len(jikan_data_mal)

        SECTIONS = [
            ("genres",       "Žánry → features.genres",
             '      - name: "{name}"\n        mal_id: {id}'),
            ("themes",       "Témata → features.themes (přidej skip_if_anilist kde vhodné)",
             '      - name: "{name}"\n        mal_id: {id}\n        skip_if_anilist: false'),
            ("demographics", "Demografie → features.demographics",
             '      - "{name}"'),
            ("sources",      "Zdroje předlohy → features.sources",
             '      - "{name}"'),
            ("types",        "Typy médií → features.types",
             '      - "{name}"'),
        ]

        for key, header, yaml_fmt in SECTIONS:
            data_raw = mal_feats[key]
            print(f"\n  ── {header} ──")
            if key in ("genres", "themes"):
                items = sorted(data_raw.items(), key=lambda x: -x[1])
                rows = [[cnt, f"{cnt/max(total,1)*100:.0f}%", mid, name]
                        for (mid, name), cnt in items]
                print(tabulate(rows, headers=["počet", "%", "MAL ID", "název"], tablefmt="simple"))
                print(f"\n  # ─── Zkopíruj do features.{key} v config.yaml ───")
                for (mid, name), cnt in items:
                    print(f"    # {name} ({cnt} titulů, {cnt/max(total,1)*100:.0f}%)")
                    print(yaml_fmt.format(name=name, id=mid))
            else:
                items = sorted(data_raw.items(), key=lambda x: -x[1])
                rows = [[cnt, f"{cnt/max(total,1)*100:.0f}%", name] for name, cnt in items]
                print(tabulate(rows, headers=["počet", "%", "hodnota"], tablefmt="simple"))
                print(f"\n  # ─── Zkopíruj do features.{key} v config.yaml ───")
                for name, cnt in items:
                    print(f"    # {name} ({cnt} titulů)")
                    print(yaml_fmt.format(name=name, id=None))
        return

    # ── Speciální příkaz: průzkum AniList studií ──────────────────────────────
    if args.list_studios:
        print("\n[ℹ] Průzkum AniList animačních studií pro trénovací tituly…")
        al_st         = anilist.get_anime_batch(train_ids)
        studio_counts = anilist.list_all_studios(train_ids)
        total         = len(al_st)
        min_count     = getattr(args, "list_tags_min", None) or 2
        top_n         = getattr(args, "list_tags_n",   None) or 50

        sorted_st = sorted(studio_counts.items(), key=lambda x: -x[1])[:top_n]
        print(f"\n  Top {top_n} animačních studií (z {total} titulů):\n")
        rows = [[cnt, f"{cnt/max(total,1)*100:.0f}%", name] for name, cnt in sorted_st]
        print(tabulate(rows, headers=["počet", "%", "studio"], tablefmt="simple"))
        print(f"\n  # ─── Zkopíruj do anilist.studios v config.yaml ─────────")
        print(f"  # ─── (práh: {min_count}+ titulů) ──────────────────────────")
        print()
        for name, cnt in sorted_st:
            if cnt < min_count:
                break
            print(f'    - "{name}"  # {cnt} titulů ({cnt/max(total,1)*100:.0f}%)')
        return

    # ── 3. Stažení Jikan dat ───────────────────────────────────────────────────
    print("\n[3/6] Stahování Jikan dat…")
    jikan_data = jikan.get_anime_batch(train_ids)

    # ── 3b. Stažení staff dat ────────────────────────────────────────────────
    staff_data_train = None
    if fc.staff_directors or fc.staff_writers:
        print("\n  Stahování staff dat (režiséři/scenáristé)…")
        staff_data_train = jikan.get_staff_batch(train_ids)
    else:
        print("\n  Staff data: přeskočeno (žádní v config.yaml)")

    # ── 3c. Agregace sérií ────────────────────────────────────────────────────
    do_aggregate = (
        cfg.get("training", {}).get("aggregate_series", True)
        and not args.no_aggregate
    )

    titles = {e.mal_id: e.title for e in entries}

    if do_aggregate:
        print("\n  Agregace sérií (sequel/prequel → max skóre)…")
        if args.analyze:
            # Při analýze zobraz skupiny před kola psením
            print_series_groups(train_entries, jikan_data, titles)
        train_entries = aggregate_entries(train_entries, jikan_data)
        print(f"  Po agregaci: {len(train_entries)} záznamů")
    else:
        print("\n  Agregace sérií: vypnuta")

    # ── 4. Stažení AniList dat ────────────────────────────────────────────────
    train_ids_agg = [e.mal_id for e in train_entries]
    al_data_train = None
    if fc.anilist.enabled:
        print(f"\n[4/6] Stahování AniList dat "
              f"({len(fc.anilist.tags)} tagů, {len(fc.anilist.studios)} studií)…")
        al_data_train = anilist.get_anime_batch(train_ids_agg)
    else:
        print("\n[4/6] AniList přeskočeno (disabled v config.yaml)")

    # ── 5. Sestavení feature matrix ────────────────────────────────────────────
    print("\n[5/6] Sestavení feature matrix…")

    if args.alpha is not None:
        cfg["model"]["alpha"] = args.alpha

    X, scores, mal_ids_train = build_feature_matrix(
        train_entries, jikan_data, fc, al_data_train, staff_data_train
    )
    al_count = sum(1 for mid in mal_ids_train if (al_data_train or {}).get(mid))
    print(f"  Feature matrix: {X.shape[0]} titulů × {X.shape[1]} příznaků")
    if fc.anilist.enabled:
        skipped_themes = sum(
            1 for _, _, skip in fc.theme_ids if skip
        )
        print(f"  MAL témata přeskočena (skip_if_anilist): {skipped_themes}")
        print(f"  AniList pokrytí: {al_count}/{len(mal_ids_train)} titulů "
              f"({al_count/max(len(mal_ids_train),1)*100:.0f}%)")
        al_feats = [c for c in X.columns if c.startswith(("anilist_", "studio_"))]
        print(f"  AniList příznaky: {len(al_feats)} "
              f"(tagy: {sum(1 for c in al_feats if c.startswith('anilist_'))}, "
              f"studia: {sum(1 for c in al_feats if c.startswith('studio_'))})")

    # ── 6. Trénování modelu ────────────────────────────────────────────────────
    print("\n[6/6] Trénování modelu…")
    model_cfg = cfg.get("model", {})
    results   = train(
        X, scores,
        model_type = model_cfg.get("type", "ridge"),
        alpha      = model_cfg.get("alpha", 1.0),
        cv_folds   = model_cfg.get("cv_folds", 5),
    )

    print_evaluation(results)
    print_coefficients(results, top_n=35)

    if not args.train_only:
        print_predicted_vs_actual(results, X, scores, mal_ids_train, titles)

    # ── Analýza příznaků (--analyze) ──────────────────────────────────────────
    if args.analyze:
        print_feature_analysis(results, X, scores, mal_ids_train, titles, top_n=20)

    # ── Cluster analýza (--cluster) ────────────────────────────────────────────
    if args.cluster:
        k = args.k or cfg.get("cluster", {}).get("k", 6)
        print(f"\n{'═'*70}")
        print(f"  CLUSTER ANALÝZA (K={k})")
        print(f"  Diferenciální skóre: user - MAL průměr")
        print(f"{'═'*70}")
        print(f"  Spouštím K-Means na {X.shape[0]} titulech × {X.shape[1]} příznacích…")

        profiles, labels, cl_scaler, centers = run_clustering(
            X, scores, mal_ids_train, jikan_data, titles, k=k
        )

        # Stáhni kandidáty pro doporučení (PTW nebo vlastní seznam)
        pred_cfg  = cfg.get("prediction", {})
        cl_mode   = args.mode or pred_cfg.get("mode", "ptw")
        min_mal   = pred_cfg.get("min_mal_score", 7.0)

        if cl_mode == "ptw":
            cand_ids = [e.mal_id for e in by_status.get("Plan to Watch", [])]
        elif cl_mode == "ids":
            cand_ids = [int(x) for x in args.ids] if args.ids else []
        else:
            cand_ids = [e.mal_id for e in by_status.get("Plan to Watch", [])]

        if cand_ids:
            print(f"  Stahování dat pro {len(cand_ids)} kandidátů…")
            cand_jikan = jikan.get_anime_batch(cand_ids)
            cand_data  = [
                v for k_id, v in cand_jikan.items()
                if v and (v.get("score") or 0) >= min_mal
                and k_id not in {e.mal_id for e in train_entries}
            ]

            cand_al = None
            if fc.anilist.enabled and cand_data:
                cand_al_ids = [d["mal_id"] for d in cand_data if d.get("mal_id")]
                cand_al     = anilist.get_anime_batch(cand_al_ids)

            cand_staff = None
            if fc.staff_directors or fc.staff_writers:
                cand_staff = jikan.get_staff_batch(
                    [d["mal_id"] for d in cand_data if d.get("mal_id")]
                )
            Xc, c_ids, c_titles_list = build_prediction_matrix(
                cand_data, fc, list(X.columns), cand_al, cand_staff
            )
            c_titles_map = dict(zip(c_ids, c_titles_list))

            top_per = cfg.get("cluster", {}).get("top_per_cluster",
                                                  pred_cfg.get("show_top", 8))

            assign_candidates(
                profiles, centers, cl_scaler,
                Xc, c_ids, c_titles_map,
                cand_jikan, results, list(X.columns),
                top_per_cluster=top_per,
            )

        print_cluster_report(profiles, titles)

        csv_path = "clusters.csv"
        export_cluster_csv(profiles, titles, csv_path)
        print(f"  Cluster report uložen: {csv_path}")

        if args.train_only:
            return

    # ── Vysvětlení konkrétní predikce (--explain) ──────────────────────────────
    if args.explain:
        explain_ids = [int(x) for x in args.explain]
        print("\n" + "═" * 60)
        print("  VYSVĚTLENÍ PREDIKCÍ")
        print("═" * 60)

        expl_jikan   = jikan.get_anime_batch(explain_ids)
        expl_anilist = anilist.get_anime_batch(explain_ids) if fc.anilist.enabled else None
        expl_list    = [expl_jikan[i] for i in explain_ids if i in expl_jikan]

        expl_staff = jikan.get_staff_batch(explain_ids) if (fc.staff_directors or fc.staff_writers) else None
        Xe, xe_ids, xe_titles = build_prediction_matrix(
            expl_list, fc, list(X.columns), expl_anilist, expl_staff
        )
        preds = predict(results, Xe)

        for i, (mid, title, pred) in enumerate(zip(xe_ids, xe_titles, preds)):
            pred_clipped = float(np.clip(pred, 1, 10))
            print(f"\n  {title} (MAL ID: {mid})")
            print(f"  Predikované skóre: {pred_clipped:.2f}")
            contrib = explain_prediction(results, Xe.iloc[i], pred_clipped)
            table = [
                [r["příznak"], f"{r['hodnota']:.2f}",
                 f"{r['koeficient']:+.4f}", f"{r['příspěvek']:+.4f}"]
                for _, r in contrib.head(15).iterrows()
            ]
            print(tabulate(
                table,
                headers=["příznak", "hodnota", "koef", "příspěvek"],
                tablefmt="simple"
            ))

    # ── Collaborative Filtering (--cf) ───────────────────────────────────────
    if args.cf:
        from cf_model import SimilarUser   # lokální import pro přehlednost

        cf_cfg = CFConfig.from_config(cfg)
        cf     = CollaborativeFilter(cf_cfg, jikan)

        print(f"\n{'═'*70}")
        print(f"  COLLABORATIVE FILTERING")
        print(f"  Seed: niche tituly se skóre ≥{cf_cfg.seed_min_score}, "
              f"members ≤{cf_cfg.seed_max_popularity:,}")
        print(f"  Podobnost: Pearson r ≥{cf_cfg.min_correlation}, "
              f"overlap ≥{cf_cfg.min_overlap} titulů")
        print(f"{'═'*70}")

        # Invaliduj cache seedů pokud --cf-resample
        if getattr(args, "cf_resample", False):
            import shutil
            seed_cache = Path(cf_cfg.cache_dir) / "cf"
            if seed_cache.exists():
                for f in seed_cache.glob("userupdates_*.json"):
                    f.unlink()
                print("  Cache seed uživatelů vymazána.")

        recommendations = cf.run(
            my_entries   = train_entries,
            jikan_data   = jikan_data,
            existing_ids = {e.mal_id for e in entries},
            titles_map   = titles,
        )

        # Načti similar_users pro výpis (z interního stavu CF)
        # CF si je ukládá v průběhu run() — vypiš z výsledků
        print_cf_report(recommendations, show_top=cf_cfg.show_top)
        export_cf_csv(recommendations)
        print(f"  Výsledky uloženy: cf_recommendations.csv")

        if args.train_only:
            return

    # ── Predikce ──────────────────────────────────────────────────────────────
    if args.train_only:
        print("\nPredikce přeskočena (--train-only).")
        return

    pred_cfg  = cfg.get("prediction", {})
    mode      = args.mode or pred_cfg.get("mode", "ptw")
    min_mal   = pred_cfg.get("min_mal_score", 7.0)
    show_top  = pred_cfg.get("show_top", 20)
    existing_ids = {e.mal_id for e in entries}

    if mode == "ptw":
        predict_ids = [e.mal_id for e in by_status.get("Plan to Watch", [])]
        print(f"\n── Predikce pro PTW seznam ({len(predict_ids)} titulů) ──")

    elif mode == "ids":
        predict_ids = [int(x) for x in args.ids] if args.ids else []
        print(f"\n── Predikce pro zadané ID ({len(predict_ids)} titulů) ──")

    elif mode == "top_mal":
        top_n = pred_cfg.get("top_mal_count", 100)
        print(f"\n── Predikce pro top {top_n} MAL anime ──")
        print("  Stahování top MAL dat…")
        top_data    = jikan.get_top_anime(limit=top_n, min_score=min_mal)
        predict_ids = [
            d["mal_id"] for d in top_data
            if d["mal_id"] not in existing_ids
        ]
        print(f"  Po odfiltrování viděných: {len(predict_ids)} titulů")
    else:
        print(f"Neznámý mód: {mode}", file=sys.stderr)
        sys.exit(1)

    if mode != "top_mal":
        pred_jikan_dict = jikan.get_anime_batch(predict_ids)
        pred_data_list  = [
            v for k, v in pred_jikan_dict.items()
            if v
            and (v.get("score") or 0) >= min_mal
            and k not in {e.mal_id for e in train_entries}
        ]
    else:
        pred_data_list = [d for d in top_data if d["mal_id"] in set(predict_ids)]

    if not pred_data_list:
        print("  Žádná data k predikci.")
        return

    pred_al_dict   = None
    pred_staff_dict = None
    pred_mal_ids = [d["mal_id"] for d in pred_data_list if d.get("mal_id")]
    if fc.anilist.enabled:
        pred_al_dict = anilist.get_anime_batch(pred_mal_ids)
    if fc.staff_directors or fc.staff_writers:
        pred_staff_dict = jikan.get_staff_batch(pred_mal_ids)

    Xp, p_ids, p_titles = build_prediction_matrix(
        pred_data_list, fc, list(X.columns), pred_al_dict, pred_staff_dict
    )
    preds         = predict(results, Xp)
    preds_clipped = np.clip(preds, 1, 10)

    pred_rows = sorted(
        zip(p_ids, p_titles, preds_clipped, pred_data_list),
        key=lambda r: r[2],
        reverse=True,
    )

    print(f"\n{'═'*72}")
    print(f"  TOP {show_top} PREDIKOVANÝCH ANIME")
    print(f"{'═'*72}")

    table = []
    for mal_id, title, pred, data in pred_rows[:show_top]:
        mal_score = data.get("score") or 0
        genres    = ", ".join(g["name"] for g in (data.get("genres") or [])[:3])
        table.append([
            f"{pred:.2f}",
            title[:45],
            f"{mal_score:.2f}" if mal_score else "—",
            genres[:35],
        ])

    print(tabulate(
        table,
        headers=["predikce", "titul", "MAL skóre", "žánry"],
        tablefmt="simple",
    ))
    print()

    output_path = Path("predictions.csv")
    pd.DataFrame(
        [(mid, t, float(p), (d.get("score") or 0))
         for mid, t, p, d in pred_rows],
        columns=["mal_id", "title", "predicted_score", "mal_score"]
    ).to_csv(output_path, index=False, encoding="utf-8")
    print(f"  Výsledky uloženy: {output_path}")


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Anime Taste Model — predikce hodnocení na základě MAL dat"
    )
    parser.add_argument("--config",       default="config.yaml", help="Cesta ke konfiguračnímu souboru")
    parser.add_argument("--mode",         choices=["ptw", "ids", "top_mal"], help="Mód predikce/kandidátů")
    parser.add_argument("--ids",          nargs="+",             help="MAL ID pro mód 'ids'")
    parser.add_argument("--alpha",        type=float,            help="Regularizační síla (přepíše config)")
    parser.add_argument("--explain",      nargs="+",             help="Vysvětli predikci pro daná MAL ID")
    parser.add_argument("--analyze",      action="store_true",   help="Analýza příznaků: koef + příklady titulů")
    parser.add_argument("--cluster",      action="store_true",   help="Cluster analýza vkusu + doporučení per cluster")
    parser.add_argument("--k",            type=int,              help="Počet clusterů (default: 6)")
    parser.add_argument("--train-only",   action="store_true",   help="Jen trénuj model, bez predikce")
    parser.add_argument("--list-tags",    action="store_true",   help="Průzkum AniList tagů + YAML ready-to-paste výstup")
    parser.add_argument("--list-staff",   action="store_true",   help="Průzkum režisérů/scenáristů + YAML ready-to-paste výstup")
    parser.add_argument("--list-mal",     action="store_true",   help="Průzkum MAL příznaků (genres, themes, demographics, sources, types)")
    parser.add_argument("--list-studios", action="store_true",   help="Průzkum AniList animačních studií + YAML ready-to-paste výstup")
    parser.add_argument("--list-tags-n",  type=int,              help="Počet tagů/staffu ve výpisu (default: 80/30)")
    parser.add_argument("--list-tags-min",type=int,              help="Minimální počet titulů pro zahrnutí do YAML (default: auto)")
    parser.add_argument("--cf",           action="store_true",   help="Collaborative filtering — doporučení přes podobné uživatele")
    parser.add_argument("--cf-resample",  action="store_true",   help="Vynutí nový výběr seed uživatelů (ignoruje cache)")
    parser.add_argument("--no-aggregate", action="store_true",   help="Vypni agregaci sérií")
    parser.add_argument("--verbose",      action="store_true",   help="Podrobný výstup")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg = load_config(args.config)
    run(cfg, args)


if __name__ == "__main__":
    main()
