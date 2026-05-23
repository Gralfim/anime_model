#!/usr/bin/env python3
"""
cf_debug.py — Diagnostika CF pipeline

Spuštění:
    python cf_debug.py

Zkontroluje každý krok CF pipeline a vypíše kde se ztrácí uživatelé.
"""
import json, sys, time
from pathlib import Path
from collections import Counter

import yaml

# ── Načti config a data ───────────────────────────────────────────────────────
cfg  = yaml.safe_load(open("config.yaml", encoding="utf-8"))
cf_c = cfg.get("cf", {})

CACHE_CF    = Path(cfg["cache_dir"]) / "cf"
CACHE_JIKAN = Path(cfg["cache_dir"]) / "jikan"

def load_json(path):
    if Path(path).exists():
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return None

# ── 1. Kolik seed uživatelů máme v cache? ─────────────────────────────────────
print("\n═══ 1. Seed uživatelé (userupdates cache) ═══")
seed_files = list(CACHE_CF.glob("userupdates_*.json")) if CACHE_CF.exists() else []
total_users = 0
user_counts = []
for f in seed_files:
    users = json.loads(f.read_text(encoding="utf-8"))
    mal_id = f.stem.replace("userupdates_", "").split("_p")[0]
    user_counts.append((len(users), mal_id))
    total_users += len(users)

user_counts.sort(reverse=True)
for cnt, mid in user_counts[:15]:
    jf = CACHE_JIKAN / f"anime_{mid}_full.json"
    title = "?"
    if jf.exists():
        d = json.loads(jf.read_text())
        title = (d.get("data") or {}).get("title", "?")[:40]
    print(f"  {cnt:4d} uživatelů  MAL {mid:6s}  {title}")

# Unikátní uživatelé
all_users: set[str] = set()
for f in seed_files:
    all_users.update(json.loads(f.read_text(encoding="utf-8")))
print(f"\n  Celkem: {len(seed_files)} seed titulů, {len(all_users)} unikátních uživatelů")

# ── 2. Kolik animlistů máme v cache? ─────────────────────────────────────────
print("\n═══ 2. Animlisty uživatelů (cache) ═══")
al_files = list(CACHE_CF.glob("user_animelist_*.json")) if CACHE_CF.exists() else []
sizes = []
for f in al_files:
    data = json.loads(f.read_text(encoding="utf-8"))
    sizes.append(len(data))

if sizes:
    sizes.sort(reverse=True)
    print(f"  Staženo animlistů: {len(sizes)}")
    print(f"  Průměr titulů/uživatel: {sum(sizes)/len(sizes):.0f}")
    print(f"  Min/Max: {min(sizes)}/{max(sizes)}")
    print(f"  Distribuce (počet titulů):")
    buckets = Counter()
    for s in sizes:
        if s < 20:   buckets["<20"] += 1
        elif s < 50: buckets["20-50"] += 1
        elif s < 100: buckets["50-100"] += 1
        elif s < 200: buckets["100-200"] += 1
        else:        buckets["200+"] += 1
    for k in ["<20","20-50","50-100","100-200","200+"]:
        print(f"    {k:10s}: {buckets[k]:4d} uživatelů")
else:
    print("  Žádné animlisty v cache!")

# ── 3. Simulace výpočtu podobnosti na cached datech ──────────────────────────
print("\n═══ 3. Výpočet korelací (na cached datech) ═══")

# Načti tvůj MAL export
from mal_parser import parse_export
entries, _ = parse_export(cfg["mal_export"])
my_ratings = {e.mal_id: e.score for e in entries if e.score > 0}

# MAL skóre z Jikan cache
my_mal_scores = {}
for mid in my_ratings:
    jf = CACHE_JIKAN / f"anime_{mid}_full.json"
    if jf.exists():
        d = json.loads(jf.read_text())
        score = (d.get("data") or {}).get("score")
        if score:
            my_mal_scores[mid] = float(score)

print(f"  Tvoje hodnocení: {len(my_ratings)}, z toho s MAL skóre: {len(my_mal_scores)}")

import numpy as np

correlations = []
overlap_counts = []
zero_overlap = 0
low_overlap  = 0

for f in al_files[:500]:  # max 500 pro rychlost
    other = json.loads(f.read_text(encoding="utf-8"))
    other_dict = {e["mal_id"]: e["score"] for e in other
                  if e.get("mal_id") and e.get("score", 0) > 0}

    common = {m for m in my_ratings if m in other_dict and my_mal_scores.get(m, 0) > 0}

    if len(common) == 0:
        zero_overlap += 1
        continue
    if len(common) < 5:
        low_overlap += 1
        continue

    overlap_counts.append(len(common))

    my_d    = np.array([my_ratings[m]  - my_mal_scores[m]     for m in common])
    other_d = np.array([other_dict[m]  - my_mal_scores.get(m, other_dict[m]) for m in common])

    my_c    = my_d    - my_d.mean()
    other_c = other_d - other_d.mean()
    denom   = np.sqrt((my_c**2).sum() * (other_c**2).sum())
    if denom < 1e-9:
        continue
    r = float(np.dot(my_c, other_c) / denom)
    correlations.append((r, len(common), f.stem.replace("user_animelist_", "")))

print(f"  Nulový překryv: {zero_overlap}, malý překryv (<5): {low_overlap}")
if overlap_counts:
    print(f"  Průměrný překryv: {np.mean(overlap_counts):.1f} titulů")
    print(f"  Median překryv:   {np.median(overlap_counts):.1f} titulů")

correlations.sort(reverse=True)
print(f"\n  Distribuce korelací (ze {len(correlations)} uživatelů s překryvem ≥5):")
buckets_r = Counter()
for r, _, _ in correlations:
    if r >= 0.4:  buckets_r["≥0.40"] += 1
    elif r >= 0.3: buckets_r["0.30-0.40"] += 1
    elif r >= 0.2: buckets_r["0.20-0.30"] += 1
    elif r >= 0.1: buckets_r["0.10-0.20"] += 1
    elif r >= 0.0: buckets_r["0.00-0.10"] += 1
    else:          buckets_r["<0.00"] += 1
for k in ["≥0.40","0.30-0.40","0.20-0.30","0.10-0.20","0.00-0.10","<0.00"]:
    print(f"    {k:12s}: {buckets_r[k]:4d} uživatelů")

print(f"\n  Top 20 nejpodobnějších uživatelů:")
for r, overlap, username in correlations[:20]:
    print(f"    r={r:+.3f}  overlap={overlap:3d}  {username}")

# ── 4. Doporučení ─────────────────────────────────────────────────────────────
print("\n═══ 4. Doporučení pro parametry ═══")

n_with_overlap = len(overlap_counts)
n_above_015 = sum(1 for r, _, _ in correlations if r >= 0.15)
n_above_010 = sum(1 for r, _, _ in correlations if r >= 0.10)

if len(al_files) == 0:
    print("  ⚠ Žádné animlisty v cache — CF ještě nestáhl data uživatelů")
    print("    Spusť: python main.py --cf  a počkej na dokončení stahování")
elif n_with_overlap < 10:
    print(f"  ⚠ Jen {n_with_overlap} uživatelů s překryvem — seed uživatelé mají moc odlišné listy")
    print("    → Zvyš users_per_seed nebo seed_titles_count")
    print("    → Nebo snižuj seed_min_score na 6")
elif n_above_015 < 5:
    print(f"  ⚠ Jen {n_above_015} uživatelů s r≥0.15 (ale {n_above_010} s r≥0.10)")
    print(f"    → Zkus min_correlation: {0.08 if n_above_010 > 10 else 0.05}")
    print(f"    → Nebo min_overlap: {max(5, int(np.median(overlap_counts))-5) if overlap_counts else 5}")
else:
    print(f"  ✓ {n_above_015} uživatelů s r≥0.15 — CF by měl fungovat")
    print(f"    Zkontroluj min_overlap (aktuálně v configu: "
          f"{cf_c.get('min_overlap', '?')}) vs median překryv "
          f"({np.median(overlap_counts):.0f if overlap_counts else '?'})")
