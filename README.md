# Anime Taste Model

Analytický systém pro modelování vkusu a predikci hodnocení anime na základě dat z MyAnimeList. Kombinuje content-based filtrování (atributy titulů z Jikan API a AniList) s Ridge Regression modelem a shlukovou analýzou vkusu.

## Architektura

```
MAL XML export
    ↓
mal_parser.py       — parsování seznamu a hodnocení
    ↓
jikan_client.py     — žánry, témata, staff, relace sérií (Jikan API)
anilist_client.py   — granulární tagy, studia, composite score (AniList GraphQL)
    ↓
series_aggregator.py — kolaps sequel/prequel sérií na jeden záznam (max skóre)
    ↓
feature_builder.py  — feature matrix (binární + spojité + numerické příznaky)
    ↓
model.py            — Ridge Regression, cross-validace, interpretace koeficientů
    ↓
cluster_analyzer.py — K-Means shlukování, detekce synergií, per-cluster doporučení
```

**Proč Ridge Regression:** koeficienty jsou po StandardScaleru přímo srovnatelné — `genre_Romance = +0.82` a `genre_Ecchi = -0.41` lze porovnat napříč příznaky. L2 regularizace zabraňuje overfittingu na ~200 trénovacích titulech.

---

## Instalace

```bash
pip install -r requirements.txt
```

Vyžaduje Python 3.10+.

---

## Příprava dat

1. Exportuj MAL seznam: https://myanimelist.net/panel.php?go=export
2. Ulož XML soubor do složky projektu jako `animelist.xml`
3. Ověř cestu v `config.yaml`:
   ```yaml
   mal_export: "animelist.xml"
   ```

---

## Spuštění — přehled příkazů

### Základní predikce PTW listu
```bash
python main.py
```

### Predikce pro konkrétní MAL ID
```bash
python main.py --mode ids --ids 5114 32979 6547
# Steins;Gate, Plastic Memories, Angel Beats!
```

### Predikce pro top MAL anime (co ještě nemáš)
```bash
python main.py --mode top_mal
```

### Jen trénink + evaluace, bez predikce
```bash
python main.py --train-only
```

### Vysvětlení predikce pro konkrétní titul
```bash
python main.py --explain 5114
# Zobrazí příspěvek každého příznaku k predikci Steins;Gate
```

### Analýza příznaků trénovacích dat
```bash
python main.py --analyze
# Pro každý příznak: koeficient + konkrétní tituly které ho pohánějí
# Pomáhá odhalit spurious correlations (např. proč Military má kladný koeficient)
```

### Cluster analýza vkusu + per-cluster doporučení
```bash
python main.py --cluster          # K=6 (default), kandidáti z PTW
python main.py --cluster --k 5    # jiný počet clusterů
python main.py --cluster --mode ids --ids 5114 32979
```

### Průzkum AniList tagů (výstup ready-to-paste do config.yaml)
```bash
python main.py --list-tags
python main.py --list-tags --list-tags-min 5   # práh: min. 5 titulů
python main.py --list-tags --list-tags-n 100   # zobraz top 100 tagů
```

### Průzkum režisérů a scenáristů (výstup ready-to-paste do config.yaml)
```bash
python main.py --list-staff
python main.py --list-staff --list-tags-min 3  # jen ti s 3+ tituly
```

### Vypnutí agregace sérií
```bash
python main.py --no-aggregate   # každá řada série jako samostatný záznam
```

### Ladění regularizace
```bash
python main.py --alpha 0.1    # slabší regularizace
python main.py --alpha 10.0   # silnější regularizace
```

---

## Výstupy

### Evaluace modelu
```
══════════════════════════════════════════════════════════════
  EVALUACE MODELU
══════════════════════════════════════════════════════════════
  Trénovacích titulů:  143   (po agregaci sérií)
  Počet příznaků:      78    (MAL + AniList tagy + staff)

  Cross-validace (Ridge, alpha=1.0):
    RMSE: 0.81 ± 0.09
    MAE:  0.62

  Intercept (baseline): 8.14
```

### Koeficienty modelu
```
příznak                    koeficient   směr
─────────────────────────  ──────────   ────────────────
genre_Romance               +0.8241     ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
anilist_Tearjerker          +0.6103     ▲▲▲▲▲▲▲▲▲▲▲▲
writer_Maeda_Jun            +0.5821     ▲▲▲▲▲▲▲▲▲▲▲
source_Visual novel         +0.3205     ▲▲▲▲▲
genre_Drama                 +0.2991     ▲▲▲▲
genre_Ecchi                 -0.4130     ▼▼▼▼▼▼▼
```

### Cluster analýza
```
CLUSTER 1: Romance + Drama + Tearjerker + LN ✦
────────────────────────────────────────────────
Titulů: 18  |  Tvůj průměr: 9.3  |  MAL průměr: 8.1  |  Diferenciál: +1.2 ▲▲▲

Synergie:  Romance + Tearjerker: diff=+1.8  (synergie: +0.7)

Doporučení:
  9.2   White Album 2              8.31   97%
  8.9   Ef - A Tale of Memories    8.00   94%
```

---

## Konfigurace — klíčové sekce config.yaml

### Příznaky
```yaml
features:
  genres:
    - name: "Romance"
      mal_id: 22
    # - name: "Ecchi"    ← zakomentuj pro vyloučení
    #   mal_id: 9

  themes:
    - name: "Military"
      mal_id: 38
      skip_if_anilist: true   # AniList ekvivalent přebírá tento příznak
                               # → zabraňuje double-countingu

  numeric:
    mal_score:
      include: true
    composite_score:
      include: false   # vážený průměr MAL + AniList; zapni pokud máš AniList data

  staff:
    directors:
      - name: "Andou_Masahiro"   # Seishun Buta Yarou, Sousou no Frieren
        mal_id: 25805
    writers:
      - name: "Maeda_Jun"        # Clannad, Angel Beats!
        mal_id: 2910
```

### AniList tagy
```yaml
anilist:
  enabled: true
  use_rank: true        # spojitý rank 0–1 (přesnější než binárně)
  min_rank: 0
  tags:
    - "Tsundere"        # archetyp postavy
    - "Love Triangle"   # narativní vzor
    - "Tearjerker"      # emocionální tón
    # ... (průzkum: python main.py --list-tags)
  studios:
    - "Kyoto Animation"
    - "Shaft"
```

### Agregace sérií
```yaml
training:
  aggregate_series: true   # sequel/prequel → jeden záznam s max skóre
                            # doporučeno: zabraňuje umělé inflaci dat
```

### Cluster analýza
```yaml
cluster:
  k: 6                  # počet clusterů
  top_per_cluster: 8    # doporučení na cluster
```

---

## Soubory

| Soubor                  | Účel                                                                 |
|-------------------------|----------------------------------------------------------------------|
| `main.py`               | CLI orchestrátor — vstupní bod, propojuje všechny moduly             |
| `config.yaml`           | Konfigurace příznaků, modelu, AniList tagů, staff, predikce          |
| `mal_parser.py`         | Parser MAL XML exportu → list `MalEntry` dataclass                  |
| `jikan_client.py`       | Jikan API v4 klient (cache, rate limiting, staff průzkum)            |
| `anilist_client.py`     | AniList GraphQL klient (batch query, tag průzkum, composite score)   |
| `feature_builder.py`    | Feature matrix: MAL žánry + AniList tagy + staff + numerické         |
| `series_aggregator.py`  | Union-Find přes sequel/prequel vazby → kolaps sérií na max skóre    |
| `model.py`              | Ridge Regression, cross-validace, koeficienty, explain, analýza      |
| `cluster_analyzer.py`   | K-Means, diferenciální skóre, synergie, per-cluster doporučení       |
| `requirements.txt`      | Python závislosti                                                    |
| `cache/`                | Cache API odpovědí — vytvoří se automaticky, **není v gitu**         |
| `animelist.xml`         | Tvůj MAL export — **není v gitu** (osobní data)                     |
| `predictions.csv`       | Výstup predikce — generuje se při spuštění                          |
| `clusters.csv`          | Výstup cluster analýzy — generuje se při `--cluster`                |

---

## Doporučený postup prvního spuštění

```bash
# 1. Průzkum dostupných AniList tagů pro tvá data
python main.py --list-tags

# 2. Průzkum režisérů a scenáristů
python main.py --list-staff

# 3. Uprav config.yaml — přidej/odeber tagy a staff dle výsledků

# 4. Trénink + evaluace
python main.py --train-only

# 5. Analýza příznaků — zkontroluj koeficienty a jejich zdůvodnění
python main.py --train-only --analyze

# 6. Cluster analýza + doporučení
python main.py --cluster

# 7. Plná predikce PTW
python main.py
```

---

## Cache

Jikan a AniList odpovědi jsou cachované v `cache/jikan/` a `cache/anilist/` jako JSON. Opakované spuštění je okamžité.

```bash
rm -rf cache/          # smaž celou cache (vynutí nové stažení)
rm -rf cache/anilist/  # smaž jen AniList cache
```
