# Anime Taste Model

Analytický systém pro modelování vkusu a predikci hodnocení anime
na základě dat z MyAnimeList.

## Jak to funguje

```
MAL XML export
    ↓
Jikan API (žánry, témata, demografie, zdroj, studio, rok, …)
    ↓
Feature matrix (binární + numerické příznaky)
    ↓
Ridge Regression (lineární model s L2 regularizací)
    ↓
Koeficienty příznaků + cross-validace + predikce
```

**Proč Ridge Regression:**
- Koeficienty jsou přímo interpretovatelné: `Romance = +0.82` znamená,
  že přítomnost žánru Romance zvyšuje predikci o 0.82 bodu (po normalizaci)
- L2 regularizace zabraňuje overfittingu na ~200 trénovacích titulech
- Predikci lze rozložit na příspěvky jednotlivých příznaků

---

## Instalace

```bash
pip install -r requirements.txt
```

Vyžaduje Python 3.10+.

---

## Příprava dat

1. Exportuj MAL seznam: https://myanimelist.net/panel.php?go=export
2. Ulož XML soubor do složky projektu
3. Nastav cestu v `config.yaml`:
   ```yaml
   mal_export: "animelist.xml"
   ```

---

## Spuštění

### Základní spuštění (predikce PTW listu)
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

### Vysvětlení konkrétní predikce
```bash
python main.py --explain 5114
# Zobrazí příspěvek každého příznaku k predikci Steins;Gate
```

### Ladění regularizace
```bash
python main.py --alpha 0.1    # slabší regularizace (může přefitovat)
python main.py --alpha 10.0   # silnější regularizace (konzervativnější)
```

---

## Výstupy

### Evaluace modelu
```
══════════════════════════════════════════════════════════════
  EVALUACE MODELU
══════════════════════════════════════════════════════════════
  Trénovacích titulů:  187
  Počet příznaků:      42

  Cross-validace (Ridge, 1.0):
    RMSE: 0.824 ± 0.091
    MAE:  0.631

  Intercept (baseline): 8.142
══════════════════════════════════════════════════════════════
```

### Koeficienty modelu
```
příznak                koeficient   směr
─────────────────────  ──────────   ────────────────────
genre_Romance           +0.8241     ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
demo_Seinen             +0.4102     ▲▲▲▲▲▲▲
source_Light novel      +0.3817     ▲▲▲▲▲▲
source_Visual novel     +0.3205     ▲▲▲▲▲
genre_Drama             +0.2991     ▲▲▲▲
genre_Ecchi             -0.4130     ▼▼▼▼▼▼▼
theme_Harem             -0.2011     ▼▼▼
```

### Predikce
```
predikce   titul                                      MAL skóre   žánry
─────────  ─────────────────────────────────────────  ─────────   ──────────────────────
9.21       Plastic Memories                            8.30        Drama, Romance, Sci-Fi
8.94       Angel Beats!                                8.13        Action, Drama, Comedy
8.71       Ef - A Tale of Memories                     8.00        Drama, Romance
```

### Vysvětlení predikce
```
příznak              hodnota   koef      příspěvek
───────────────────  ───────   ───────   ─────────
genre_Romance              1   +0.8241   +0.7190
genre_Drama                1   +0.2991   +0.2614
source_Visual novel        1   +0.3205   +0.2801
mal_score               8.30   +0.1923   +0.1541
genre_Ecchi                0   -0.4130   +0.0000
```

---

## Ladění modelu

Edituj `config.yaml`:

```yaml
# Přidej/odeber příznaky
features:
  genres:
    - name: "Romance"
      mal_id: 22
    # - name: "Ecchi"    ← zakomentuj pro vyloučení
    #   mal_id: 9

# Zesil/oslab regularizaci
model:
  alpha: 2.0     # vyšší = konzervativnější koeficienty

# Filtruj trénovací data
training:
  min_score: 5   # ignoruj tituly se skóre pod 5
```

---

## Cache

Jikan API odpovědi jsou uloženy do složky `cache/` jako JSON soubory.
Opakované spuštění je proto rychlé (bez síťových volání).

Pro vymazání cache: `rm -rf cache/`

---

## Soubory

| Soubor             | Účel                                           |
|--------------------|------------------------------------------------|
| `main.py`          | Hlavní skript (CLI + orchestrace)              |
| `config.yaml`      | Konfigurace příznaků, modelu, predikce         |
| `mal_parser.py`    | Parser MAL XML exportu                         |
| `jikan_client.py`  | Jikan API klient (cache + rate limiting)       |
| `feature_builder.py` | Sestavení feature matrix                     |
| `model.py`         | Ridge Regression + evaluace + interpretace     |
| `requirements.txt` | Python závislosti                              |
| `cache/`           | Cache Jikan API odpovědí (vytvoří se auto)     |
| `predictions.csv`  | Export výsledků predikce (vytvoří se po spuštění) |
