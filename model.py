"""
model.py — Trénování, evaluace a interpretace modelu

Model: Ridge Regression (lineární regrese s L2 regularizací)

Proč Ridge a ne složitější model?
  - Koeficienty jsou přímo interpretovatelné: každý příznak má jeden koeficient
  - Po StandardScaleru jsou koeficienty srovnatelné napříč příznaky
  - L2 regularizace zabraňuje overfittingu na malém datasetu (~200 titulů)
  - Predikci lze rozložit na příspěvky jednotlivých příznaků (vysvětlitelnost)

Interpretace koeficientů:
  Po StandardScaleru platí: koeficient = změna predikovaného skóre při
  změně příznaku o 1 std. odchylku (nebo přítomnost/nepřítomnost u binárních)
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tabulate import tabulate

log = logging.getLogger(__name__)


@dataclass
class ModelResults:
    """Výsledky trénování modelu."""
    feature_names:   list[str]
    coefficients:    np.ndarray
    intercept:       float
    scaler:          StandardScaler
    model:           object                    # sklearn model
    cv_rmse_mean:    float
    cv_rmse_std:     float
    cv_mae_mean:     float
    train_rmse:      float
    train_mae:       float
    n_train:         int


def train(
    X: pd.DataFrame,
    y: list[int],
    model_type: str = "ridge",
    alpha: float    = 1.0,
    cv_folds: int   = 5,
) -> ModelResults:
    """
    Natrénuje model a vrátí ModelResults.

    Parametry:
        X          — feature matrix (pandas DataFrame)
        y          — cílová proměnná (skóre uživatele)
        model_type — "ridge" | "lasso" | "linear"
        alpha      — regularizační síla (platí jen pro ridge/lasso)
        cv_folds   — počet foldů pro cross-validaci
    """
    feature_names = list(X.columns)
    y_arr = np.array(y, dtype=float)

    # Normalizace příznaků
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Výběr modelu
    if model_type == "ridge":
        mdl = Ridge(alpha=alpha)
    elif model_type == "lasso":
        mdl = Lasso(alpha=alpha, max_iter=10000)
    else:
        mdl = LinearRegression()

    # Cross-validace
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_neg_rmse = cross_val_score(
        mdl, X_scaled, y_arr, cv=kf,
        scoring="neg_root_mean_squared_error"
    )
    cv_neg_mae = cross_val_score(
        mdl, X_scaled, y_arr, cv=kf,
        scoring="neg_mean_absolute_error"
    )

    # Trénování na celém datasetu
    mdl.fit(X_scaled, y_arr)
    y_pred_train = mdl.predict(X_scaled)

    return ModelResults(
        feature_names  = feature_names,
        coefficients   = mdl.coef_,
        intercept      = mdl.intercept_,
        scaler         = scaler,
        model          = mdl,
        cv_rmse_mean   = float(-cv_neg_rmse.mean()),
        cv_rmse_std    = float(cv_neg_rmse.std()),
        cv_mae_mean    = float(-cv_neg_mae.mean()),
        train_rmse     = float(np.sqrt(mean_squared_error(y_arr, y_pred_train))),
        train_mae      = float(mean_absolute_error(y_arr, y_pred_train)),
        n_train        = len(y),
    )


def predict(
    results: ModelResults,
    X_new: pd.DataFrame,
) -> np.ndarray:
    """Predikuje skóre pro nová anime (před normalizací)."""
    X_scaled = results.scaler.transform(X_new)
    return results.model.predict(X_scaled)


def explain_prediction(
    results: ModelResults,
    x_row: pd.Series,
    predicted_score: float,
) -> pd.DataFrame:
    """
    Rozloží predikci na příspěvky jednotlivých příznaků.

    Vrací DataFrame seřazený podle absolutního příspěvku.
    """
    x_scaled = results.scaler.transform(x_row.values.reshape(1, -1))[0]
    contributions = x_scaled * results.coefficients

    df = pd.DataFrame({
        "příznak":     results.feature_names,
        "hodnota":     x_row.values,
        "koeficient":  results.coefficients,
        "příspěvek":   contributions,
    })
    df = df[df["hodnota"] != 0].copy()   # skryj nulové příznaky
    df = df.reindex(df["příspěvek"].abs().sort_values(ascending=False).index)
    return df


# ── Výstupy na konzoli ─────────────────────────────────────────────────────────

def print_evaluation(res: ModelResults) -> None:
    """Vypíše evaluaci modelu na stdout."""
    print("\n" + "═" * 60)
    print("  EVALUACE MODELU")
    print("═" * 60)
    print(f"  Trénovacích titulů:  {res.n_train}")
    print(f"  Počet příznaků:      {len(res.feature_names)}")
    print()
    print(f"  Cross-validace ({res.model.__class__.__name__}, {res.model.alpha if hasattr(res.model, 'alpha') else '—'}):")
    print(f"    RMSE: {res.cv_rmse_mean:.3f} ± {res.cv_rmse_std:.3f}")
    print(f"    MAE:  {res.cv_mae_mean:.3f}")
    print()
    print(f"  Trénovací data (optimistické):")
    print(f"    RMSE: {res.train_rmse:.3f}")
    print(f"    MAE:  {res.train_mae:.3f}")
    print(f"  Intercept (baseline): {res.intercept:.3f}")
    print("═" * 60)


def print_coefficients(res: ModelResults, top_n: int = 30) -> None:
    """
    Vypíše koeficienty modelu seřazené podle absolutní hodnoty.

    Koeficienty jsou v prostoru StandardScaleru — přímo srovnatelné.
    Pozitivní = zvyšuje skóre, negativní = snižuje skóre.
    """
    coef_df = pd.DataFrame({
        "příznak":    res.feature_names,
        "koeficient": res.coefficients,
    }).reindex(
        pd.Series(res.coefficients).abs().sort_values(ascending=False).index
    ).head(top_n)

    print("\n" + "═" * 60)
    print(f"  KOEFICIENTY MODELU (top {top_n})")
    print(f"  (standardizované — přímo srovnatelné)")
    print("═" * 60)

    rows = []
    for _, row in coef_df.iterrows():
        coef = row["koeficient"]
        bar_len = int(abs(coef) * 8)
        bar = ("▲" * bar_len) if coef > 0 else ("▼" * bar_len)
        rows.append([
            row["příznak"],
            f"{coef:+.4f}",
            bar[:20],
        ])

    print(tabulate(rows, headers=["příznak", "koeficient", "směr"], tablefmt="simple"))
    print()


def print_predicted_vs_actual(
    res: ModelResults,
    X: pd.DataFrame,
    y: list[int],
    mal_ids: list[int],
    titles: dict[int, str],
    n: int = 20,
) -> None:
    """Vypíše predikované vs. skutečné skóre pro trénovací data."""
    X_scaled = res.scaler.transform(X)
    y_pred = res.model.predict(X_scaled)

    rows = []
    for i, (pred, actual, mid) in enumerate(zip(y_pred, y, mal_ids)):
        title = titles.get(mid, f"ID:{mid}")[:40]
        diff  = pred - actual
        rows.append((abs(diff), title, actual, pred, diff))

    rows.sort(key=lambda r: r[0], reverse=True)

    print("\n" + "═" * 60)
    print(f"  NEJVĚTŠÍ ODCHYLKY (predikce vs. skutečnost)")
    print("═" * 60)
    table = [
        [r[1], r[2], f"{r[3]:.2f}", f"{r[4]:+.2f}"]
        for r in rows[:n]
    ]
    print(tabulate(table, headers=["titul", "skutečné", "predikce", "Δ"], tablefmt="simple"))
    print()


def print_feature_analysis(
    res:      ModelResults,
    X:        pd.DataFrame,
    y:        list[int],
    mal_ids:  list[int],
    titles:   dict[int, str],
    top_n:    int = 15,
    examples: int = 5,
) -> None:
    """
    Analýza trénovacích dat: pro každý příznak zobrazí koeficient
    a konkrétní tituly, které ho pohánějí (s jejich skóre).

    Pomáhá odhalit "spurious correlations" — kdy příznak má vysoký
    koeficient ne proto, že je skutečnou příčinou, ale proto, že
    koreluje s jiným faktorem (např. Military koreluje s Youjo Senki,
    které má 9 z jiných důvodů).

    Výstup pro každý příznak:
      - koeficient (standardizovaný)
      - průměrné skutečné skóre titulů S daným příznakem
      - průměrné skutečné skóre titulů BEZ daného příznaku
      - rozdíl průměrů (raw korelace, bez modelové normalizace)
      - příklady titulů: top scoring s příznakem + low scoring s příznakem
    """
    y_arr    = np.array(y, dtype=float)
    titles_v = [titles.get(mid, f"ID:{mid}") for mid in mal_ids]

    # Seřaď příznaky podle abs(koeficient)
    coef_order = np.argsort(np.abs(res.coefficients))[::-1][:top_n]

    print("\n" + "═" * 70)
    print(f"  ANALÝZA PŘÍZNAKŮ (top {top_n} dle abs. koeficientu)")
    print(f"  Pro každý příznak: koef, průměrné skóre s/bez, příklady titulů")
    print("═" * 70)

    for idx in coef_order:
        feat_name = res.feature_names[idx]
        coef      = res.coefficients[idx]
        feat_vals = X.iloc[:, idx].values

        # Tituly s příznakem (hodnota > 0) vs. bez
        mask_with    = feat_vals > 0
        mask_without = feat_vals == 0

        n_with    = mask_with.sum()
        n_without = mask_without.sum()

        avg_with    = float(y_arr[mask_with].mean())    if n_with    > 0 else float("nan")
        avg_without = float(y_arr[mask_without].mean()) if n_without > 0 else float("nan")
        raw_diff    = avg_with - avg_without if not (
            float("nan") in (avg_with, avg_without)
        ) else float("nan")

        sign = "▲" if coef > 0 else "▼"
        print(f"\n  {feat_name}")
        print(f"    Koeficient:  {coef:+.4f} {sign}   "
              f"(n s příznakem: {n_with}, bez: {n_without})")
        if not np.isnan(raw_diff):
            print(f"    Průměr s/bez:  {avg_with:.2f} / {avg_without:.2f}  "
                  f"(raw Δ = {raw_diff:+.2f})")

        if n_with == 0:
            print("    [žádné trénovací tituly s tímto příznakem]")
            continue

        # Příklady: top skóre s příznakem
        with_indices  = np.where(mask_with)[0]
        sorted_desc   = with_indices[np.argsort(y_arr[with_indices])[::-1]]
        sorted_asc    = with_indices[np.argsort(y_arr[with_indices])]

        top_ex = [(titles_v[i][:38], int(y_arr[i]), float(feat_vals[i]))
                  for i in sorted_desc[:examples]]
        bot_ex = [(titles_v[i][:38], int(y_arr[i]), float(feat_vals[i]))
                  for i in sorted_asc[:min(3, len(sorted_asc))]
                  if y_arr[i] < avg_with]  # jen pokud jsou pod průměrem

        print(f"    Nejvyšší skóre s příznakem:")
        for t, s, v in top_ex:
            val_str = f" (rank={v:.2f})" if v not in (0.0, 1.0) else ""
            print(f"      {s}  {t}{val_str}")

        if bot_ex:
            print(f"    Nejnižší skóre s příznakem (potenciální šum):")
            for t, s, v in bot_ex:
                val_str = f" (rank={v:.2f})" if v not in (0.0, 1.0) else ""
                print(f"      {s}  {t}{val_str}")

    print()
