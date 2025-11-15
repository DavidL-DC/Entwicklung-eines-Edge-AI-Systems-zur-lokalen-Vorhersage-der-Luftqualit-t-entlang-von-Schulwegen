import argparse
import datetime as dt
import json
import sys
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

OPENAQ_ENDPOINT = "https://api.openaq.org/v3/measurements"

OPENAQ_API_KEY = "_API_KEY_FALLS_VORHANDEN" # Erfordert Registrierung auf https://explore.openaq.org/register

USE_OFFLINE = True  # für Präsentation


def fetch_openaq_measurements(
    parameter: str = "no2",
    city: Optional[str] = "Munich",
    country: Optional[str] = "DE",
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    limit: int = 10000,
) -> pd.DataFrame:

    params = {
        "parameter": parameter.lower(),
        "limit": limit,
        "sort": "desc",  # neueste zuerst
        "order_by": "datetime",
    }

    if date_from:
        params["date_from"] = date_from
    if date_to:
        params["date_to"] = date_to

    if bbox:
        # south,west,north,east
        s, w, n, e = bbox
        params["bbox"] = f"{s},{w},{n},{e}"
    else:
        if city:
            params["city"] = city
        if country:
            params["country"] = country

    try:
        headers = {
            "X-API-Key": OPENAQ_API_KEY
        }

        r = requests.get(OPENAQ_ENDPOINT, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[ERROR] OpenAQ Abruf fehlgeschlagen: {e}", file=sys.stderr)
        return pd.DataFrame()

    results = data.get("results", [])
    if not results:
        return pd.DataFrame()

    rows = []
    for it in results:
        date_info = it.get("date", {})
        utc = date_info.get("utc")
        if not utc:
            continue
        rows.append({
            "datetime": pd.to_datetime(utc),
            "value": it.get("value"),
            "unit": it.get("unit"),
            "lon": (it.get("coordinates") or {}).get("longitude"),
            "lat": (it.get("coordinates") or {}).get("latitude"),
            "location": it.get("location"),
            "city": it.get("city"),
        })

    df = pd.DataFrame(rows).dropna(subset=["datetime", "value"])
    df = df.sort_values("datetime")
    return df


def to_hourly_series(df: pd.DataFrame) -> pd.DataFrame:
    # Aggregiert zu stündlichen Mittelwerten.
    s = (
        df.set_index("datetime")["value"]
        .resample("1H")
        .mean()
        .to_frame(name="target")
    )
    return s

def add_time_features(s: pd.DataFrame) -> pd.DataFrame:
    # Fügt Zeitmerkmale hinzu: hour, weekday, weekend.
    X = s.copy()
    X["hour"] = X.index.hour
    X["weekday"] = X.index.weekday
    X["weekend"] = (X["weekday"] >= 5).astype(int)
    return X

def add_lag_rolling_features(s: pd.DataFrame, lag_hours: int = 1, rolling_hours: int = 3) -> pd.DataFrame:
    # Fügt Lag- und Rolling-Features hinzu.
    X = s.copy()
    X[f"lag{lag_hours}"] = X["target"].shift(lag_hours)
    X[f"roll{rolling_hours}"] = X["target"].rolling(rolling_hours, min_periods=1).mean()
    return X

def build_feature_matrix(hourly: pd.DataFrame) -> pd.DataFrame:
    # Kombiniert alle Features; droppt Zeilen mit NaN nach Lags.
    X = hourly.pipe(add_time_features).pipe(add_lag_rolling_features, 1, 3)
    X = X.dropna()
    return X

def time_train_test_split(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Zeitbasierter Split
    n = len(df)
    split = int(n * (1 - test_size))
    return df.iloc[:split, :].copy(), df.iloc[split:, :].copy()

def train_linear_regression(X: pd.DataFrame) -> Tuple[Pipeline, pd.DataFrame, pd.DataFrame]:
    # Ziel und Features trennen
    y = X["target"].copy()
    features = X.drop(columns=["target"])

    # Zeitbasierter Split
    Xy = features.copy()
    Xy["target"] = y.values
    train, test = time_train_test_split(Xy, test_size=0.2)
    X_train, y_train = train.drop(columns=["target"]), train["target"]
    X_test, y_test = test.drop(columns=["target"]), test["target"]

    numeric_cols = list(X_train.columns)

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=True, with_std=True), numeric_cols),
        ],
        remainder="drop"
    )

    pipe = Pipeline(steps=[
        ("pre", pre),
        ("reg", LinearRegression()),
    ])

    pipe.fit(X_train, y_train)
    return pipe, (X_test, y_test), (X_train, y_train)

def evaluate_and_plot(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, out_png: str = "evaluation_timeseries.png") -> Tuple[float, float]:
    # Berechnet MAE/RMSE und erzeugt Zeitreihendiagramm (Ist vs. Vorhersage).
    y_pred = pd.Series(model.predict(X_test), index=y_test.index)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Graph: Zeitverlauf (Ist vs. Vorhersage)
    plt.figure(figsize=(10, 4.5))
    y_test.plot(label="Ist (Stickstoffdioxid)", linewidth=1.5)
    y_pred.plot(label="Vorhersage", linewidth=1.2)
    plt.title("Zeitverlauf: Stickstoffdioxid – Ist vs. Vorhersage (Testset)")
    plt.xlabel("Zeit")
    plt.ylabel("Stickstoffdioxid (µg/m³)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    print(f"[INFO] MAE:  {mae:.3f}")
    print(f"[INFO] RMSE: {rmse:.3f}")
    print(f"[INFO] Plot gespeichert unter: {out_png}")
    return mae, rmse

def persist_model(model: Pipeline, path: str = "model_no2_edge.pkl") -> None:
    # Speichert Modell für lokale Edge-Nutzung.
    joblib.dump(model, path)
    print(f"[INFO] Modell gespeichert unter: {path}")



def parse_bbox(bbox_str: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    if not bbox_str:
        return None
    try:
        parts = [float(x.strip()) for x in bbox_str.split(",")]
        assert len(parts) == 4
        return tuple(parts)  # south, west, north, east
    except Exception:
        raise ValueError("bbox muss 'south,west,north,east' sein, z. B. '48.045,11.43,48.22,11.68'")

def main():
    parser = argparse.ArgumentParser(description="Edge-AI Luftqualitätsvorhersage (OpenAQ, München, NO₂)")
    parser.add_argument("--city", type=str, default="Munich", help="Stadtname (OpenAQ City, Standard: Munich)")
    parser.add_argument("--country", type=str, default="DE", help="Ländercode (Standard: DE)")
    parser.add_argument("--parameter", type=str, default="no2", choices=["no2", "pm25", "pm10", "o3"], help="Luftschadstoff (Standard: no2)")
    parser.add_argument("--days", type=int, default=21, help="Tage Rückschau (Standard: 21)")
    parser.add_argument("--bbox", type=str, default=None, help="Optional: Bounding Box 'south,west,north,east' (überschreibt city/country)")
    parser.add_argument("--limit", type=int, default=10000, help="Max. Messungen (Standard: 10000)")
    args = parser.parse_args()

    if USE_OFFLINE:
        print("[INFO] Offline-Modus aktiv: CSV wird geladen.")
        df = pd.read_csv("data_sample.csv", parse_dates=["datetime"], sep=";", decimal=",")
    else:
        # Zeitraum bestimmen (UTC)
        date_to = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
        date_from = date_to - dt.timedelta(days=args.days)
        df = fetch_openaq_measurements(
            parameter=args.parameter,
            city=args.city,
            country=args.country,
            date_from=date_from.isoformat() + "Z",
            date_to=date_to.isoformat() + "Z",
            bbox=parse_bbox(args.bbox),
            limit=args.limit,
        )

    if df.empty:
        print("[WARN] Keine Daten erhalten. Prüfe Parameter (--city/--bbox/--parameter) oder Zeitraum (--days).\n"
            "Tipp: Nutze --bbox für München, z. B.: --bbox \"48.045,11.43,48.22,11.68\"")
        sys.exit(1)

    print(f"[INFO] Rohdaten: {len(df)} Messwerte, Zeitraum {df['datetime'].min()} → {df['datetime'].max()}")

    # Stündlich aggregieren und Features bauen
    hourly = to_hourly_series(df)
    X = build_feature_matrix(hourly)
    if X.empty or X.shape[0] < 48:
        print("[WARN] Zu wenige Daten nach Aggregation/Features. Erhöhe --days oder prüfe Parameter.")
        sys.exit(1)

    # Trainieren + Evaluieren
    model, (X_test, y_test), _ = train_linear_regression(X)
    mae, rmse = evaluate_and_plot(model, X_test, y_test, out_png="evaluation_timeseries.png")

    # Modell speichern
    persist_model(model, path=f"model_{args.parameter}_edge.pkl")

    # Beispiel: Vorhersage für die nächste Stunde 
    last_row = X.iloc[[-1]].drop(columns=["target"])
    next_pred = float(model.predict(last_row)[0])
    print(f"[INFO] Beispiel-Vorhersage (nächste Stunde, heuristisch): {next_pred:.2f} (Einheit wie Rohdaten)")

if __name__ == "__main__":
    main()
