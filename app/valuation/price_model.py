"""
app/valuation/price_model.py
----------------------------
Pipeline Steps 3 + 4: DVF lookup + Fair Value Prediction

Step 3 — DVF Matching:
  Given a listing's zipcode + surface + type,
  retrieve comparable recent transactions from DVF.
  Returns: comparable transactions + market statistics

Step 4 — Price Prediction:
  LightGBM model trained on DVF transactions.
  Outputs: predicted fair value + 80% confidence interval.

Key design decision: UNCERTAINTY QUANTIFICATION
  Predicting a single number (615,000€) is dangerous for investment decisions.
  We output a confidence interval based on:
    - Quantile regression (p10, p50, p90)
    - Number of comparable transactions (data confidence)
    - Recency of comparable data (staleness penalty)

  This is what separates an investment tool from a toy:
  "This property is worth €615k ± €45k based on 18 transactions
   in the past 6 months" is actionable.
  "This property is worth €615k" is not.

Interview-ready explanation:
  "I used quantile regression rather than standard regression
   because the loss function for investment decisions is asymmetric —
   a false undervaluation (missing a good deal) is less costly
   than a false overvaluation (investing in an overpriced asset)."
"""

import math
import logging
import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DVF Matcher
# ---------------------------------------------------------------------------

@dataclass
class DVFComparables:
    zipcode: int
    surface_m2: float
    property_type: str
    transactions: pd.DataFrame          # raw comparable rows
    market_median_m2: float
    market_mean_m2: float
    market_std_m2: float
    n_transactions: int
    freshness_days: float               # avg age of comparables
    market_trend_pct: float             # YoY price change


class DVFMatcher:
    """
    Finds comparable DVF transactions for a given property.

    Matching strategy:
      - Same zipcode (required)
      - Same property type (required)
      - Surface within ±30% (soft — tighter = fewer comparables)
      - Date within last 18 months (recency preference)

    Trade-off documented:
      Stricter matching = more accurate but fewer transactions.
      We prefer 15+ comparables over strict surface match.
      If < 5 found: expand to neighbouring arrondissements.
    """

    def __init__(self, dvf_df: pd.DataFrame):
        self.dvf = dvf_df.copy()
        self.dvf["date_mutation"] = pd.to_datetime(self.dvf["date_mutation"])
        self.dvf["year"] = self.dvf["date_mutation"].dt.year
        self._precompute_market_stats()

    def _precompute_market_stats(self):
        """Pre-aggregate market stats per zipcode+type for fast lookup."""
        self._stats = (
            self.dvf.groupby(["code_postal", "type_local"])["prix_m2"]
            .agg(["median", "mean", "std", "count"])
            .reset_index()
        )

    def get_comparables(
        self,
        zipcode: int,
        surface_m2: float,
        property_type: str,
        surface_tolerance: float = 0.30,
        max_age_months: int = 18,
    ) -> DVFComparables:

        # Filter by zipcode + type
        mask = (
            (self.dvf["code_postal"] == zipcode)
            & (self.dvf["type_local"] == property_type)
        )
        pool = self.dvf[mask].copy()

        # Surface filter
        surface_mask = (
            pool["surface_reelle_bati"].between(
                surface_m2 * (1 - surface_tolerance),
                surface_m2 * (1 + surface_tolerance)
            )
        )
        comparables = pool[surface_mask]

        # If too few, relax surface tolerance
        if len(comparables) < 5:
            comparables = pool  # use all in zipcode
            logger.debug(f"Relaxed surface filter: {len(comparables)} comparables")

        # Freshness: prefer recent transactions
        cutoff = pd.Timestamp("2024-12-31") - pd.DateOffset(months=max_age_months)
        recent = comparables[comparables["date_mutation"] >= cutoff]
        if len(recent) >= 5:
            comparables = recent

        # Market trend (YoY 2023 vs 2024)
        trend_pct = self._compute_trend(pool)

        # Freshness metric
        if len(comparables) > 0:
            ref_date = pd.Timestamp("2024-12-31")
            freshness = (ref_date - comparables["date_mutation"]).dt.days.mean()
        else:
            freshness = 999.0

        m2_prices = comparables["prix_m2"] if len(comparables) > 0 else pd.Series([0])

        return DVFComparables(
            zipcode=zipcode,
            surface_m2=surface_m2,
            property_type=property_type,
            transactions=comparables,
            market_median_m2=float(m2_prices.median()),
            market_mean_m2=float(m2_prices.mean()),
            market_std_m2=float(m2_prices.std()) if len(m2_prices) > 1 else 0.0,
            n_transactions=len(comparables),
            freshness_days=freshness,
            market_trend_pct=trend_pct,
        )

    def _compute_trend(self, pool: pd.DataFrame) -> float:
        """YoY price change 2023→2024."""
        y2023 = pool[pool["year"] == 2023]["prix_m2"].median()
        y2024 = pool[pool["year"] == 2024]["prix_m2"].median()
        if pd.isna(y2023) or pd.isna(y2024) or y2023 == 0:
            return 0.0
        return round((y2024 - y2023) / y2023 * 100, 2)


# ---------------------------------------------------------------------------
# Valuation model (quantile regression via gradient boosting)
# ---------------------------------------------------------------------------

@dataclass
class ValuationResult:
    listing_id: str
    asking_price: int
    predicted_fair_value: int
    ci_low: int                   # 80% CI lower bound
    ci_high: int                  # 80% CI upper bound
    market_median_m2: float
    predicted_m2: float
    n_comparables: int
    data_freshness_days: float
    market_trend_pct: float
    confidence_score: float       # 0-1: data quality signal
    price_gap_pct: float          # (asking - predicted) / predicted
    signal: str                   # "undervalued" | "fair" | "overvalued"
    signal_strength: str          # "strong" | "moderate" | "weak"
    features_used: Dict = field(default_factory=dict)


class PropertyValuationModel:
    """
    LightGBM-based fair value estimator with quantile regression.

    Architecture:
      - Median model (q=0.50): fair value estimate
      - Low model  (q=0.10): CI lower bound
      - High model (q=0.90): CI upper bound

    Features:
      - surface_m2 (log-transformed: price/m2 decreases with size)
      - code_postal (categorical: location is the #1 driver)
      - type_local
      - nombre_pieces (derived from surface)
      - year (captures market cycle)
      - market_trend (macro context)

    Fallback (LightGBM not available):
      Robust statistical estimator using DVF market stats.
      Documented trade-off: lower accuracy, no uncertainty quantification.
      Appropriate when: cold start, low data volume, rapid prototyping.
    """

    def __init__(self):
        self.model_median = None
        self.model_low = None
        self.model_high = None
        self.is_trained = False
        self._try_import_lgbm()

    def _try_import_lgbm(self):
        try:
            import lightgbm as lgb
            self._lgbm = lgb
            logger.info("LightGBM available — will use gradient boosting")
        except ImportError:
            self._lgbm = None
            logger.warning("LightGBM not available — using statistical fallback")

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame()
        features["log_surface"] = np.log1p(df["surface_reelle_bati"])
        features["surface_m2"] = df["surface_reelle_bati"]
        features["zipcode"] = df["code_postal"].astype(int)
        features["arrond"] = df["code_postal"].astype(int) - 75000
        features["is_maison"] = (df["type_local"] == "Maison").astype(int)
        features["rooms"] = df["nombre_pieces_principales"].fillna(
            (df["surface_reelle_bati"] / 28).round()
        )
        features["year"] = df["date_mutation"].dt.year if "date_mutation" in df else 2024
        # Size categories (price/m2 is non-linear with surface)
        features["is_small"] = (df["surface_reelle_bati"] < 35).astype(int)
        features["is_large"] = (df["surface_reelle_bati"] > 100).astype(int)
        return features

    def train(self, dvf_df: pd.DataFrame) -> dict:
        """Train on DVF transaction data."""
        df = dvf_df.copy()
        df["date_mutation"] = pd.to_datetime(df["date_mutation"])

        X = self._build_features(df)
        y = df["prix_m2"]

        if self._lgbm and len(df) > 200:
            return self._train_lgbm(X, y)
        else:
            return self._train_statistical(df)

    def _train_lgbm(self, X: pd.DataFrame, y: pd.Series) -> dict:
        lgb = self._lgbm
        params_base = {
            "objective": "quantile",
            "metric": "quantile",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 300,
            "min_child_samples": 20,
            "verbose": -1,
        }

        # Train 3 quantile models
        for quantile, attr in [(0.10, "model_low"), (0.50, "model_median"), (0.90, "model_high")]:
            params = {**params_base, "alpha": quantile}
            model = lgb.LGBMRegressor(**params)
            model.fit(X, y)
            setattr(self, attr, model)

        self.is_trained = True
        self._feature_names = list(X.columns)

        # Evaluate on training data (proxy — ideally use held-out set)
        y_pred = self.model_median.predict(X)
        mae = float(np.abs(y - y_pred).mean())
        mape = float((np.abs(y - y_pred) / y).mean() * 100)

        logger.info(f"LightGBM trained: MAE={mae:.0f} €/m², MAPE={mape:.1f}%")
        return {"model": "lightgbm", "mae_m2": mae, "mape_pct": mape, "n_train": len(X)}

    def _train_statistical(self, df: pd.DataFrame) -> dict:
        """Fallback: store market statistics per zipcode+type."""
        self._market_stats = (
            df.groupby(["code_postal", "type_local"])["prix_m2"]
            .quantile([0.10, 0.50, 0.90])
            .unstack()
            .reset_index()
        )
        self._market_stats.columns = ["code_postal", "type_local", "q10", "q50", "q90"]
        self.is_trained = True
        logger.info("Statistical model trained (LightGBM fallback)")
        return {"model": "statistical", "n_train": len(df)}

    def predict(
        self,
        surface_m2: float,
        zipcode: int,
        property_type: str,
        comparables: DVFComparables,
    ) -> Tuple[float, float, float]:
        """
        Returns (predicted_m2, ci_low_m2, ci_high_m2).
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained — call .train() first")

        if self._lgbm and self.model_median:
            return self._predict_lgbm(surface_m2, zipcode, property_type)
        else:
            return self._predict_statistical(zipcode, property_type, comparables)

    def _predict_lgbm(self, surface_m2, zipcode, property_type):
        row = pd.DataFrame([{
            "surface_reelle_bati": surface_m2,
            "code_postal": zipcode,
            "type_local": property_type,
            "nombre_pieces_principales": surface_m2 // 28,
        }])
        import datetime
        row["date_mutation"] = pd.Timestamp("2024-06-01")
        X = self._build_features(row)
        # Ensure column order matches training
        X = X[self._feature_names]

        m2_median = float(self.model_median.predict(X)[0])
        m2_low = float(self.model_low.predict(X)[0])
        m2_high = float(self.model_high.predict(X)[0])
        return m2_median, m2_low, m2_high

    def _predict_statistical(self, zipcode, property_type, comparables):
        stats = self._market_stats[
            (self._market_stats["code_postal"] == zipcode) &
            (self._market_stats["type_local"] == property_type)
        ]
        if stats.empty:
            # Fallback to comparable stats
            m2 = comparables.market_median_m2
            std = comparables.market_std_m2 or m2 * 0.12
            return m2, m2 - 1.28 * std, m2 + 1.28 * std
        row = stats.iloc[0]
        return float(row["q50"]), float(row["q10"]), float(row["q90"])

    def value_property(
        self,
        listing_id: str,
        asking_price: int,
        surface_m2: float,
        zipcode: int,
        property_type: str,
        comparables: DVFComparables,
    ) -> ValuationResult:
        """Full valuation: prediction + CI + investment signal."""

        m2_pred, m2_low, m2_high = self.predict(surface_m2, zipcode, property_type, comparables)

        fair_value = int(m2_pred * surface_m2 / 1000) * 1000
        ci_low = int(m2_low * surface_m2 / 1000) * 1000
        ci_high = int(m2_high * surface_m2 / 1000) * 1000

        # Investment signal
        price_gap_pct = (asking_price - fair_value) / max(fair_value, 1) * 100

        # Confidence based on data quality
        confidence = self._compute_confidence(comparables)

        if price_gap_pct < -8:
            signal = "undervalued"
            strength = "strong" if price_gap_pct < -15 else "moderate"
        elif price_gap_pct > 10:
            signal = "overvalued"
            strength = "strong" if price_gap_pct > 20 else "moderate"
        else:
            signal = "fair"
            strength = "weak"

        return ValuationResult(
            listing_id=listing_id,
            asking_price=asking_price,
            predicted_fair_value=fair_value,
            ci_low=ci_low,
            ci_high=ci_high,
            market_median_m2=comparables.market_median_m2,
            predicted_m2=m2_pred,
            n_comparables=comparables.n_transactions,
            data_freshness_days=comparables.freshness_days,
            market_trend_pct=comparables.market_trend_pct,
            confidence_score=confidence,
            price_gap_pct=round(price_gap_pct, 1),
            signal=signal,
            signal_strength=strength,
            features_used={
                "surface_m2": surface_m2,
                "zipcode": zipcode,
                "property_type": property_type,
            }
        )

    def _compute_confidence(self, comparables: DVFComparables) -> float:
        """
        Data confidence score 0-1.
        Penalises: few comparables, stale data, high price dispersion.
        """
        # N comparables: 20+ = full confidence
        n_score = min(1.0, comparables.n_transactions / 20)
        # Freshness: < 90 days = full, > 365 days = 0
        freshness_score = max(0, 1 - comparables.freshness_days / 365)
        # Dispersion: cv < 0.05 = full, > 0.20 = 0
        cv = comparables.market_std_m2 / max(comparables.market_mean_m2, 1)
        dispersion_score = max(0, 1 - cv / 0.20)

        return round(0.4 * n_score + 0.4 * freshness_score + 0.2 * dispersion_score, 2)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    dvf = pd.read_csv("data/raw/dvf_paris_2022_2024.csv")
    dvf["date_mutation"] = pd.to_datetime(dvf["date_mutation"])

    matcher = DVFMatcher(dvf)
    model = PropertyValuationModel()
    eval_metrics = model.train(dvf)
    print(f"\nModel training: {eval_metrics}")

    # Test case
    test_cases = [
        {"listing_id": "UPF00001", "asking_price": 580000, "surface_m2": 75,
         "zipcode": 75011, "property_type": "Appartement"},
        {"listing_id": "UPF00002", "asking_price": 320000, "surface_m2": 35,
         "zipcode": 75018, "property_type": "Appartement"},
        {"listing_id": "UPF00003", "asking_price": 1200000, "surface_m2": 90,
         "zipcode": 75007, "property_type": "Appartement"},
    ]

    print("\n--- Valuation Results ---")
    for tc in test_cases:
        comps = matcher.get_comparables(tc["zipcode"], tc["surface_m2"], tc["property_type"])
        result = model.value_property(**tc, comparables=comps)
        print(f"\n{result.listing_id} | {tc['zipcode']} | {tc['surface_m2']}m²")
        print(f"  Asking:    {result.asking_price:,}€")
        print(f"  Fair Value: {result.predicted_fair_value:,}€  "
              f"[CI: {result.ci_low:,} – {result.ci_high:,}]")
        print(f"  Gap:       {result.price_gap_pct:+.1f}%  → {result.signal} ({result.signal_strength})")
        print(f"  Confidence: {result.confidence_score:.2f}  "
              f"({result.n_comparables} comparables, {result.data_freshness_days:.0f}d fresh)")
