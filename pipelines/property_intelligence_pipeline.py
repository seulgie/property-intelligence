"""
pipelines/property_intelligence_pipeline.py
--------------------------------------------
End-to-end Property Intelligence Agent orchestration.

Pipeline:
  1. Ingest & Deduplicate listings
  2. Extract / recover addresses
  3. Match DVF comparables
  4. Predict fair value (LightGBM + CI)
  5. Generate investment memo (LLM or template)

Design: each step is independently testable and replaceable.
The pipeline runner is a thin orchestrator — no business logic here.

This mirrors production orchestration (Airflow/Dagster DAGs)
where each task has clear inputs, outputs, and failure handling.
"""

import logging
import sys
import os
import json
import time
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestion.deduplicator import run_ingestion, deduplicate_listings
from app.extraction.address_extractor import enrich_addresses
from app.valuation.price_model import DVFMatcher, PropertyValuationModel
from app.reporting.investment_report import InvestmentReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

PROP_TYPE_MAP = {"appartement": "Appartement", "maison": "Maison"}


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    listings_path: str = "data/raw/listings_raw.csv",
    dvf_path: str = "data/raw/dvf_paris_2022_2024.csv",
    output_dir: str = "data/processed",
    llm_client=None,
    max_listings: int = None,
) -> pd.DataFrame:

    os.makedirs(output_dir, exist_ok=True)
    t_start = time.time()

    # --- Step 1: Ingest + Deduplicate ---
    logger.info("=== STEP 1: Ingestion + Deduplication ===")
    raw_df = pd.read_csv(listings_path)
    clean_df, dup_log = deduplicate_listings(raw_df)
    clean_df.to_csv(f"{output_dir}/listings_clean.csv", index=False)
    dup_log.to_csv(f"{output_dir}/dedup_log.csv", index=False)
    logger.info(f"  {len(raw_df)} raw → {len(clean_df)} clean ({len(dup_log)} duplicates removed)")

    if max_listings:
        clean_df = clean_df.head(max_listings)
        logger.info(f"  Limited to {max_listings} listings for demo")

    # --- Step 2: Address extraction ---
    logger.info("=== STEP 2: Address Extraction ===")
    addressed_df = enrich_addresses(clean_df, llm_client=llm_client)
    addressed_df.to_csv(f"{output_dir}/listings_addressed.csv", index=False)

    # --- Step 3+4: Load DVF + Train model ---
    logger.info("=== STEP 3+4: DVF Matching + Model Training ===")
    dvf_df = pd.read_csv(dvf_path)
    dvf_df["date_mutation"] = pd.to_datetime(dvf_df["date_mutation"])

    matcher = DVFMatcher(dvf_df)
    model = PropertyValuationModel()
    train_metrics = model.train(dvf_df)
    logger.info(f"  Model: {train_metrics}")

    # --- Step 5: Generate reports ---
    logger.info("=== STEP 5: Valuation + Report Generation ===")
    reporter = InvestmentReportGenerator(llm_client=llm_client)

    results = []
    for _, listing in addressed_df.iterrows():
        zipcode = int(listing.get("address_zipcode") or listing.get("code_postal", 75011))
        if not str(zipcode).startswith("75"):
            zipcode = int(listing.get("code_postal", 75011))

        surface = float(listing["surface_m2"])
        prop_type = "Appartement"  # default; would parse from description in production

        # DVF comparables
        comps = matcher.get_comparables(zipcode, surface, prop_type)

        if comps.n_transactions < 3:
            logger.debug(f"Skipping {listing['listing_id']}: insufficient DVF data")
            continue

        # Valuation
        valuation = model.value_property(
            listing_id=listing["listing_id"],
            asking_price=int(listing["asking_price"]),
            surface_m2=surface,
            zipcode=zipcode,
            property_type=prop_type,
            comparables=comps,
        )

        # Report
        report = reporter.generate(
            valuation=valuation,
            comparables=comps,
            listing_metadata=listing.to_dict(),
        )

        results.append({
            "listing_id": listing["listing_id"],
            "address": listing.get("address_clean", ""),
            "zipcode": zipcode,
            "surface_m2": surface,
            "asking_price": valuation.asking_price,
            "fair_value": valuation.predicted_fair_value,
            "ci_low": valuation.ci_low,
            "ci_high": valuation.ci_high,
            "predicted_m2": round(valuation.predicted_m2, 0),
            "market_m2": round(comps.market_median_m2, 0),
            "price_gap_pct": valuation.price_gap_pct,
            "signal": valuation.signal,
            "signal_strength": valuation.signal_strength,
            "confidence_score": valuation.confidence_score,
            "n_comparables": comps.n_transactions,
            "freshness_days": round(comps.freshness_days, 0),
            "trend_pct": comps.market_trend_pct,
            "executive_summary": report.executive_summary,
            "recommendation": report.recommendation,
            "report_full": report.to_text(),
            "generated_by": report.generated_by,
            # Ground truth (for evaluation)
            "pricing_bias_gt": listing.get("pricing_bias", "unknown"),
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{output_dir}/valuation_results.csv", index=False)

    elapsed = round(time.time() - t_start, 1)
    logger.info(
        f"Pipeline complete: {len(results_df)} properties valued in {elapsed}s | "
        f"signals: {results_df['signal'].value_counts().to_dict()}"
    )
    return results_df


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

def print_sample_reports(results_df: pd.DataFrame, n: int = 3):
    print(f"\n{'='*60}")
    print(f"PROPERTY INTELLIGENCE AGENT — SAMPLE REPORTS")
    print(f"{'='*60}")

    # Show one of each signal type
    shown = 0
    for signal in ["undervalued", "overvalued", "fair"]:
        subset = results_df[results_df["signal"] == signal]
        if subset.empty:
            continue
        row = subset.sort_values("confidence_score", ascending=False).iloc[0]
        print(f"\n{'─'*60}")
        print(row["report_full"])
        shown += 1
        if shown >= n:
            break


def evaluate_signal_accuracy(results_df: pd.DataFrame):
    """
    Check if model signal matches ground truth pricing bias.
    Ground truth was injected at data generation time.
    """
    if "pricing_bias_gt" not in results_df.columns:
        return

    gt_map = {"over": "overvalued", "under": "undervalued", "fair": "fair"}
    results_df = results_df.copy()
    results_df["signal_gt"] = results_df["pricing_bias_gt"].map(gt_map)

    valid = results_df.dropna(subset=["signal_gt"])
    correct = (valid["signal"] == valid["signal_gt"]).sum()
    total = len(valid)

    print(f"\n{'='*50}")
    print(f"SIGNAL ACCURACY EVALUATION")
    print(f"{'='*50}")
    print(f"Overall accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"\nConfusion matrix:")
    print(pd.crosstab(
        valid["signal_gt"],
        valid["signal"],
        rownames=["Ground Truth"],
        colnames=["Predicted"],
    ))
    print(f"\nNote: 'fair' bucket is large (~75% of data) and has wide CI overlap.")
    print(f"Precision on overvalued detection is more relevant for investment use case.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=50, help="Max listings to process")
    args = parser.parse_args()

    results = run_pipeline(
        listings_path="data/raw/listings_raw.csv",
        dvf_path="data/raw/dvf_paris_2022_2024.csv",
        output_dir="data/processed",
        llm_client=None,    # template mode
        max_listings=args.max,
    )

    print_sample_reports(results, n=3)
    evaluate_signal_accuracy(results)

    print(f"\n\nFull results saved to: data/processed/valuation_results.csv")
    print(f"Duplicate log saved to: data/processed/dedup_log.csv")
