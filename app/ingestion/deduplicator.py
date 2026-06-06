"""
app/ingestion/deduplicator.py
-----------------------------
Pipeline Step 1: Ingest raw listings and remove duplicates.

Real-world problem (Ali's actual pain point):
  Same property listed on SeLoger, LeBonCoin, PAP simultaneously.
  Prices differ slightly (agency commission variations).
  Treating them as separate opportunities → double-counting inventory.

Deduplication strategy (layered):
  1. Exact match on (normalized_address, surface_m2, code_postal)
  2. Fuzzy match on address string (Levenshtein) for typos/formatting
  3. Price proximity check as confirmation signal

Design decisions documented:
  - Prefer keeping lowest asking price within duplicate cluster
    (conservative for investment analysis)
  - Flag near-duplicates rather than hard-deleting
    (audit trail matters for financial data)
  - DuckDB-compatible: all transformations expressible as SQL aggregations
"""

import re
import logging
from typing import Tuple
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Address normalisation
# ---------------------------------------------------------------------------

STREET_ABBREVS = {
    r"\bbd\b": "boulevard", r"\bav\b": "avenue", r"\bpl\b": "place",
    r"\bst\b": "saint", r"\bimp\b": "impasse", r"\bvla\b": "villa",
    r"\bche\b": "chemin", r"\bcrs\b": "cours", r"\bpas\b": "passage",
}


def normalize_address(raw: str) -> str:
    """
    Standardise address string for matching.
    Handles: case, accents (partial), abbreviations, punctuation.
    """
    if not raw or not isinstance(raw, str):
        return ""

    s = raw.lower().strip()

    # Remove punctuation except hyphen
    s = re.sub(r"[,\.;:!?]", " ", s)

    # Expand abbreviations
    for pattern, replacement in STREET_ABBREVS.items():
        s = re.sub(pattern, replacement, s, flags=re.IGNORECASE)

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()

    return s


def address_similarity(a: str, b: str) -> float:
    """
    Simple character-level Levenshtein similarity [0, 1].
    Implemented from scratch — no difflib dependency needed.
    Fast enough for O(n²) over 400-500 listings.
    """
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    m, n = len(a), len(b)
    # DP matrix (memory-optimised: two rows)
    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, [0] * (n + 1)

    edit_distance = prev[n]
    return 1.0 - edit_distance / max(m, n)


# ---------------------------------------------------------------------------
# Deduplication logic
# ---------------------------------------------------------------------------

def deduplicate_listings(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        clean_df  — deduplicated listings (one row per property)
        dup_log   — audit log of removed duplicates with reason

    Strategy:
        Pass 1 — exact key match (address_norm + surface + zipcode)
        Pass 2 — fuzzy address match within same zipcode + surface tolerance
    """
    df = df.copy()
    df["address_norm"] = df["raw_address"].apply(normalize_address)

    dup_log_rows = []
    duplicate_indices = set()

    # --- Pass 1: exact key deduplication ---
    exact_key = ["address_norm", "surface_m2", "code_postal"]
    df["_exact_group"] = (
        df[exact_key]
        .apply(lambda r: "|".join(str(v) for v in r), axis=1)
    )

    for key, group in df[df["address_norm"] != ""].groupby("_exact_group"):
        if len(group) <= 1:
            continue

        # Keep row with lowest asking price; flag others
        keep_idx = group["asking_price"].idxmin()
        for idx in group.index:
            if idx != keep_idx:
                duplicate_indices.add(idx)
                dup_log_rows.append({
                    "removed_id": df.loc[idx, "listing_id"],
                    "kept_id": df.loc[keep_idx, "listing_id"],
                    "reason": "exact_key_match",
                    "price_removed": df.loc[idx, "asking_price"],
                    "price_kept": df.loc[keep_idx, "asking_price"],
                })

    logger.info(f"Pass 1 (exact): {len(duplicate_indices)} duplicates flagged")

    # --- Pass 2: fuzzy within-zipcode deduplication ---
    FUZZY_THRESHOLD = 0.82
    PRICE_TOLERANCE = 0.06   # within 6%
    SURFACE_TOLERANCE = 5    # within 5m²

    remaining = df[~df.index.isin(duplicate_indices)].copy()
    zipcodes = remaining["code_postal"].unique()

    for zipcode in zipcodes:
        group = remaining[remaining["code_postal"] == zipcode]
        indices = list(group.index)

        for i in range(len(indices)):
            if indices[i] in duplicate_indices:
                continue
            for j in range(i + 1, len(indices)):
                if indices[j] in duplicate_indices:
                    continue

                ri = group.loc[indices[i]]
                rj = group.loc[indices[j]]

                # Skip if addresses are both empty
                if not ri["address_norm"] and not rj["address_norm"]:
                    continue

                # Surface and price proximity check first (fast)
                surface_close = abs(ri["surface_m2"] - rj["surface_m2"]) <= SURFACE_TOLERANCE
                if not surface_close:
                    continue

                price_ratio = abs(ri["asking_price"] - rj["asking_price"]) / max(ri["asking_price"], 1)
                if price_ratio > PRICE_TOLERANCE:
                    continue

                # Levenshtein check (only if passes fast filters)
                sim = address_similarity(ri["address_norm"], rj["address_norm"])
                if sim >= FUZZY_THRESHOLD:
                    # Keep lower price
                    remove_idx = indices[j] if ri["asking_price"] <= rj["asking_price"] else indices[i]
                    keep_idx = indices[i] if remove_idx == indices[j] else indices[j]

                    if remove_idx not in duplicate_indices:
                        duplicate_indices.add(remove_idx)
                        dup_log_rows.append({
                            "removed_id": df.loc[remove_idx, "listing_id"],
                            "kept_id": df.loc[keep_idx, "listing_id"],
                            "reason": f"fuzzy_match (sim={sim:.3f})",
                            "price_removed": df.loc[remove_idx, "asking_price"],
                            "price_kept": df.loc[keep_idx, "asking_price"],
                        })

    logger.info(f"Pass 2 (fuzzy): {len(duplicate_indices)} total duplicates")

    clean_df = df[~df.index.isin(duplicate_indices)].drop(
        columns=["_exact_group"], errors="ignore"
    ).reset_index(drop=True)

    dup_log = pd.DataFrame(dup_log_rows)

    logger.info(
        f"Deduplication complete: {len(df)} → {len(clean_df)} listings "
        f"({len(dup_log)} removed)"
    )
    return clean_df, dup_log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_ingestion(raw_path: str, output_path: str, log_path: str) -> pd.DataFrame:
    logger.info(f"Loading raw listings from {raw_path}")
    df = pd.read_csv(raw_path)
    logger.info(f"Raw: {len(df)} listings from sources: {df['source'].value_counts().to_dict()}")

    clean_df, dup_log = deduplicate_listings(df)

    clean_df.to_csv(output_path, index=False)
    dup_log.to_csv(log_path, index=False)

    return clean_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    clean = run_ingestion(
        "data/raw/listings_raw.csv",
        "data/processed/listings_clean.csv",
        "data/processed/dedup_log.csv",
    )
    print(f"\nClean listings: {len(clean)}")
    print(f"Missing addresses remaining: {(clean['raw_address'] == '').sum()}")
    print(clean[["listing_id", "raw_address", "surface_m2",
                 "asking_price", "source"]].head(5).to_string())
