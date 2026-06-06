"""
app/extraction/address_extractor.py
------------------------------------
Pipeline Step 2: Extract or recover missing addresses from listing descriptions.

Problem context (Ali's actual use case):
  ~8-10% of scraped listings have empty or malformed address fields.
  Without a valid address → can't match DVF → can't value the property.
  This module recovers those addresses using rule-based extraction + LLM fallback.

Design decisions:
  - Rule-based extraction first (fast, free, deterministic)
  - LLM fallback only for ambiguous cases (cost control)
  - Confidence score on each extraction (transparency for downstream)
  - Never blocks the pipeline: returns "unknown" with confidence=0 if stuck
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extraction result
# ---------------------------------------------------------------------------

@dataclass
class AddressExtractionResult:
    address: str
    zipcode: Optional[int]
    confidence: float          # 0.0 - 1.0
    method: str                # "existing" | "regex" | "llm" | "zipcode_only" | "unknown"

    def is_usable(self) -> bool:
        """Can we match this to DVF?"""
        return self.zipcode is not None and self.confidence >= 0.3


# ---------------------------------------------------------------------------
# Regex patterns for French addresses
# ---------------------------------------------------------------------------

# Matches: "12 rue de la Roquette", "12, rue des Martyrs"
STREET_PATTERN = re.compile(
    r"(\d+[a-zA-Z]?)[,\s]+("
    r"(?:rue|boulevard|avenue|place|impasse|villa|chemin|voie|passage|allée|cité)"
    r"\s+[A-Za-zÀ-ÿ\s\-']+)",
    re.IGNORECASE
)

# Matches Paris zipcodes: 75001-75020, 750XX
ZIPCODE_PATTERN = re.compile(r"\b(75\d{3})\b")

# Paris arrondissement mentions
ARROND_PATTERN = re.compile(
    r"(?:Paris\s+)?(\d{1,2})(?:ème|eme|er|ère|e)\s*(?:arrondissement)?",
    re.IGNORECASE
)

ARROND_TO_ZIP = {i: 75000 + i for i in range(1, 21)}


def arrond_to_zipcode(arrond_num: int) -> Optional[int]:
    return ARROND_TO_ZIP.get(arrond_num)


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class AddressExtractor:
    """
    Layered address extraction:
      1. Use existing address if valid
      2. Regex extraction from description
      3. Zipcode-only fallback (still useful for DVF zone matching)
      4. LLM fallback (if client provided)
      5. Unknown
    """

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def extract(self, raw_address: str, description: str, code_postal: int) -> AddressExtractionResult:

        # --- Step 1: existing address valid? ---
        if raw_address and isinstance(raw_address, str) and len(raw_address.strip()) > 8:
            zip_match = ZIPCODE_PATTERN.search(raw_address)
            zipcode = int(zip_match.group(1)) if zip_match else code_postal
            return AddressExtractionResult(
                address=raw_address.strip(),
                zipcode=zipcode,
                confidence=0.95,
                method="existing"
            )

        # --- Step 2: regex from description ---
        street_match = STREET_PATTERN.search(description)
        zip_from_desc = ZIPCODE_PATTERN.search(description)
        arrond_match = ARROND_PATTERN.search(description)

        if street_match:
            number = street_match.group(1)
            street = street_match.group(2).strip()
            zipcode = (
                int(zip_from_desc.group(1)) if zip_from_desc
                else (arrond_to_zipcode(int(arrond_match.group(1))) if arrond_match else code_postal)
            )
            address = f"{number} {street}, Paris {zipcode}"
            return AddressExtractionResult(
                address=address,
                zipcode=zipcode,
                confidence=0.75,
                method="regex"
            )

        # --- Step 3: zipcode-only (still useful for DVF zone lookup) ---
        zipcode = None
        if zip_from_desc:
            zipcode = int(zip_from_desc.group(1))
        elif arrond_match:
            zipcode = arrond_to_zipcode(int(arrond_match.group(1)))
        elif code_postal and str(code_postal).startswith("75"):
            zipcode = int(code_postal)

        if zipcode:
            return AddressExtractionResult(
                address=f"Paris {zipcode}",
                zipcode=zipcode,
                confidence=0.35,
                method="zipcode_only"
            )

        # --- Step 4: LLM fallback ---
        if self.llm_client:
            try:
                result = self._extract_with_llm(description, code_postal)
                if result:
                    return result
            except Exception as e:
                logger.warning(f"LLM extraction failed: {e}")

        # --- Step 5: unknown ---
        return AddressExtractionResult(
            address="", zipcode=None, confidence=0.0, method="unknown"
        )

    def _extract_with_llm(self, description: str, code_postal: int) -> Optional[AddressExtractionResult]:
        prompt = (
            f"Extract the street address from this French real estate listing description. "
            f"Return ONLY a JSON object: {{\"address\": \"...\", \"zipcode\": 75XXX}} "
            f"or {{\"address\": null}} if no address found.\n\n"
            f"Description: {description[:500]}"
        )
        response = self.llm_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )
        import json, re as re2
        raw = response.content[0].text.strip()
        raw = re2.sub(r"```json|```", "", raw).strip()
        data = json.loads(raw)
        if data.get("address"):
            return AddressExtractionResult(
                address=data["address"],
                zipcode=data.get("zipcode", code_postal),
                confidence=0.65,
                method="llm"
            )
        return None


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def enrich_addresses(df: pd.DataFrame, llm_client=None) -> pd.DataFrame:
    """Add extracted address fields to listings dataframe."""
    extractor = AddressExtractor(llm_client=llm_client)
    df = df.copy()

    results = df.apply(
        lambda r: extractor.extract(
            r.get("raw_address", ""),
            r.get("description", ""),
            r.get("code_postal", 0)
        ),
        axis=1
    )

    df["address_clean"] = [r.address for r in results]
    df["address_zipcode"] = [r.zipcode for r in results]
    df["address_confidence"] = [r.confidence for r in results]
    df["address_method"] = [r.method for r in results]

    method_counts = df["address_method"].value_counts().to_dict()
    usable = sum(1 for r in results if r.is_usable())
    logger.info(
        f"Address extraction: {usable}/{len(df)} usable "
        f"| methods: {method_counts}"
    )
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = pd.read_csv("data/processed/listings_clean.csv")
    enriched = enrich_addresses(df)
    enriched.to_csv("data/processed/listings_addressed.csv", index=False)

    print(f"\nAddress extraction results:")
    print(enriched["address_method"].value_counts())
    print(f"\nUsable for DVF matching: "
          f"{(enriched['address_confidence'] >= 0.3).sum()}/{len(enriched)}")
    print(enriched[["listing_id", "raw_address", "address_clean",
                     "address_method", "address_confidence"]].head(8).to_string())
