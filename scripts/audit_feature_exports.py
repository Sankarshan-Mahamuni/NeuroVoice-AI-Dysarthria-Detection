"""Compare saved feature metadata with the validated dataset catalog."""
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
CATALOG_PATH = ROOT_DIR / "data" / "metadata" / "dataset_catalog.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
AUDIT_PATH = ROOT_DIR / "data" / "metadata" / "feature_extraction_audit.csv"


def audit_feature_exports():
    if not CATALOG_PATH.exists():
        raise FileNotFoundError("Run scripts/validate_dataset.py before auditing feature exports.")

    catalog = pd.read_csv(CATALOG_PATH)
    processed_metadata = []
    for split in ("train", "validation", "test"):
        metadata_path = PROCESSED_DIR / f"{split}_metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Processed metadata is missing: {metadata_path}")
        frame = pd.read_csv(metadata_path, usecols=["filename", "speaker"])
        frame["split"] = split
        processed_metadata.append(frame)

    processed = pd.concat(processed_metadata, ignore_index=True)
    catalog = catalog.rename(columns={"hf_split": "split"})
    audit = catalog.merge(
        processed[["filename", "speaker"]].drop_duplicates(),
        on=["filename", "speaker"],
        how="left",
        indicator=True
    )
    missing = audit.loc[audit["_merge"] == "left_only", [
        "split", "speaker", "filename", "class_label", "dysarthria_severity"
    ]].copy()
    missing["status"] = "missing_from_saved_feature_exports"
    missing["reason"] = (
        "The original extraction run did not retain per-record errors; "
        "rerun scripts/extract_features.py to diagnose current failures."
    )

    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    missing.to_csv(AUDIT_PATH, index=False)
    print(f"Catalog records: {len(catalog)}")
    print(f"Saved feature records: {len(processed)}")
    print(f"Missing feature records: {len(missing)}")
    print("Missing by split:")
    print(missing.groupby("split").size().to_string())
    print(f"Audit written to: {AUDIT_PATH}")


if __name__ == "__main__":
    audit_feature_exports()
