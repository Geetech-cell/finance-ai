import pandas as pd
from pathlib import Path


REQUIRED_COLUMNS = ["date", "description", "amount"]


def validate_columns(df: pd.DataFrame) -> None:
    """Ensure required columns exist."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw transaction data:
    - validate columns
    - parse dates
    - clean text
    - convert amount to numeric
    - remove duplicates & nulls
    """
    df = df.copy()

    # Normalize column names
    df.columns = df.columns.str.lower().str.strip()

    validate_columns(df)

    # Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Clean description text
    df["description"] = (
        df["description"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # Convert amount
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # Drop invalid rows
    df = df.dropna(subset=["date", "description", "amount"])

    # Remove duplicates
    df = df.drop_duplicates()

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    return df


def save_cleaned_data(df: pd.DataFrame, filename: str = "cleaned_transactions.csv") -> Path:
    """Save cleaned data to data/processed."""
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename
    df.to_csv(output_path, index=False)

    return output_path
