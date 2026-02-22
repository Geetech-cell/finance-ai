import pandas as pd


def prepare_forecast_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts transaction dataframe into Prophet format:
    ds = date
    y  = total spending per day
    """
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    df = df.dropna(subset=["date", "amount"])

    # Convert expenses into positive values for forecasting spending
    df["spend"] = df["amount"].apply(lambda x: abs(x) if x < 0 else 0)

    # Aggregate daily spending
    daily = df.groupby(df["date"].dt.date)["spend"].sum().reset_index()
    daily.columns = ["ds", "y"]

    daily["ds"] = pd.to_datetime(daily["ds"])

    return daily


def prepare_forecast_data_by_category(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """
    Prepare Prophet dataset for a single category.
    """
    if "predicted_category" not in df.columns:
        raise ValueError("DataFrame must contain 'predicted_category' column.")

    filtered = df[df["predicted_category"] == category]
    return prepare_forecast_data(filtered)
