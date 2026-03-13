import numpy as np
import pandas as pd


def load_customer_data(csv_path: str) -> pd.DataFrame:
    """Load raw customer data from CSV."""
    return pd.read_csv(csv_path)


def save_dataframe(df: pd.DataFrame, csv_path: str, index: bool = False) -> None:
    """Save a dataframe to CSV with a consistent project-wide interface."""
    df.to_csv(csv_path, index=index)


def drop_customer_id(df: pd.DataFrame, id_col: str = "CUST_ID") -> pd.DataFrame:
    """Drop identifier column if present."""
    if id_col in df.columns:
        return df.drop(columns=[id_col]).copy()
    return df.copy()


def median_impute(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Median-impute selected columns in a copy of the input frame."""
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].fillna(out[col].median())
    return out


def log1p_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Apply elementwise log(1 + x) to all numeric columns."""
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = np.log1p(out[numeric_cols])
    return out


def attach_cluster_labels(
    features_df: pd.DataFrame,
    cluster_csv_path: str,
    cluster_col: str = "Cluster",
) -> pd.DataFrame:
    """Attach cluster labels from a CSV file to a feature dataframe."""
    cluster_df = pd.read_csv(cluster_csv_path)
    if cluster_col not in cluster_df.columns:
        raise KeyError(f"Missing '{cluster_col}' column in {cluster_csv_path}")
    if len(cluster_df) != len(features_df):
        raise ValueError("Cluster label count does not match feature row count")

    out = features_df.copy()
    out[cluster_col] = cluster_df[cluster_col].values
    return out
