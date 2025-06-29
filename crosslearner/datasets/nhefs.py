"""Loader for the NHEFS smoking cessation dataset."""

from causaldata import nhefs_complete

from .utils import dataframe_to_dataloader


def get_nhefs_dataloader(batch_size: int = 256):
    """Return dataloader for the full NHEFS data."""
    df = nhefs_complete.load_pandas().data
    for col in df.select_dtypes(["category"]).columns:
        df[col] = df[col].cat.codes
    df = df.fillna(0)
    loader = dataframe_to_dataloader(
        df,
        treatment_col="qsmk",
        outcome_col="wt82_71",
        batch_size=batch_size,
        drop=["seqn"],
    )
    return loader, (None, None)
