"""Loader for the Close Elections panel data."""

from causaldata import close_elections_lmb

from .utils import dataframe_to_dataloader


def get_close_elections_dataloader(batch_size: int = 256):
    """Return dataloader for the close US House elections study."""
    df = close_elections_lmb.load_pandas().data
    loader = dataframe_to_dataloader(
        df,
        treatment_col="democrat",
        outcome_col="score",
        batch_size=batch_size,
        drop=["state"],
    )
    return loader, (None, None)
