"""Loader for the CPS Mixtape dataset."""

from causaldata import cps_mixtape

from .utils import dataframe_to_dataloader


def get_cps_mixtape_dataloader(batch_size: int = 256):
    """Return dataloader for the observational CPS sample."""
    df = cps_mixtape.load_pandas().data
    loader = dataframe_to_dataloader(
        df,
        treatment_col="treat",
        outcome_col="re78",
        batch_size=batch_size,
        drop=["data_id"],
    )
    return loader, (None, None)
