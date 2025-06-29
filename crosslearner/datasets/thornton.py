"""Loader for the Thornton HIV cash-incentive study."""

from causaldata import thornton_hiv

from .utils import dataframe_to_dataloader


def get_thornton_hiv_dataloader(batch_size: int = 256):
    """Return dataloader for the Thornton HIV experiment."""
    df = thornton_hiv.load_pandas().data.fillna(0)
    loader = dataframe_to_dataloader(
        df,
        treatment_col="any",
        outcome_col="got",
        batch_size=batch_size,
    )
    return loader, (None, None)
