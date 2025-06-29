"""Loader for the Social Insure network experiment."""

from causaldata import social_insure

from .utils import dataframe_to_dataloader


def get_social_insure_dataloader(batch_size: int = 256):
    """Return dataloader for the Social Insure dataset."""
    df = social_insure.load_pandas().data
    df["any"] = (df["intensive"] == 1) | (df["default"] == 1)
    loader = dataframe_to_dataloader(
        df,
        treatment_col="any",
        outcome_col="takeup_survey",
        batch_size=batch_size,
        drop=["address", "village", "intensive", "default"],
    )
    return loader, (None, None)
