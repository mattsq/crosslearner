"""Loader for the credit card delinquency dataset."""

from causaldata import credit_cards

from .utils import dataframe_to_dataloader


def get_credit_cards_dataloader(batch_size: int = 256):
    """Return dataloader for the credit card late-payment data."""
    df = credit_cards.load_pandas().data
    loader = dataframe_to_dataloader(
        df,
        treatment_col="LateApril",
        outcome_col="LateSept",
        batch_size=batch_size,
    )
    return loader, (None, None)
