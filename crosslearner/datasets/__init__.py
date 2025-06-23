"""Dataset loading helpers."""

from .toy import get_toy_dataloader
from .complex import get_complex_dataloader
from .synthetic import get_confounding_dataloader
from .aircraft import get_aircraft_dataloader
from .tricks import get_tricky_dataloader
from .masked import MaskedFeatureDataset


def get_ihdp_dataloader(*args, **kwargs):
    """Load the IHDP dataset on demand."""
    from .ihdp import get_ihdp_dataloader as _loader

    return _loader(*args, **kwargs)


def get_jobs_dataloader(*args, **kwargs):
    """Load the Jobs dataset on demand."""
    from .jobs import get_jobs_dataloader as _loader

    return _loader(*args, **kwargs)


def get_acic2016_dataloader(*args, **kwargs):
    """Load the ACIC 2016 dataset on demand."""
    from .acic2016 import get_acic2016_dataloader as _loader

    return _loader(*args, **kwargs)


def get_acic2018_dataloader(*args, **kwargs):
    """Load the ACIC 2018 dataset on demand."""
    from .acic2018 import get_acic2018_dataloader as _loader

    return _loader(*args, **kwargs)


def get_twins_dataloader(*args, **kwargs):
    """Load the Twins dataset on demand."""
    from .twins import get_twins_dataloader as _loader

    return _loader(*args, **kwargs)


def get_lalonde_dataloader(*args, **kwargs):
    """Load the LaLonde dataset on demand."""
    from .lalonde import get_lalonde_dataloader as _loader

    return _loader(*args, **kwargs)


__all__ = [
    "get_toy_dataloader",
    "get_complex_dataloader",
    "get_ihdp_dataloader",
    "get_jobs_dataloader",
    "get_acic2016_dataloader",
    "get_acic2018_dataloader",
    "get_twins_dataloader",
    "get_lalonde_dataloader",
    "get_confounding_dataloader",
    "get_aircraft_dataloader",
    "get_tricky_dataloader",
    "MaskedFeatureDataset",
]
