from .toy import get_toy_dataloader
from .complex import get_complex_dataloader


def get_ihdp_dataloader(*args, **kwargs):
    """Load the IHDP dataset on demand."""
    from .ihdp import get_ihdp_dataloader as _loader

    return _loader(*args, **kwargs)


def get_jobs_dataloader(*args, **kwargs):
    """Load the Jobs dataset on demand."""
    from .jobs import get_jobs_dataloader as _loader

    return _loader(*args, **kwargs)


__all__ = [
    "get_toy_dataloader",
    "get_complex_dataloader",
    "get_ihdp_dataloader",
    "get_jobs_dataloader",
]
