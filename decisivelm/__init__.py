"""DecisiveLM: constrained, calibrated Choice / Score / Noul answers in one pass (docs/guide/).

    from decisivelm import DecisiveLM
    dlm = DecisiveLM.load("bundles/decisivelm-e4b-general")
"""
import pathlib

CONFIGS = pathlib.Path(__file__).resolve().parent.parent / "configs"


def __getattr__(name):   # DecisiveLM on first use, so importing the package for CONFIGS does not load torch
    if name == "DecisiveLM":
        from decisivelm.infer import DecisiveLM
        return DecisiveLM
    raise AttributeError(f"module 'decisivelm' has no attribute {name!r}")
