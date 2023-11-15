__all__ = ["FlamlTreeExplainer", "xai_to_text"]

"""Explainable AI (XAI) enables humans to understand the decisions or predictions made by the AI.

This package contains mostly interfaces for well-established AI models in XAI domain such as **SHAP** and **LIME**.
The final goal of all modules (interfaces) is to even further extract, simplify, and transform these
"explanatory" values to be understandable by non-technical users (an actual pleb and/or pepega).
Hence, these modules might intentionally oversimplify things that
actually might be theoretically unsound! (See issue #56)

Submodules:

    * :mod:`vizard.xai.shap <vizard.xai.shap>`: contains all necessary interfaces around SHAP_ library. See notebook ``notebooks/shap.ipynb``.


.. _SHAP: https://github.com/slundberg/shap

"""


# ours
from .shap import FlamlTreeExplainer
from .core import xai_to_text

# helpers
import logging


# set logger
logger = logging.getLogger(__name__)
