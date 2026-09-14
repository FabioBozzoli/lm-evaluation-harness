"""Rebase methods copied from merge-and-rebase (theseus, steer, steer_text).

Only import lines differ from the originals; see ``lm_eval/models/hf_rebased.py``
for how they are driven on causal LMs.
"""

from . import adapters, steer, steer_text, theseus  # noqa: F401  -- registration + theseus hook patch
from .registry import get_method, list_methods  # noqa: F401
