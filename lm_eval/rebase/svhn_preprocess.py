"""
``task_artifact_name`` from merge-and-rebase's ``data/svhn_preprocess.py``.

``steer.py`` imports it to name its feature-cache directories. The torchvision
transforms of the original module are not needed for that and are left out, so
lm-eval does not need torchvision. For every task other than ``"SVHN"`` it
returns the name unchanged.
"""

from __future__ import annotations

import os

SVHN_TASK = "SVHN"

# Settings that keep the artifact directory name short. See task_artifact_name: these are
# compared against, not reassigned, so flipping the switches below still yields a distinct
# directory instead of colliding with a run that used the canonical values.
CANONICAL_TARGET_SIZE = 96
CANONICAL_AUGMENTATION = "autoaugment"

# Artifacts produced under the custom preprocessing live beside, never inside, the stock ones.
SVHN_ARTIFACT_BASENAME = "SVHN_custom"

# Master switch. Env override: MR_SVHN_PREPROCESS=1
SVHN_CUSTOM_PREPROCESS = False
# Size the digit is resized to before being padded back to the backbone input size.
# Env override: MR_SVHN_TARGET_SIZE
SVHN_TARGET_SIZE = CANONICAL_TARGET_SIZE
# Augmentation applied to the train split. Env override: MR_SVHN_AUGMENTATION
SVHN_AUGMENTATION = CANONICAL_AUGMENTATION

# Keys of the original AUGMENTATIONS dict (its values are torchvision pipelines).
AUGMENTATIONS = (
    "affine_jitter_strong",
    "affine_jitter_light",
    "autoaugment",
    "randaugment_2_5",
    "randaugment_3_5",
    "randaugment_3_7",
    "photometric",
    "none",
)

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    raise ValueError(f"{name} must be one of: {sorted(_TRUTHY | _FALSY)}. Got {raw!r}.")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer. Got {raw!r}.") from exc


def svhn_preprocess_enabled() -> bool:
    return _env_bool("MR_SVHN_PREPROCESS", SVHN_CUSTOM_PREPROCESS)


def svhn_target_size() -> int:
    size = _env_int("MR_SVHN_TARGET_SIZE", SVHN_TARGET_SIZE)
    if size <= 0:
        raise ValueError(f"SVHN target size must be positive. Got {size}.")
    return size


def svhn_augmentation_name() -> str:
    name = str(os.environ.get("MR_SVHN_AUGMENTATION", SVHN_AUGMENTATION)).strip()
    if name not in AUGMENTATIONS:
        raise ValueError(f"SVHN augmentation must be one of: {sorted(AUGMENTATIONS)}. Got {name!r}.")
    return name


def task_artifact_name(task: str) -> str:
    """
    Directory (or cache-key) name for artifacts of ``task`` that depend on preprocessing.

    Checkpoints, curvature caches and feature caches computed under the custom SVHN
    pipeline are not interchangeable with the stock ones, so they get their own name:
    ``SVHN_custom`` on the canonical settings, and ``SVHN_custom_t<size>_<augmentation>``
    as soon as either differs, so an augmentation sweep cannot overwrite itself.

    Every other task, and SVHN with the override off, is returned unchanged.
    """
    name = str(task)
    if name != SVHN_TASK or not svhn_preprocess_enabled():
        return name
    target_size = svhn_target_size()
    augmentation = svhn_augmentation_name()
    if target_size == CANONICAL_TARGET_SIZE and augmentation == CANONICAL_AUGMENTATION:
        return SVHN_ARTIFACT_BASENAME
    return f"{SVHN_ARTIFACT_BASENAME}_t{target_size}_{augmentation}"
