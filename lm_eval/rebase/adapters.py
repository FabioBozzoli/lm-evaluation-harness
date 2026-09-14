"""
Thin adapters that let the *existing* rebase methods run on HuggingFace text
models without touching a single line of the method files.

Subset of merge-and-rebase's ``rebase/text/adapters.py``: only what ``theseus``
and ``steer_text`` need (bico, gradfix, probing and reporting helpers are not
ported).

- ``theseus`` is *mostly* agnostic. ``theseus._visual_module``,
  ``_visual_state_dict`` and ``_visual_delta_keys`` all fall back to "the whole
  model / the whole state dict" when there is no ``visual.`` prefix, and
  ``_has_fused_mha`` is False on HF attention (already split q/k/v), so the
  OpenCLIP qkv split/merge is a no-op. What it *does* assume is an image-shaped
  forward: ``theseus._encode_image`` calls ``model(images)`` positionally, and
  ``theseus._extract_model_inputs`` only looks for image-ish batch keys.

:class:`TextEncoderShim` and :func:`alias_inputs_loader` close exactly that gap.

Importing this module also applies :func:`_patch_theseus_embedding_hooks`,
a small runtime patch (not a file edit) needed because ``theseus``'s
activation hooks assume every parameterized submodule captures a
``[batch, token, hidden]`` tensor. An ``nn.Embedding`` breaks that assumption;
see the function's docstring.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Head roots emitted by finetune/train_text.py (kept in sync with its
# _detect_head_roots); ordered by preference.
_HEAD_ROOTS: tuple[str, ...] = ("score", "classifier", "classification_head")

# Parameter-name fragments that never belong in a transported task vector.
_NEVER_TRANSPORT: tuple[str, ...] = ("position_ids", "num_batches_tracked", "rotary_emb.inv_freq")


class TextEncoderShim(nn.Module):
    """Present an HF text model to ``theseus`` as if it were a CLIP model.

    Two attributes carry the whole trick:

    ``visual``
        ``theseus._visual_module(model)`` returns ``model.visual`` when present.
        Pointing it at the *unwrapped* HF model means the activation hooks in
        ``theseus._ActivationHook`` register on the HF submodules and are keyed
        by their plain names (``layers.0.self_attn.q_proj``) -- exactly the
        names used by ``state_dict()``, by ``target_base`` and by the delta.
        Wrapping the model as a normal submodule instead would prefix every hook
        key and silently match nothing in ``_precompute_transforms``.

    ``encode_image``
        ``theseus._encode_image`` prefers it over a positional ``model(x)`` call,
        so this is where the ``attention_mask`` that a positional call would
        drop gets rebuilt from the pad id.
    """

    def __init__(self, model: nn.Module, pad_token_id: int) -> None:
        super().__init__()
        self.visual = model
        self.pad_token_id = int(pad_token_id)

    def encode_image(self, input_ids: torch.Tensor) -> torch.Tensor:
        attention_mask = (input_ids != self.pad_token_id).long()
        out = self.visual(input_ids=input_ids, attention_mask=attention_mask)
        logits = getattr(out, "logits", None)
        if logits is not None:
            return logits
        return out[0] if isinstance(out, (tuple, list)) else out

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.visual(*args, **kwargs)


def alias_inputs_loader(loader: DataLoader) -> DataLoader:
    """Re-emit ``loader``'s batches with an extra ``"inputs"`` alias for ``input_ids``.

    ``theseus._extract_model_inputs`` accepts the key ``"inputs"`` already, so
    this single alias satisfies ``theseus``'s ``_encode_image`` feed without any
    change to that file.

    The returned loader keeps ``.dataset``, ``.batch_size`` and ``.collate_fn``,
    which is what ``theseus._iter_random_dataset_batches`` needs to build paired
    source/target batches over the *same* dataset indices.
    """
    inner = loader.collate_fn
    if inner is None:
        raise ValueError("alias_inputs_loader expects a loader with an explicit collate_fn.")

    def _collate(batch: list[Any]) -> dict[str, Any]:
        out = dict(inner(batch))
        out["inputs"] = out["input_ids"]
        return out

    return DataLoader(
        loader.dataset,
        batch_size=int(loader.batch_size or 1),
        shuffle=False,
        num_workers=int(getattr(loader, "num_workers", 0)),
        pin_memory=bool(getattr(loader, "pin_memory", False)),
        drop_last=bool(getattr(loader, "drop_last", False)),
        collate_fn=_collate,
    )


def _head_root_and_linears(model: nn.Module) -> tuple[str, list[tuple[str, nn.Linear]]]:
    for root in _HEAD_ROOTS:
        module = getattr(model, root, None)
        if module is None:
            continue
        if isinstance(module, nn.Linear):
            return root, [(root, module)]
        linears = [(f"{root}.{n}", m) for n, m in module.named_modules() if isinstance(m, nn.Linear)]
        if linears:
            return root, linears
    raise ValueError(
        "Could not locate a classification head on this model. Expected one of "
        f"{list(_HEAD_ROOTS)} holding an nn.Linear (an AutoModelForSequenceClassification). "
        "Methods needing a head (steer_text) require model_kind='sequence_classification'."
    )


def head_linear(model: nn.Module) -> tuple[str, nn.Linear]:
    """Return ``(qualified_name, module)`` of the final classification ``nn.Linear``.

    Its *input* is, by construction, the pooled sentence feature the classifier
    consumes -- whatever pooling rule the architecture uses internally (T5 takes
    the decoder's eos position, Qwen/Llama the last non-pad token). Capturing it
    with a forward pre-hook is therefore architecture-independent, which is why
    ``steer_text`` never has to reimplement per-model pooling.

    Note this is the input to the *final* Linear only. Some heads (T5's)
    insert randomly-initialized layers before it, so "input to the final
    Linear" is not the same as "the model's own pretrained representation" --
    ``steer_text`` doesn't care (it fits and evaluates in that same space
    consistently either way), but a method that wants the model's actual
    pretrained features does.
    """
    _, linears = _head_root_and_linears(model)
    return linears[-1]


def _patch_theseus_embedding_hooks() -> None:
    """
    Make ``theseus._ActivationHook`` skip ``nn.Embedding`` submodules when it
    registers activation hooks.

    ``_ActivationHook`` hooks *every* submodule that owns parameters directly
    (``if list(module.parameters(recurse=False))``), with no assumption
    weaker than "captures a ``[batch, token, hidden]`` tensor" -- see
    ``theseus._standardize_tokens``/``_align_features``. No CLIP submodule
    that logic was written against looks like anything else.

    Embeddings break that assumption: T5's relative-position bias and every
    token embedding capture an input whose trailing dimension is the *token
    count of the current batch*. Two calibration batches with different token
    counts (ordinary dynamic padding) then produce differently-shaped captures
    for the very same hook key, and ``ActivationStore.update``'s cross-batch
    covariance accumulation (``self.at_b += a.T @ b``) crashes with a shape
    mismatch.

    ``theseus.py`` is not to be edited, so this patches the hook class's
    ``_register_hooks`` in place instead of the file on disk. It is applied
    once, at import time of this module.
    """
    from . import theseus as _theseus

    if getattr(_theseus._ActivationHook, "_skips_embeddings", False):
        return  # idempotent: harmless if this module is imported more than once

    def _register_hooks_no_embeddings(self: Any) -> None:
        self.handles.append(self.model.register_forward_hook(self._make_hook("")))
        for name, module in self.model.named_modules():
            if name == "" or isinstance(module, nn.Embedding):
                continue
            if list(module.parameters(recurse=False)):
                self.handles.append(module.register_forward_hook(self._make_hook(name)))

    _theseus._ActivationHook._register_hooks = _register_hooks_no_embeddings
    _theseus._ActivationHook._skips_embeddings = True


_patch_theseus_embedding_hooks()
