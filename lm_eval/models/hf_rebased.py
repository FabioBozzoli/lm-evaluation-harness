"""HFLM whose target model is rebased with a merge-and-rebase method before evaluation.

``method=theseus`` transports the source task vector (finetuned - pretrained) into
the target backbone's weights. ``method=steer_text`` leaves the weights alone and
adds its feature-space correction at the input of ``lm_head``, which stands in for
the classification head: every next-token position is one example and the next
token is its label.

The method code under ``lm_eval/rebase`` is a verbatim copy of merge-and-rebase;
everything causal-LM specific lives here.

    lm_eval --model rebased \
        --model_args pretrained=EleutherAI/pythia-160m,source_pretrained=EleutherAI/pythia-70m,source_finetuned=<ckpt>,method=theseus \
        --tasks lambada_openai
"""

from __future__ import annotations

import hashlib
import itertools
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils_hf import get_dtype
from lm_eval.rebase import get_method
from lm_eval.rebase.adapters import (
    _NEVER_TRANSPORT,
    TextEncoderShim,
    alias_inputs_loader,
)
from lm_eval.rebase.steer_text import steer_text_correction_context


_THESEUS_PARAMS = {
    "covariance_mode",
    "whiten_power",
    "whiten_eps",
    "center_acts",
    "n_batches",
    "num_batches",
    "seq_align",
    "seed",
    "verbose",
    "show_progress",
}
_STEER_TEXT_PARAMS = {
    "feature_regime",
    "stage_2_strategy",
    "block_group_strategy",
    "few_shot",
    "total_support_examples",
    "stage1_lambda",
    "ridge_lambda",
    "block_ridge_mode",
    "rho",
    "mlp_hidden_dim",
    "mlp_epochs",
    "force_recompute_features",
    "seed",
    "verbose",
}


def _next_token_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """``[B, T-1]`` positions whose next token is real: the rows steer_text fits on."""
    mask = attention_mask.bool()
    return mask[:, :-1] & mask[:, 1:]


def _block_output(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, (tuple, list)) else output


class CausalLMHeadShim(nn.Module):
    """Present a causal LM to steer_text as a classifier whose head is ``lm_head``.

    ``score`` is the LM's own output-embedding Linear under a name steer_text's
    ``head_linear`` already recognises, so the correction pre-hook lands on the very
    module the real model calls during evaluation. ``forward`` returns one feature row
    per next-token position, flattened, which steer_text's collection loops treat as
    one example each. The backbone must not be stored as ``model``:
    ``steer_text_correction_context`` reads ``llm.model``.

    ``taps.layers.N`` exist for block_ridge. steer_text's ``_TextBlockCapture`` hooks
    the first module named ``layers.N`` and mean-pools what it captures over the
    sequence; the taps are registered before ``backbone`` so they are the modules it
    finds, and they receive block ``N``'s output already flattened to the same
    next-token rows as the feature, which ``_masked_mean`` passes through untouched.
    """

    def __init__(self, lm: nn.Module) -> None:
        super().__init__()
        self.taps = nn.Module()
        self.taps.layers = nn.ModuleList(nn.Identity() for _ in getattr(lm.base_model, "layers", ()))
        self.backbone = lm.base_model
        self.score = lm.get_output_embeddings()

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> SimpleNamespace:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        valid = _next_token_mask(attention_mask)

        def feed_tap(tap: nn.Module):
            def hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
                tap(_block_output(output)[:, :-1][valid])

            return hook

        handles = []
        if any(tap._forward_hooks for tap in self.taps.layers):  # only while a capture is listening
            handles = [
                block.register_forward_hook(feed_tap(tap))
                for block, tap in zip(self.backbone.layers, self.taps.layers, strict=True)
            ]
        try:
            hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask)[0]
        finally:
            for handle in handles:
                handle.remove()
        return SimpleNamespace(logits=self.score(hidden[:, :-1][valid]))


@contextmanager
def block_ridge_correction_context(lm: nn.Module, prepared: dict[str, Any], *, alpha: float = 1.0):
    """Apply a block_ridge ``prepared["correction_fn"]`` to a live causal LM, per position.

    Stands in for ``steer_text_correction_context``, whose block capture pools each
    sequence into one row and whose mask hook only fires when the shim is called.
    Here the real blocks are hooked, and the head input ``[B, k, D]`` is matched with
    the last ``k`` positions of every block output: ``k`` is the full length for a
    scoring pass, 1 for a generation step with KV cache, and the kept tail whenever
    the model only projects the last positions (``logits_to_keep``). Rows are
    flattened exactly as they were at fit time, so ``correction_fn`` (steer_text's,
    unchanged) sees the same shapes.
    """
    correction_fn = prepared["correction_fn"]
    blocks = list(lm.base_model.layers)
    activations: dict[int, torch.Tensor] = {}

    def capture(block_id: int):
        def hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
            activations[block_id] = _block_output(output)

        return hook

    def head_pre_hook(_module: nn.Module, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        feature = inputs[0]
        if len(activations) != len(blocks):
            raise RuntimeError("block_ridge: not every block output was captured before lm_head ran.")
        width, dim = feature.shape[1], feature.shape[-1]
        flat = feature.reshape(-1, dim)
        rows = {b: act[:, -width:].reshape(-1, act.shape[-1]) for b, act in activations.items()}
        rows[len(blocks)] = flat
        correction = correction_fn({"global": flat, "blocks": rows})
        return (feature + float(alpha) * correction.reshape(feature.shape),) + tuple(inputs[1:])

    handles = [block.register_forward_hook(capture(i)) for i, block in enumerate(blocks)]
    handles.append(lm.get_output_embeddings().register_forward_pre_hook(head_pre_hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


class _TokenRows(Dataset):
    def __init__(self, rows: list[list[int]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> list[int]:
        return self.rows[idx]


def calibration_loader(
    rows: list[list[int]], *, pad_token_id: int, batch_size: int
) -> DataLoader:
    """Right-padded batches with ``attention_mask`` and flattened next-token ``labels``."""

    def collate(batch: list[list[int]]) -> dict[str, torch.Tensor]:
        width = max(len(row) for row in batch)
        input_ids = torch.full((len(batch), width), int(pad_token_id), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for i, row in enumerate(batch):
            input_ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
            attention_mask[i, : len(row)] = 1
        labels = input_ids[:, 1:][_next_token_mask(attention_mask)]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return DataLoader(
        _TokenRows(rows), batch_size=int(batch_size), shuffle=False, collate_fn=collate
    )


def _backbone_state(lm: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().float().cpu() for k, v in lm.base_model.state_dict().items()}


def theseus_rebase(
    source_pretrained: nn.Module,
    source_finetuned: nn.Module,
    target: nn.Module,
    source_loader: DataLoader,
    target_loader: DataLoader,
    *,
    alpha: float = 1.0,
    device: str = "cuda",
    source_pad_token_id: int = 0,
    target_pad_token_id: int = 0,
    **params: Any,
) -> int:
    """Add ``alpha`` x the theseus-transported source task vector to ``target``'s backbone, in place.

    Only the backbone goes through theseus: hooking the whole LM would also hook
    ``lm_head``, whose ``[tokens, vocab]`` outputs make a vocab x vocab covariance.
    The output head and the input embeddings (never hooked, see ``adapters``) are
    therefore not transported. Returns the number of transported keys.
    """
    pre = _backbone_state(source_pretrained)
    finetuned = _backbone_state(source_finetuned)
    mismatched = [k for k, v in pre.items() if k in finetuned and v.shape != finetuned[k].shape]
    if mismatched:
        # A finetune that extended the vocabulary changes embed_tokens.weight's row
        # count; theseus never transports embeddings anyway (see the docstring), so
        # skipping them here is consistent, not a new limitation.
        print(f"[theseus_rebase] skipping {len(mismatched)} shape-mismatched key(s), e.g. {mismatched[:3]}")
    delta = {
        k: finetuned[k] - v
        for k, v in pre.items()
        if k in finetuned
        and v.is_floating_point()
        and v.shape == finetuned[k].shape
        and not any(fragment in k for fragment in _NEVER_TRANSPORT)
    }
    target_base = _backbone_state(target)

    params.setdefault("seq_align", "interpolate")  # token sequences are not a square patch grid
    method = get_method("theseus")
    prepared = method.prepare(
        source_model=TextEncoderShim(source_pretrained.base_model, source_pad_token_id),
        target_model=TextEncoderShim(target.base_model, target_pad_token_id),
        source_dataloader=alias_inputs_loader(source_loader),
        target_dataloader=alias_inputs_loader(target_loader),
        target_base=target_base,
        delta=delta,
        device=device,
        patch_qkv=False,
        **params,
    )
    transported = method.transport(
        source_base=pre,
        target_base=target_base,
        delta=delta,
        prepared=prepared,
        verbose=params.get("verbose", True),
        show_progress=params.get("show_progress", True),
    )
    target.base_model.load_state_dict(
        {k: target_base[k] + float(alpha) * v.float() for k, v in transported.items()},
        strict=False,
    )
    return len(transported)


def fit_steer_text(
    source_pretrained: nn.Module,
    source_finetuned: nn.Module,
    target: nn.Module,
    *,
    train: tuple[DataLoader, DataLoader],
    test: tuple[DataLoader, DataLoader],
    device: str = "cuda",
    **params: Any,
) -> dict[str, Any]:
    """steer_text's ``prepare`` with ``lm_head`` standing in for the classification head.

    ``train``/``test`` are ``(source_loader, target_loader)`` pairs over identical
    token rows. ``few_shot`` is rejected up front: it would mean one class per
    vocabulary token. ``block_ridge`` needs ``feature_regime="linear"`` (steer_text
    enforces it) and must be applied with :func:`block_ridge_correction_context`.
    """
    params.setdefault("total_support_examples", 4096)
    if params.get("few_shot") is not None or params.get("total_support_examples") is None:
        raise ValueError(
            "steer_text on lm_head requires total_support_examples; few_shot would mean "
            "one class per vocabulary token."
        )
    labels = torch.cat([batch["labels"] for batch in train[0]])
    return get_method("steer_text").prepare(
        llm_source=SimpleNamespace(model=CausalLMHeadShim(source_finetuned)),
        llm_source_pretrained=SimpleNamespace(model=CausalLMHeadShim(source_pretrained)),
        llm_target=SimpleNamespace(model=CausalLMHeadShim(target)),
        source_loaders=SimpleNamespace(
            train=train[0], test=test[0], local_labels={"train": labels.tolist()}
        ),
        target_loaders=SimpleNamespace(train=train[1], test=test[1]),
        mask_class=None,
        device=device,
        **params,
    )


def _calibration_texts(dataset: str, config: str | None, split: str, fields: str, n: int) -> list[str]:
    """First ``n`` non-empty rows, ``fields`` joined by blank lines.

    ``fields`` is ``+``-separated because ``model_args`` splits on commas, e.g.
    ``query+response`` (hkust-nlp/dart-math-*), ``problem+solution``
    (ise-uiuc/Magicoder-OSS-Instruct-75K), ``instruction+response``
    (ise-uiuc/Magicoder-Evol-Instruct-110K).
    """
    from datasets import load_dataset

    names = str(fields).split("+")
    rows = load_dataset(dataset, config, split=split, streaming=True)
    texts = ("\n\n".join(str(row[k]) for k in names if row.get(k)) for row in rows)
    return list(itertools.islice((t for t in texts if t.strip()), int(n)))


def _pad_id(tokenizer: Any) -> int:
    return tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id


def _path_tag(name: str) -> str:
    return str(name).replace("/", "__")


@register_model("rebased")
class RebasedHFLM(HFLM):
    def __init__(
        self,
        pretrained: str,
        source_pretrained: str,
        source_finetuned: str,
        method: str = "theseus",
        alpha: float = 1.0,
        calib_dataset: str = "wikitext",
        calib_config: str | None = "wikitext-2-raw-v1",
        calib_split: str = "train",
        calib_fields: str = "text",
        calib_samples: int = 64,
        # ponytail: steer_text's test diagnostics build [test tokens, vocab] float64
        # logits, so this stays tiny; raise it only with RAM to spare.
        calib_test_samples: int = 4,
        calib_max_length: int = 256,
        calib_batch_size: int = 8,
        # None = load both sources in the target's own resolved dtype (self.model.dtype,
        # e.g. bfloat16): two float32 copies of a 3B model cost ~12 GB more GPU memory
        # than bf16 ones, a real cause of CUDA OOM on this pair. theseus's weight delta
        # is computed on values upcast to float32 right when read off the state dict
        # (see _backbone_state), so it ends up at bf16 *precision* either way when the
        # sources are loaded in bf16 -- pass calib_dtype: float32 explicitly if that
        # precision loss on a small delta is a concern and GPU memory allows it.
        calib_dtype: str | None = None,
        feature_cache_dir: str | None = None,
        **kwargs,
    ) -> None:
        """HFLM evaluated after rebasing the target (``pretrained``) with ``method``.

        The source task vector is ``source_finetuned - source_pretrained``; source and
        target must share parameter names (same family, any width/depth). Method
        hyperparameters are passed through ``model_args`` unchanged (see
        ``_THESEUS_PARAMS`` / ``_STEER_TEXT_PARAMS``).
        """
        allowed = {"theseus": _THESEUS_PARAMS, "steer_text": _STEER_TEXT_PARAMS}.get(method)
        if allowed is None:
            raise ValueError(f"method must be 'theseus' or 'steer_text', got {method!r}.")
        method_params = {
            k: kwargs.pop(k) for k in list(kwargs) if k in _THESEUS_PARAMS | _STEER_TEXT_PARAMS
        }
        wrong = sorted(set(method_params) - allowed)
        if wrong:
            raise ValueError(f"{wrong} are not {method} parameters.")

        super().__init__(pretrained=pretrained, **kwargs)

        import transformers
        from packaging.version import parse as vparse
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = str(self.device)
        source_tokenizer = AutoTokenizer.from_pretrained(source_pretrained)
        # dtype= only exists from transformers>=4.56; older versions silently ignore it
        # (it reaches the model's __init__ unconsumed and raises TypeError) and need
        # torch_dtype= instead -- same check HFLM.build itself uses.
        dtype_arg = "dtype" if vparse(transformers.__version__) >= vparse("4.56.0") else "torch_dtype"
        resolved_dtype = get_dtype(calib_dtype) if calib_dtype is not None else self.model.dtype
        sources = [
            AutoModelForCausalLM.from_pretrained(path, **{dtype_arg: resolved_dtype})
            .to(device=self.device)
            .eval()
            for path in (source_pretrained, source_finetuned)
        ]

        # One read of the split: the first calib_samples texts calibrate, the next
        # calib_test_samples feed steer_text's diagnostics (dart-math and magicoder
        # only ship a train split).
        texts = _calibration_texts(
            calib_dataset, calib_config, calib_split, calib_fields, int(calib_samples) + int(calib_test_samples)
        )

        def token_rows(texts: list[str]) -> tuple[list[list[int]], list[list[int]]]:
            encode = dict(truncation=True, max_length=int(calib_max_length))
            pairs = [
                (source_tokenizer(t, **encode)["input_ids"], self.tokenizer(t, **encode)["input_ids"])
                for t in texts
            ]
            pairs = [(s, t) for s, t in pairs if len(s) >= 2 and len(t) >= 2]
            return [s for s, _ in pairs], [t for _, t in pairs]

        def loaders(rows: tuple[list[list[int]], list[list[int]]]) -> tuple[DataLoader, DataLoader]:
            return (
                calibration_loader(rows[0], pad_token_id=_pad_id(source_tokenizer), batch_size=calib_batch_size),
                calibration_loader(rows[1], pad_token_id=_pad_id(self.tokenizer), batch_size=calib_batch_size),
            )

        train_rows = token_rows(texts[: int(calib_samples)])
        if method == "theseus":
            theseus_rebase(
                *sources,
                self.model,
                *loaders(train_rows),
                alpha=float(alpha),
                device=device,
                source_pad_token_id=_pad_id(source_tokenizer),
                target_pad_token_id=_pad_id(self.tokenizer),
                **method_params,
            )
        else:
            test_rows = token_rows(texts[int(calib_samples) :])
            if train_rows[0] != train_rows[1] or test_rows[0] != test_rows[1]:
                raise ValueError(
                    "steer_text pairs source and target token by token, so both must tokenize identically "
                    "(use models from the same family)."
                )
            calib_key = f"{calib_dataset}|{calib_config}|{calib_split}|{calib_fields}|{calib_samples}|{calib_test_samples}|{calib_max_length}"
            prepared = fit_steer_text(
                *sources,
                self.model,
                train=loaders(train_rows),
                test=loaders(test_rows),
                device=device,
                task=f"lm_{hashlib.sha1(calib_key.encode()).hexdigest()[:12]}",
                source_tag=_path_tag(f"{source_pretrained}+{source_finetuned}"),
                target_tag=_path_tag(self.model.name_or_path),
                feature_cache_dir=str(feature_cache_dir or Path.home() / ".cache" / "lm_eval" / "steer_text"),
                **method_params,
            )
            # Kept open for the lifetime of this object: the pre-hook sits on the real
            # lm_head, so every _model_call/_model_generate is corrected.
            if prepared["stage_2_strategy"] == "block_ridge":
                correction = block_ridge_correction_context(self.model, prepared, alpha=float(alpha))
            else:
                correction = steer_text_correction_context(
                    SimpleNamespace(model=CausalLMHeadShim(self.model)), prepared, alpha=float(alpha)
                )
            self._steer_hooks = ExitStack()
            self._steer_hooks.enter_context(correction)
        del sources
