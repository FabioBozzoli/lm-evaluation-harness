from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


pytest.importorskip("transformers")

from lm_eval.models.hf_rebased import (  # noqa: E402
    CausalLMHeadShim,
    block_ridge_correction_context,
    calibration_loader,
    fit_steer_text,
    next_token_loss,
    steering_loss_report,
    theseus_rebase,
)
from lm_eval.rebase import get_method, list_methods  # noqa: E402
from lm_eval.rebase import theseus as theseus_mod  # noqa: E402
from lm_eval.rebase.steer_text import steer_text_correction_context  # noqa: E402


# --------------------------------------------------------------------------
# Ported from merge-and-rebase/tests/test_theseus_rebase.py (imports only)
# --------------------------------------------------------------------------


class _TinyVisual(nn.Module):
    def __init__(self, in_dim: int = 6, hid_dim: int = 8, out_dim: int = 5) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.ln = nn.LayerNorm(hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.ln(x)
        return self.fc2(x)


class _TinyModel(nn.Module):
    def __init__(self, in_dim: int = 6, hid_dim: int = 8, out_dim: int = 5) -> None:
        super().__init__()
        self.visual = _TinyVisual(in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.visual(x)


def _make_loader(n_samples: int = 16, in_dim: int = 6, batch_size: int = 4) -> DataLoader:
    x = torch.randn(n_samples, in_dim)
    y = torch.zeros(n_samples, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def test_methods_registered() -> None:
    assert {"theseus", "steer", "steer_text"} <= set(list_methods())
    assert get_method("theseus").name == "theseus"


@pytest.mark.parametrize("extra", [{}, {"whiten_power": 0.25}])
def test_theseus_transport_smoke(extra) -> None:
    source_model = _TinyModel(in_dim=6, hid_dim=8, out_dim=5)
    target_model = _TinyModel(in_dim=6, hid_dim=7, out_dim=5)

    source_base = {k: v.detach().clone() for k, v in source_model.state_dict().items()}
    target_base = {k: v.detach().clone() for k, v in target_model.state_dict().items()}

    delta = {
        key: torch.randn_like(tensor)
        for key, tensor in source_base.items()
        if key.startswith("visual.") and tensor.is_floating_point()
    }

    loader = _make_loader(in_dim=6)
    transported = get_method("theseus").transport(
        source_base=source_base,
        target_base=target_base,
        delta=delta,
        source_model=source_model,
        target_model=target_model,
        source_dataloader=loader,
        target_dataloader=loader,
        device="cpu",
        seq_align="mean",
        num_batches=1,
        strict=True,
        **extra,
    )

    assert transported
    assert set(transported.keys()) == set(delta.keys())
    for key, tensor in transported.items():
        assert tensor.shape == target_base[key].shape
        assert tensor.dtype == target_base[key].dtype


def test_theseus_data_free_transport_smoke_without_dataloaders() -> None:
    source_model = _TinyModel(in_dim=6, hid_dim=8, out_dim=5)
    target_model = _TinyModel(in_dim=6, hid_dim=7, out_dim=5)

    source_base = {k: v.detach().clone() for k, v in source_model.state_dict().items()}
    target_base = {k: v.detach().clone() for k, v in target_model.state_dict().items()}

    delta = {
        key: torch.randn_like(tensor)
        for key, tensor in source_base.items()
        if key.startswith("visual.") and tensor.is_floating_point()
    }

    transported = get_method("theseus").transport(
        source_base=source_base,
        target_base=target_base,
        delta=delta,
        source_model=source_model,
        target_model=target_model,
        device="cpu",
        covariance_mode="data_free",
        whiten_power=0.25,
        strict=True,
    )

    assert transported
    assert set(transported.keys()) == set(delta.keys())
    for key, tensor in transported.items():
        assert tensor.shape == target_base[key].shape


def test_partial_whitening_changes_alignment_map() -> None:
    store = theseus_mod.ActivationStore(store_a_gram=True, store_b_gram=True)
    source_rows = torch.tensor([[3.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    target_rows = torch.tensor([[1.0, 2.0], [2.0, 0.0], [0.0, 1.0]])
    store.update(source_rows, target_rows)

    raw_map = theseus_mod._compute_alignment_map(store, center=False, whiten_power=0.0, whiten_eps=1e-6)
    whitened_map = theseus_mod._compute_alignment_map(store, center=False, whiten_power=0.5, whiten_eps=1e-6)

    assert raw_map is not None
    assert whitened_map is not None
    assert raw_map.shape == whitened_map.shape == (2, 2)
    assert not torch.allclose(raw_map, whitened_map)


def test_data_free_covariance_map_uses_weight_proxies() -> None:
    source = torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float32)
    target = source @ rotation

    t_in = theseus_mod._compute_alignment_map_from_matrix_proxies(
        source, target, side="input", whiten_power=0.0, whiten_eps=1e-6
    )

    assert t_in.shape == (2, 2)
    aligned_cov = t_in.T @ (source.T @ source) @ t_in
    assert torch.allclose(aligned_cov, target.T @ target, atol=1e-5, rtol=1e-5)


def test_random_dataset_subsampling_uses_randperm_seed() -> None:
    x = torch.arange(20, dtype=torch.float32).unsqueeze(1)
    y = torch.zeros(20, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)

    iterator = theseus_mod._iter_random_dataset_batches(loader, loader, n_batches=3, seed=123, batch_size=4)
    assert iterator is not None

    seen: list[int] = []
    for source_batch, _ in iterator:
        seen.extend(int(v) for v in source_batch[0].squeeze(1).tolist())

    g = torch.Generator(device="cpu")
    g.manual_seed(123)
    assert seen == torch.randperm(20, generator=g)[:12].tolist()


# --------------------------------------------------------------------------
# Causal LM integration
# --------------------------------------------------------------------------

VOCAB = 64
PAD = 0


def _tiny_lm(hidden: int, seed: int):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=VOCAB,
        hidden_size=hidden,
        intermediate_size=2 * hidden,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    return Qwen2ForCausalLM(config).eval()


def _rows(n: int = 12, seed: int = 0) -> list[list[int]]:
    g = torch.Generator().manual_seed(seed)
    lengths = torch.randint(4, 9, (n,), generator=g).tolist()
    # ids start at 1 so PAD never collides with a real token
    return [torch.randint(1, VOCAB, (length,), generator=g).tolist() for length in lengths]


def _loader() -> DataLoader:
    return calibration_loader(_rows(), pad_token_id=PAD, batch_size=4)


def test_calibration_loader_labels_are_next_tokens() -> None:
    batch = next(iter(calibration_loader([[5, 6, 7], [8, 9]], pad_token_id=PAD, batch_size=2)))
    assert batch["labels"].tolist() == [6, 7, 9]


def test_theseus_rebases_backbone_across_widths() -> None:
    source_pre, source_ft = _tiny_lm(32, seed=0), _tiny_lm(32, seed=1)
    target = _tiny_lm(48, seed=2)
    before = {k: v.clone() for k, v in target.state_dict().items()}
    test_loader = _loader()
    # theseus writes A's delta straight into B's weights: "before"/"after" is the same
    # loss metric steering_loss_report uses, just two states of one model over time
    # instead of a live correction vs. an oracle (see hf_rebased.__init__'s theseus branch).
    loss_before = next_token_loss(target, test_loader, torch.device("cpu"))

    n = theseus_rebase(
        source_pre, source_ft, target, _loader(), _loader(),
        device="cpu", n_batches=2, verbose=False, show_progress=False,
    )

    loss_after = next_token_loss(target, test_loader, torch.device("cpu"))
    assert loss_before != loss_after
    assert torch.isfinite(torch.tensor(loss_after))

    after = target.state_dict()
    changed = [k for k in before if not torch.equal(before[k], after[k])]
    assert n > 0 and changed
    assert all(k.startswith("model.layers.") or k.startswith("model.norm.") for k in changed)
    assert all(torch.isfinite(after[k]).all() for k in changed)


def test_theseus_skips_shape_mismatched_embeddings_instead_of_crashing() -> None:
    # A finetune that extended the vocabulary (e.g. new special/reasoning tokens) leaves
    # embed_tokens.weight with more rows than the pretrained source's -- theseus never
    # transports embeddings anyway, so this must be skipped, not raise a shape error.
    source_pre, source_ft = _tiny_lm(32, seed=0), _tiny_lm(32, seed=1)
    target = _tiny_lm(48, seed=2)
    with torch.no_grad():
        extra_row = source_ft.model.embed_tokens.weight[:1].clone()
        source_ft.model.embed_tokens.weight = nn.Parameter(
            torch.cat([source_ft.model.embed_tokens.weight, extra_row], dim=0)
        )
    source_ft.config.vocab_size = VOCAB + 1

    n = theseus_rebase(
        source_pre, source_ft, target, _loader(), _loader(),
        device="cpu", n_batches=2, verbose=False, show_progress=False,
    )
    assert n > 0


def _fit_steer(tmp_path, source_pre, source_ft, target, **overrides):
    params = {
        "feature_regime": "standard",
        "stage_2_strategy": "global_ridge",
        "total_support_examples": 32,
        "feature_cache_dir": str(tmp_path),
        "task": "lm",
        "source_tag": "src",
        "target_tag": "tgt",
        "seed": 0,
        "verbose": False,
    }
    params.update(overrides)
    return fit_steer_text(
        source_pre, source_ft, target,
        train=(_loader(), _loader()), test=(_loader(), _loader()), device="cpu", **params,
    )


@pytest.mark.parametrize(
    "stage_2",
    [
        pytest.param({"stage_2_strategy": "global_ridge"}, id="global_ridge"),
        pytest.param({"stage_2_strategy": "global_mlp", "mlp_hidden_dim": 16, "mlp_epochs": 5}, id="global_mlp"),
    ],
)
def test_steer_text_on_lm_head_shifts_logits_and_restores(tmp_path, stage_2) -> None:
    source_pre, source_ft = _tiny_lm(32, seed=0), _tiny_lm(32, seed=1)
    target = _tiny_lm(48, seed=2)
    prepared = _fit_steer(tmp_path, source_pre, source_ft, target, **stage_2)
    assert prepared["stage_2_strategy"] == stage_2["stage_2_strategy"]
    assert set(prepared["diagnostics"]) == {"stage0_test_acc", "stage1_test_acc", "stage2_test_acc"}

    x = torch.tensor(_rows(n=2, seed=5)[0]).unsqueeze(0)
    with torch.no_grad():
        base = target(input_ids=x).logits
        with steer_text_correction_context(SimpleNamespace(model=CausalLMHeadShim(target)), prepared):
            steered = target(input_ids=x).logits
        restored = target(input_ids=x).logits

    assert not torch.allclose(base, steered)
    assert torch.equal(base, restored)


def test_steer_text_block_ridge_linear_corrects_every_position(tmp_path) -> None:
    source_pre, source_ft = _tiny_lm(32, seed=0), _tiny_lm(32, seed=1)
    target = _tiny_lm(48, seed=2)
    prepared = _fit_steer(tmp_path, source_pre, source_ft, target, stage_2_strategy="block_ridge", feature_regime="linear")
    assert prepared["num_source_blocks"] == 3  # 2 layers + the output block

    x = torch.tensor(_rows(n=2, seed=5)[0]).unsqueeze(0)
    with torch.no_grad():
        base = target(input_ids=x).logits
        with block_ridge_correction_context(target, prepared):
            steered = target(input_ids=x).logits
            last_only = target(input_ids=x, logits_to_keep=1).logits
        restored = target(input_ids=x).logits

    assert not torch.allclose(base, steered)
    # A generation-style pass that only projects the last position gets the same correction there.
    assert torch.allclose(last_only[:, -1], steered[:, -1], atol=1e-5)
    assert torch.equal(base, restored)


@pytest.mark.parametrize(
    "stage_2",
    [
        pytest.param({"stage_2_strategy": "global_ridge"}, id="global_ridge"),
        pytest.param({"stage_2_strategy": "block_ridge", "feature_regime": "linear"}, id="block_ridge"),
    ],
)
def test_steering_loss_report_scores_all_three_stages(tmp_path, stage_2) -> None:
    # Task-appropriate diagnostic (loss/perplexity) instead of steer_text's own
    # accuracy-over-the-full-vocabulary one; needs prepared["logit_map"]/["p_b"],
    # the one addition to steer_text.py's own return dict (see its diff against
    # merge-and-rebase: import lines plus these two keys, nothing else).
    source_pre, source_ft = _tiny_lm(32, seed=0), _tiny_lm(32, seed=1)
    target = _tiny_lm(48, seed=2)
    test_loaders = (_loader(), _loader())
    prepared = fit_steer_text(
        source_pre, source_ft, target,
        train=(_loader(), _loader()), test=test_loaders, device="cpu",
        feature_cache_dir=str(tmp_path), task="lm", source_tag="src", target_tag="tgt",
        total_support_examples=32, seed=0, verbose=False, **stage_2,
    )
    assert prepared["logit_map"].shape == (VOCAB, 32)  # [V, source hidden dim]
    assert prepared["p_b"].shape == (48, VOCAB)  # pinv(target head): [target hidden dim, V]

    report = steering_loss_report(target, prepared, source_pre, source_ft, test_loaders)
    assert set(report) == {
        "stage0_loss", "stage1_oracle_loss", "stage2_loss",
        "stage0_ppl", "stage1_oracle_ppl", "stage2_ppl",
    }
    for key, value in report.items():
        assert torch.isfinite(torch.tensor(value)) and value > 0, (key, value)
    for key in ("stage0", "stage1_oracle", "stage2"):
        assert report[f"{key}_ppl"] == pytest.approx(torch.tensor(report[f"{key}_loss"]).exp().item())


@pytest.mark.parametrize(
    "overrides",
    [{"stage_2_strategy": "block_ridge", "feature_regime": "standard"}, {"few_shot": 2, "total_support_examples": None}],
)
def test_steer_text_rejects_what_a_vocab_head_cannot_support(tmp_path, overrides) -> None:
    source_pre, source_ft, target = _tiny_lm(32, seed=0), _tiny_lm(32, seed=1), _tiny_lm(48, seed=2)
    with pytest.raises(ValueError):
        _fit_steer(tmp_path, source_pre, source_ft, target, **overrides)
