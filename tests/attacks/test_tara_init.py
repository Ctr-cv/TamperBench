"""Tests for TaRA initialization of LoRA adapters."""

from pathlib import Path

import datasets
import peft
import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from tamperbench.whitebox.attacks.lora_finetune.tara import (
    apply_tara_init,
    collect_covariances,
    tara_factors,
)

PYTHIA = "EleutherAI/pythia-14m"
PYTHIA_TARGET_MODULES = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
RANK = 4
DAMPING = 1e-2


# ---------------------------------------------------------------------------
# tara_factors: pure linear algebra on random matrices
# ---------------------------------------------------------------------------


def _random_spd(d: int, scale: float, gen: torch.Generator) -> torch.Tensor:
    """Random symmetric positive-definite matrix, like an accumulated covariance."""
    x = torch.randn(4 * d, d, generator=gen)
    return scale * (x.T @ x)


def _damped(sigma: torch.Tensor) -> torch.Tensor:
    """Apply the same damping as ``tara_factors``."""
    d = sigma.shape[0]
    return sigma + DAMPING * (sigma.trace() / d) * torch.eye(d)


@pytest.fixture
def random_problem() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """W0 (24 x 16), Sigma_X (16 x 16), Sigma_G (24 x 24) with per-token gradient scale."""
    gen = torch.Generator().manual_seed(0)
    w0 = torch.randn(24, 16, generator=gen)
    return w0, _random_spd(16, 1.0, gen), _random_spd(24, 1e-4, gen)


def test_tara_factors_full_rank_reconstructs_w0(
    random_problem: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """At full rank, s * B @ A recovers W0, so the frozen residual is zero."""
    w0, sigma_x, sigma_g = random_problem
    s = 2.0
    b, a = tara_factors(w0, sigma_x, sigma_g, min(w0.shape), s, DAMPING)

    torch.testing.assert_close(s * b @ a, w0, atol=1e-4, rtol=1e-4)


def test_tara_factors_shapes_balance_and_scale_invariance(
    random_problem: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Factors have LoRA shapes, equal norms, and B @ A ignores the overall scale of Sigma_G."""
    w0, sigma_x, sigma_g = random_problem
    b, a = tara_factors(w0, sigma_x, sigma_g, RANK, 2.0, DAMPING)

    assert b.shape == (24, RANK)
    assert a.shape == (RANK, 16)
    assert b.dtype == a.dtype == torch.float32
    torch.testing.assert_close(b.norm(), a.norm(), atol=1e-4, rtol=1e-4)

    b_scaled, a_scaled = tara_factors(w0, sigma_x, 1000.0 * sigma_g, RANK, 2.0, DAMPING)
    torch.testing.assert_close(b_scaled @ a_scaled, b @ a, atol=1e-4, rtol=1e-4)


def test_tara_factors_beats_truncated_svd_on_weighted_error(
    random_problem: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """TaRA's rank-r part of W0 has lower error than plain truncated SVD, weighted by the damped covariances."""
    w0, sigma_x, sigma_g = random_problem
    s = 2.0
    b, a = tara_factors(w0, sigma_x, sigma_g, RANK, s, DAMPING)

    u, svals, vh = torch.linalg.svd(w0, full_matrices=False)
    w_svd = u[:, :RANK] @ torch.diag(svals[:RANK]) @ vh[:RANK]

    def weighted_error(w_approx: torch.Tensor) -> float:
        return torch.linalg.norm(_damped(sigma_g) @ (w0 - w_approx) @ _damped(sigma_x)).item()

    assert weighted_error(s * b @ a) < weighted_error(w_svd)


def test_tara_factors_accepts_bf16_weights(
    random_problem: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Real model weights are bf16; the factors must still come back in fp32."""
    w0, sigma_x, sigma_g = random_problem
    b, a = tara_factors(w0.bfloat16(), sigma_x, sigma_g, RANK, 2.0, DAMPING)

    assert b.dtype == a.dtype == torch.float32


# ---------------------------------------------------------------------------
# collect_covariances and apply_tara_init: pythia-14m with a real SFTTrainer
# ---------------------------------------------------------------------------


def _build_trainer(output_dir: Path) -> SFTTrainer:
    """Build a pythia-14m LoRA SFTTrainer configured like ``run_lora_attack`` (completion-only loss, checkpointing)."""
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(PYTHIA, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(PYTHIA, torch_dtype=torch.float32)
    model.enable_input_require_grads()
    peft_model = peft.get_peft_model(
        model,
        peft.LoraConfig(r=RANK, lora_alpha=2 * RANK, target_modules=PYTHIA_TARGET_MODULES, task_type="CAUSAL_LM"),
    )

    # Varying lengths so that batches contain padding.
    dataset = datasets.Dataset.from_dict(
        {
            "prompt": [f"Question {i}: what is {i} plus {i}?" for i in range(16)],
            "completion": [" The answer is" + " yes" * (i % 5 + 1) for i in range(16)],
        }
    )
    args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=4,
        max_steps=1,
        completion_only_loss=True,
        gradient_checkpointing=True,
        use_cpu=not torch.cuda.is_available(),
        report_to="none",
        seed=0,
    )
    return SFTTrainer(model=peft_model, processing_class=tokenizer, train_dataset=dataset, args=args)


def _lora_layers(model: torch.nn.Module) -> dict[str, peft.tuners.lora.Linear]:
    """All PEFT LoRA linear layers in ``model``, keyed by module name."""
    return {name: m for name, m in model.named_modules() if isinstance(m, peft.tuners.lora.Linear)}


def _first_batch(trainer: SFTTrainer) -> dict[str, torch.Tensor]:
    """First training batch, on the model's device."""
    return trainer._prepare_inputs(next(iter(trainer.get_train_dataloader())))  # pyright: ignore[reportPrivateUsage]


@pytest.mark.gpu_optional
def test_collect_covariances_ignores_gradient_checkpointing(tmp_path: Path) -> None:
    """Checkpointing recomputes forwards; covariances must not be double-counted when it is on."""
    trainer = _build_trainer(tmp_path)
    model = trainer.model
    assert model is not None
    assert model.is_gradient_checkpointing, "SFTTrainer is expected to enable checkpointing before train()"

    batches = [_first_batch(trainer)]
    layers = {name: m.base_layer for name, m in _lora_layers(model).items()}

    model.gradient_checkpointing_disable()
    off = collect_covariances(model, batches, trainer.compute_loss, layers)
    model.gradient_checkpointing_enable()
    on = collect_covariances(model, batches, trainer.compute_loss, layers)

    assert len(layers) == 24  # 6 blocks x 4 target modules
    for name in layers:
        sigma_x, sigma_g = off[name]
        assert sigma_g.abs().sum() > 0, f"{name}: Sigma_G is zero, the gradient hook did not fire"
        torch.testing.assert_close(sigma_x, sigma_x.T)
        torch.testing.assert_close(on[name][0], sigma_x, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(on[name][1], sigma_g, rtol=1e-5, atol=1e-10)

    assert all(p.grad is None for n, p in model.named_parameters() if "lora_" in n)


@pytest.mark.gpu_optional
def test_apply_tara_init_preserves_outputs(tmp_path: Path) -> None:
    """Right after TaRA init, the model computes the same function; only the split into W_res + s*BA changed."""
    trainer = _build_trainer(tmp_path)
    model = trainer.model
    assert model is not None
    batch = _first_batch(trainer)
    mask = batch["attention_mask"].bool()
    b_before = {name: m.lora_B["default"].weight.detach().clone() for name, m in _lora_layers(model).items()}

    model.eval()
    with torch.no_grad():
        logits_before = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits

    rng_before = torch.get_rng_state()
    apply_tara_init(trainer, DAMPING, 16)
    assert torch.equal(torch.get_rng_state(), rng_before), "apply_tara_init must restore the RNG state"

    model.eval()
    with torch.no_grad():
        logits_after = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits

    # Elementwise tolerances are too strict here: |s*BA| can exceed |W0|, so W_res + s*BA cancels and fp32
    # rounding through 6 layers gives ~1e-4 relative error. Compare the whole logit tensor instead.
    before, after = logits_before[mask], logits_after[mask]
    relative_error = ((after - before).norm() / before.norm()).item()
    assert relative_error < 1e-3, f"logits changed by {relative_error:.2e} (relative) after TaRA init"
    for name, m in _lora_layers(model).items():
        assert not torch.equal(m.lora_B["default"].weight, b_before[name]), f"{name}: lora_B was not overwritten"


def _mean_gradient_alignment(trainer: SFTTrainer, batch: dict[str, torch.Tensor]) -> float:
    """Mean over LoRA layers of the cosine between LoRA's first-step change to W and the full gradient of W.

    For W_eff = W_res + s * B @ A, one gradient step on (A, B) changes W_eff in the direction
    s * (B @ grad_A + grad_B @ A). Full fine-tuning would move it along grad_W.
    """
    model = trainer.model
    assert model is not None
    layers = _lora_layers(model)
    for m in layers.values():
        m.base_layer.weight.requires_grad_(True)

    model.eval()
    model.zero_grad(set_to_none=True)
    trainer.compute_loss(model, batch).backward()

    cosines = []
    for m in layers.values():
        a = m.lora_A["default"].weight
        b = m.lora_B["default"].weight
        assert a.grad is not None and b.grad is not None and m.base_layer.weight.grad is not None
        delta = m.scaling["default"] * (b @ a.grad + b.grad @ a)
        cosines.append(F.cosine_similarity(delta.flatten(), m.base_layer.weight.grad.flatten(), dim=0))
    return torch.stack(cosines).mean().item()


@pytest.mark.gpu_optional
def test_tara_first_step_aligns_better_with_full_gradient(tmp_path: Path) -> None:
    """TaRA's first LoRA update points closer to the full fine-tuning gradient than standard LoRA's."""
    lora_trainer = _build_trainer(tmp_path / "lora")
    tara_trainer = _build_trainer(tmp_path / "tara")
    batch = _first_batch(lora_trainer)  # same batch for both

    apply_tara_init(tara_trainer, DAMPING, 16)

    lora_alignment = _mean_gradient_alignment(lora_trainer, batch)
    tara_alignment = _mean_gradient_alignment(tara_trainer, batch)
    print(f"mean cosine with full gradient: LoRA {lora_alignment:.3f}, TaRA {tara_alignment:.3f}")

    assert tara_alignment > lora_alignment
