"""TaRA (PEFT) fine-tuning attack helpers, based on LoraFinetune interface."""

import math
from collections.abc import Callable

import peft
import torch
from trl.trainer.sft_trainer import SFTTrainer


def tara_factors(
    W0: torch.Tensor,
    sumx: torch.Tensor,
    sumg: torch.Tensor,
    r: int,
    s: float,
    c: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the TaRA factors B and A for one weight matrix.

    Args:
        W0: Original pre-trained weight (d_out x d_in).
        sumx: Input-activation covariance (d_in x d_in).
        sumg: Output-gradient covariance (d_out x d_out).
        r: LoRA rank.
        s: PEFT LoRA scaling (lora_alpha / r).
        c: Damping factor, relative to the mean eigenvalue of each covariance.

    Returns:
        B (d_out x r) and A (r x d_in) in fp32.
    """
    d_out, d_in = W0.shape

    # 1. Dampen each covariance by a factor
    sx, sg = sumx.float().clone(), sumg.float().clone()
    sx.diagonal().add_(c * (sx.trace() / d_in))
    sg.diagonal().add_(c * (sg.trace() / d_out))

    # 2. Compute the matrix M for SVD, take top-r
    M = sg @ W0.float() @ sx
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    U_r = U[:, :r]  # (d_out, r)
    S_r_sqrt = S[:r].sqrt()  # (r,)
    Vh_r = Vh[:r, :]  # (r, d_in)

    # 3. Compute B and A
    B = torch.linalg.solve(sg, U_r * S_r_sqrt.unsqueeze(0))
    A = torch.linalg.solve(sx, (S_r_sqrt[:, None] * Vh_r).T).T

    # 4. Frobenius norm over the two matrices before returning
    # First scale by inv sqrt s
    inv_sqrt = 1.0 / math.sqrt(s)
    B.mul_(inv_sqrt)
    A.mul_(inv_sqrt)

    norm_a = torch.linalg.norm(A, ord="fro")
    norm_b = torch.linalg.norm(B, ord="fro")
    if norm_a > 0 and norm_b > 0:
        lam = torch.sqrt(norm_a / norm_b)
        B.mul_(lam)
        A.div_(lam)

    return B, A


def collect_covariances(
    model: torch.nn.Module,
    batches: list[dict[str, torch.Tensor]],
    compute_loss: Callable[[torch.nn.Module, dict[str, torch.Tensor]], torch.Tensor],
    layers: dict[str, torch.nn.Linear],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Accumulate the input-activation and output-gradient covariances of each layer.

    Args:
        model: PEFT model to run.
        batches: Calibration batches, already on the model's device.
        compute_loss: Loss function taken from the trainer.
        layers: Base linear layers to collect covariances for, keyed by module name.

    Returns:
        Map from layer name to (Sigma_X, Sigma_G) in fp32.

    Raises:
        ValueError: If a layer's output does not require gradients.
    """
    result: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, layer in layers.items():
        device = layer.weight.device
        sx = torch.zeros((layer.in_features, layer.in_features), dtype=torch.float32, device=device)
        sg = torch.zeros((layer.out_features, layer.out_features), dtype=torch.float32, device=device)
        result[name] = (sx, sg)

    state: dict[str, torch.Tensor] = {}  # Holds the current batch's attention mask under "mask"
    handles: list[torch.utils.hooks.RemovableHandle] = []  # Per layer handles

    # Register a forward hook here
    def forward_hook(
        layer_name: str,
    ) -> Callable[[torch.nn.Module, tuple[torch.Tensor, ...], torch.Tensor], None]:
        sx, sg = result[layer_name]

        def g_hook(grad_output: torch.Tensor) -> None:
            mask = state["mask"].to(grad_output.device)
            g = grad_output[mask].float()
            sg.add_(g.T @ g)

        def fwd_hook(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            mask = state["mask"].to(inputs[0].device)
            x = inputs[0].detach()[mask].float()  # New tensor, apply mask, convert to float
            sx.add_(x.T @ x)

            if output.requires_grad:
                output.register_hook(g_hook)
            else:
                raise ValueError(f"{layer_name} does not have gradients to register g_hook")

        return fwd_hook

    for name, layer in layers.items():
        handles.append(layer.register_forward_hook(forward_hook(name)))

    was_training = model.training
    # Note eval mode disables gradient checkpointing recompute
    model.eval()

    # Here accumulate the X^T X and G^T G matrices across the batches.
    try:
        # The microbatch here is to reduce GPU memory consumption. Run each batch as a smaller piece to use only a fraction of activ mem.
        micro_batch_size = 2 
        for batch in batches:
            for i in range(0, batch["input_ids"].shape[0], micro_batch_size):
                mb = {k: v[i:i + micro_batch_size] for k, v in batch.items()}
                n_tokens = (mb["labels"][:, 1:] != -100).sum()  # tokens the loss averages over (labels are shifted)
                if n_tokens == 0:
                    continue  # mean over zero tokens is NaN
                state["mask"] = mb["attention_mask"].bool()
                (compute_loss(model, mb) * n_tokens).backward()
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
        for param_name, param in model.named_parameters():
            if "lora_" in param_name:
                param.grad = None

    return result


def apply_tara_init(trainer: SFTTrainer, damping: float, calibration_size: int):
    """Overwrite the trainer's LoRA factors and base weights with the TaRA initialization.

    Args:
        trainer: SFTTrainer wrapping a PEFT LoRA model, built but not yet trained.
        damping: Damping factor passed to ``tara_factors``.
        calibration_size: Number of training examples used for calibration.
    """
    # Note: to ensure the same torch random state is used for better comparison between LoRA and TaRA,
    # mark the state here and restore it at the end.
    state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    try:
        # 1. cal batches: small subset of mini-batches drawn from training data for LoRA matrices
        batches: list[dict[str, torch.Tensor]] = []
        model = trainer.model
        assert model is not None, "trainer.model must be set before TaRA init"
        seen = 0
        dataLoader = trainer.get_train_dataloader()
        for raw_batch in dataLoader:
            if seen >= calibration_size:
                break
            batch = trainer._prepare_inputs(raw_batch)  # pyright: ignore[reportPrivateUsage]
            batch_size = batch["input_ids"].shape[0]

            # Caution: in case final batch overflows leftover space, trim it
            remaining = calibration_size - seen
            if batch_size > remaining:
                batch = {k: v[:remaining] for k, v in batch.items()}
                batch_size = remaining
            batches.append(batch)
            seen += batch_size

        # 2. Get the LoRA layers
        lora_layers: list[tuple[str, peft.tuners.lora.Linear]] = []
        for name, module in model.named_modules():
            if isinstance(module, peft.tuners.lora.Linear):
                lora_layers.append((name, module))

        # 3. Process all the layers in with some group size
        # Concern: entire model + all LoRA layers may not fit on a smaller GPU at once.
        # Multiples of 7: each decoder block has 7 linear layers, so take 4 decoder blocks per group for processing
        group_size = 56  # In colab, setting 56 took ~15GB peak memory on a L4 which still leaves sufficient room.
        for i in range(0, len(lora_layers), group_size):
            group = lora_layers[i : i + group_size]
            covs = collect_covariances(model, batches, trainer.compute_loss, {name: m.base_layer for name, m in group})

            with torch.no_grad():
                # Do initialization
                for name, m in group:
                    r = m.r["default"]
                    s = m.scaling["default"]
                    W0 = m.base_layer.weight
                    sx, sg = covs[name]

                    B, A = tara_factors(W0, sx, sg, r, s, damping)

                    # Copy over to the model
                    m.lora_A["default"].weight.copy_(A.to(m.lora_A["default"].weight.dtype))
                    m.lora_B["default"].weight.copy_(B.to(m.lora_B["default"].weight.dtype))
                    m.base_layer.weight.copy_((W0.float() - s * B @ A).to(W0.dtype))

            # evict this group from cache entirely
            del covs
            torch.cuda.empty_cache()

    finally:
        torch.set_rng_state(state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
