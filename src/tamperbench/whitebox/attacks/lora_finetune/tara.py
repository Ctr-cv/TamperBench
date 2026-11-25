"""TaRA (PEFT) fine-tuning attack helpers, based on LoraFinetune interface."""

from collections.abc import Callable
import math
import torch

def tara_factors(W0, sumx, sumg, r, s, c):
    """Helper function to compute the BA initialization. Returns B (d_out x r) and A (r x d_in)

    Args:
        W0: original pre-trained model weights (d_out x d_in)
        sumx: input-activation covariance matrix (d_in x d_in)
        sumg: output-gradient covariance matrix (d_out x d_out)
        r: rank r  
        s: scaling s
        c: damping c
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

def collect_covariances(model: torch.nn.Module, batches: list[dict[str, torch.Tensor]], compute_loss: Callable[[torch.nn.Module, dict[str, torch.Tensor]], torch.Tensor], layers: dict[str, torch.nn.Linear]):
    """Returns list of covariance matrices from TaRA initialization

    Args:
        model: PEFT model to be ran
        batches: calibration patches. 
        compute_loss: loss function directly from the LoRA Trainer
        layers: LoRA layers to be covered
    """
    result: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, layer in layers.items():
        device = layer.weight.device
        sx = torch.zeros((layer.in_features, layer.in_features), dtype=torch.float32, device=device)
        sg = torch.zeros((layer.out_features, layer.out_features), dtype=torch.float32, device=device)
        result[name] = (sx, sg)
        
    state = {"mask": None}
    handles: list[torch.utils.hooks.RemovableHandle]= []  # Per layer handles

    # Register a forward hook here
    def forward_hook(layer_name: str):
        sx, sg = result[layer_name]

        def g_hook(grad_output: torch.Tensor):
            mask = state["mask"].to(grad_output.device)
            g = grad_output[mask].float()
            sg.add_(g.T @ g)

        def fwd_hook(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor):
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
    model.eval()

    # Here accumulate the X^T X and G^T G matrices across the batches.
    try:
        for batch in batches:
            state["mask"] = batch["attention_mask"].bool()
            loss = compute_loss(model, batch)
            loss.backward()
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
        for param_name, param in model.named_parameters():
            if "lora_" in param_name:
                param.grad = None

    return result


def apply_tara_init(trainer, rank, damping, calibration_size):
    """Applies appropriate weight initialization to trainer

    Args:
    """