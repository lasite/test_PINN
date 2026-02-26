"""Training loop for the cylinder-flow PINN."""

from __future__ import annotations

import time

import torch
import torch.optim as optim

from .equations import compute_velocities, navier_stokes_residuals
from .geometry import CylinderDomain
from .model import PINNModel


def _mse(t: torch.Tensor) -> torch.Tensor:
    return torch.mean(t ** 2)


def compute_loss(
    model: PINNModel,
    domain: CylinderDomain,
    Re: float,
    U_inf: float,
    n_interior: int = 5000,
    n_boundary: int = 500,
    n_initial: int = 1000,
):
    """Compute the total PINN loss.

    Returns ``(total_loss, loss_dict)`` where *loss_dict* contains the
    individual loss components for logging.
    """
    losses: dict[str, torch.Tensor] = {}

    # --- PDE residuals (interior) ---
    x_i, y_i, t_i = domain.sample_interior(n_interior)
    res_x, res_y, _, _, _ = navier_stokes_residuals(model, x_i, y_i, t_i, Re)
    losses["pde_x"] = _mse(res_x)
    losses["pde_y"] = _mse(res_y)

    # --- cylinder surface (no-slip) ---
    x_c, y_c, t_c = domain.sample_cylinder_surface(n_boundary)
    u_c, v_c, _, _ = compute_velocities(model, x_c, y_c, t_c)
    losses["cyl_u"] = _mse(u_c)
    losses["cyl_v"] = _mse(v_c)

    # --- inlet ---
    x_in, y_in, t_in = domain.sample_inlet(n_boundary)
    u_in, v_in, _, _ = compute_velocities(model, x_in, y_in, t_in)
    losses["inlet_u"] = _mse(u_in - U_inf)
    losses["inlet_v"] = _mse(v_in)

    # --- top / bottom (free-stream) ---
    x_tb, y_tb, t_tb = domain.sample_top_bottom(n_boundary // 2)
    u_tb, v_tb, _, _ = compute_velocities(model, x_tb, y_tb, t_tb)
    losses["wall_u"] = _mse(u_tb - U_inf)
    losses["wall_v"] = _mse(v_tb)

    # --- initial condition (uniform flow) ---
    x_0, y_0, t_0 = domain.sample_initial(n_initial)
    u_0, v_0, _, _ = compute_velocities(model, x_0, y_0, t_0)
    losses["ic_u"] = _mse(u_0 - U_inf)
    losses["ic_v"] = _mse(v_0)

    # --- weighted total ---
    w_pde = 1.0
    w_bc = 10.0
    w_ic = 10.0
    total = (
        w_pde * (losses["pde_x"] + losses["pde_y"])
        + w_bc * (losses["cyl_u"] + losses["cyl_v"]
                  + losses["inlet_u"] + losses["inlet_v"]
                  + losses["wall_u"] + losses["wall_v"])
        + w_ic * (losses["ic_u"] + losses["ic_v"])
    )
    losses["total"] = total
    return total, losses


def train(
    model: PINNModel,
    domain: CylinderDomain,
    Re: float = 100.0,
    U_inf: float = 1.0,
    epochs: int = 5000,
    lr: float = 1e-3,
    n_interior: int = 5000,
    n_boundary: int = 500,
    n_initial: int = 1000,
    print_every: int = 100,
    device: torch.device | str = "cpu",
):
    """Train the PINN model.

    Parameters
    ----------
    model : PINNModel
    domain : CylinderDomain
    Re : float
        Reynolds number.
    U_inf : float
        Free-stream velocity.
    epochs : int
        Number of training iterations.
    lr : float
        Learning rate.
    n_interior, n_boundary, n_initial : int
        Number of collocation points per category.
    print_every : int
        Logging frequency.
    device : torch.device | str
        Computation device.

    Returns
    -------
    history : list[dict[str, float]]
        Per-epoch loss values.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

    history: list[dict[str, float]] = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        total_loss, loss_dict = compute_loss(
            model, domain, Re, U_inf,
            n_interior=n_interior,
            n_boundary=n_boundary,
            n_initial=n_initial,
        )
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        record = {k: v.item() for k, v in loss_dict.items()}
        history.append(record)

        if epoch % print_every == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:>5d}/{epochs} | "
                f"Loss {record['total']:.4e} | "
                f"PDE {record['pde_x'] + record['pde_y']:.4e} | "
                f"BC {record['cyl_u'] + record['cyl_v']:.4e} | "
                f"Time {elapsed:.1f}s"
            )

    return history
