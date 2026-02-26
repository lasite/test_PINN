"""Visualisation utilities for the cylinder-flow PINN.

Provides functions to plot velocity magnitude, vorticity, pressure,
streamlines, and training loss curves.  All heavy computation is done
with PyTorch; Matplotlib is used only for rendering.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .equations import compute_velocities
from .geometry import CylinderDomain
from .model import PINNModel


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _cylinder_mask(X: torch.Tensor, Y: torch.Tensor, domain: CylinderDomain):
    """Boolean mask: True inside the cylinder."""
    return ((X - domain.cx) ** 2 + (Y - domain.cy) ** 2) <= domain.radius ** 2


@torch.no_grad()
def evaluate_on_grid(
    model: PINNModel,
    domain: CylinderDomain,
    t_val: float,
    nx: int = 200,
    ny: int = 100,
):
    """Evaluate u, v, p on a regular grid.

    Returns numpy arrays ``(U, V, P, X, Y, mask)`` where *mask* marks
    the cylinder interior.
    """
    model.eval()
    # Need gradients for computing velocities from stream function
    x, y, t, X, Y = domain.mesh_grid(nx, ny, t_val)
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)

    with torch.enable_grad():
        u, v, p, _ = compute_velocities(model, x, y, t)

    U = _to_numpy(u).reshape(nx, ny)
    V = _to_numpy(v).reshape(nx, ny)
    P = _to_numpy(p).reshape(nx, ny)
    Xn = _to_numpy(X)
    Yn = _to_numpy(Y)
    mask = _to_numpy(_cylinder_mask(X, Y, domain))

    return U, V, P, Xn, Yn, mask


def plot_velocity_magnitude(
    model: PINNModel,
    domain: CylinderDomain,
    t_val: float = 0.0,
    nx: int = 200,
    ny: int = 100,
    save_path: str | Path | None = None,
):
    """Plot the velocity magnitude field with streamlines."""
    U, V, P, X, Y, mask = evaluate_on_grid(model, domain, t_val, nx, ny)
    speed = np.sqrt(U ** 2 + V ** 2)
    speed[mask] = np.nan

    fig, ax = plt.subplots(figsize=(14, 4))
    cf = ax.contourf(X.T, Y.T, speed.T, levels=50, cmap="jet")
    plt.colorbar(cf, ax=ax, label="Velocity magnitude")

    # streamlines
    xs = np.linspace(domain.x_min, domain.x_max, nx)
    ys = np.linspace(domain.y_min, domain.y_max, ny)
    U_plot = np.copy(U)
    V_plot = np.copy(V)
    U_plot[mask] = 0
    V_plot[mask] = 0
    ax.streamplot(xs, ys, U_plot.T, V_plot.T, color="w", linewidth=0.5, density=1.5)

    # cylinder
    circle = plt.Circle((domain.cx, domain.cy), domain.radius, color="grey", zorder=5)
    ax.add_patch(circle)
    ax.set_xlim(domain.x_min, domain.x_max)
    ax.set_ylim(domain.y_min, domain.y_max)
    ax.set_aspect("equal")
    ax.set_title(f"Velocity magnitude  (t = {t_val:.2f})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_vorticity(
    model: PINNModel,
    domain: CylinderDomain,
    t_val: float = 0.0,
    nx: int = 200,
    ny: int = 100,
    save_path: str | Path | None = None,
):
    """Plot the vorticity field (omega = dv/dx - du/dy)."""
    U, V, _, X, Y, mask = evaluate_on_grid(model, domain, t_val, nx, ny)

    xs = np.linspace(domain.x_min, domain.x_max, nx)
    ys = np.linspace(domain.y_min, domain.y_max, ny)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    # finite-difference vorticity
    dv_dx = np.gradient(V, dx, axis=0)
    du_dy = np.gradient(U, dy, axis=1)
    omega = dv_dx - du_dy
    omega[mask] = np.nan

    fig, ax = plt.subplots(figsize=(14, 4))
    vmax = np.nanmax(np.abs(omega)) * 0.8
    if vmax < 1e-8:
        vmax = 1.0
    cf = ax.contourf(X.T, Y.T, omega.T, levels=50, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(cf, ax=ax, label="Vorticity ω")

    circle = plt.Circle((domain.cx, domain.cy), domain.radius, color="grey", zorder=5)
    ax.add_patch(circle)
    ax.set_xlim(domain.x_min, domain.x_max)
    ax.set_ylim(domain.y_min, domain.y_max)
    ax.set_aspect("equal")
    ax.set_title(f"Vorticity  (t = {t_val:.2f})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_pressure(
    model: PINNModel,
    domain: CylinderDomain,
    t_val: float = 0.0,
    nx: int = 200,
    ny: int = 100,
    save_path: str | Path | None = None,
):
    """Plot the pressure field."""
    _, _, P, X, Y, mask = evaluate_on_grid(model, domain, t_val, nx, ny)
    P[mask] = np.nan

    fig, ax = plt.subplots(figsize=(14, 4))
    cf = ax.contourf(X.T, Y.T, P.T, levels=50, cmap="coolwarm")
    plt.colorbar(cf, ax=ax, label="Pressure p")

    circle = plt.Circle((domain.cx, domain.cy), domain.radius, color="grey", zorder=5)
    ax.add_patch(circle)
    ax.set_xlim(domain.x_min, domain.x_max)
    ax.set_ylim(domain.y_min, domain.y_max)
    ax.set_aspect("equal")
    ax.set_title(f"Pressure  (t = {t_val:.2f})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_loss_history(
    history: list[dict[str, float]],
    save_path: str | Path | None = None,
):
    """Plot training loss curves."""
    epochs = range(1, len(history) + 1)
    total = [h["total"] for h in history]
    pde = [h["pde_x"] + h["pde_y"] for h in history]
    bc = [h["cyl_u"] + h["cyl_v"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(epochs, total, label="Total")
    ax.semilogy(epochs, pde, label="PDE residual")
    ax.semilogy(epochs, bc, label="Cylinder BC")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss history")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_all(
    model: PINNModel,
    domain: CylinderDomain,
    history: list[dict[str, float]],
    t_values: list[float] | None = None,
    output_dir: str | Path = "outputs",
):
    """Generate all visualisation plots and save them to *output_dir*.

    Parameters
    ----------
    t_values : list[float] | None
        Time instants to visualise.  Defaults to five equally-spaced
        values across the time range.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if t_values is None:
        t_min, t_max = domain.t_min, domain.t_max
        t_values = [t_min + i * (t_max - t_min) / 4 for i in range(5)]

    plot_loss_history(history, save_path=out / "loss_history.png")

    for t_val in t_values:
        tag = f"t{t_val:.2f}".replace(".", "_")
        plot_velocity_magnitude(model, domain, t_val, save_path=out / f"velocity_{tag}.png")
        plot_vorticity(model, domain, t_val, save_path=out / f"vorticity_{tag}.png")
        plot_pressure(model, domain, t_val, save_path=out / f"pressure_{tag}.png")

    print(f"All plots saved to {out.resolve()}")
