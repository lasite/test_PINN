"""Navier-Stokes equation residuals for 2D incompressible flow.

Using the stream-function formulation:
    u  =  dpsi/dy
    v  = -dpsi/dx

The continuity equation is automatically satisfied.  The momentum
equations become the two PDE residual terms computed in this module.
"""

import torch

from .model import PINNModel


def compute_velocities(
    model: PINNModel,
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
):
    """Compute velocity components from the stream function.

    Returns
    -------
    u, v, p, psi : torch.Tensor
    """
    psi, p = model(x, y, t)

    u = torch.autograd.grad(
        psi, y, grad_outputs=torch.ones_like(psi),
        create_graph=True, retain_graph=True,
    )[0]
    v = -torch.autograd.grad(
        psi, x, grad_outputs=torch.ones_like(psi),
        create_graph=True, retain_graph=True,
    )[0]

    return u, v, p, psi


def navier_stokes_residuals(
    model: PINNModel,
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    Re: float,
):
    """Compute Navier-Stokes momentum residuals.

    Parameters
    ----------
    model : PINNModel
    x, y, t : torch.Tensor
        Collocation points with ``requires_grad=True``.
    Re : float
        Reynolds number.

    Returns
    -------
    residual_x, residual_y : torch.Tensor
        Momentum equation residuals in x and y directions.
    u, v, p : torch.Tensor
        Velocity components and pressure (for logging / visualisation).
    """
    u, v, p, _ = compute_velocities(model, x, y, t)
    nu = 1.0 / Re

    # --- first-order derivatives ---
    ones = torch.ones_like(u)
    u_t = torch.autograd.grad(u, t, ones, create_graph=True, retain_graph=True)[0]
    u_x = torch.autograd.grad(u, x, ones, create_graph=True, retain_graph=True)[0]
    u_y = torch.autograd.grad(u, y, ones, create_graph=True, retain_graph=True)[0]

    v_t = torch.autograd.grad(v, t, ones, create_graph=True, retain_graph=True)[0]
    v_x = torch.autograd.grad(v, x, ones, create_graph=True, retain_graph=True)[0]
    v_y = torch.autograd.grad(v, y, ones, create_graph=True, retain_graph=True)[0]

    p_x = torch.autograd.grad(p, x, ones, create_graph=True, retain_graph=True)[0]
    p_y = torch.autograd.grad(p, y, ones, create_graph=True, retain_graph=True)[0]

    # --- second-order derivatives ---
    u_xx = torch.autograd.grad(u_x, x, ones, create_graph=True, retain_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, ones, create_graph=True, retain_graph=True)[0]

    v_xx = torch.autograd.grad(v_x, x, ones, create_graph=True, retain_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, ones, create_graph=True, retain_graph=True)[0]

    # --- residuals ---
    residual_x = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    residual_y = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)

    return residual_x, residual_y, u, v, p
