"""Domain geometry and collocation point sampling for cylinder flow.

Coordinate system
-----------------
* The rectangular channel spans ``[x_min, x_max] × [y_min, y_max]``.
* A circular cylinder of diameter *D* is centred at ``(cx, cy)``.
* Time ranges from 0 to ``t_max``.

Default layout (non-dimensionalised by *D*)::

        y_max ─────────────────────────────
              │                             │
              │         ○ (cx, cy)          │  → flow direction
              │                             │
        y_min ─────────────────────────────
            x_min                         x_max
"""

from __future__ import annotations

import torch


class CylinderDomain:
    """Manages the computational domain and sampling.

    Parameters
    ----------
    x_range : tuple[float, float]
        ``(x_min, x_max)`` of the channel.
    y_range : tuple[float, float]
        ``(y_min, y_max)`` of the channel.
    t_range : tuple[float, float]
        ``(0, t_max)``.
    cx, cy : float
        Centre of the cylinder.
    radius : float
        Radius of the cylinder.
    device : torch.device | str
        Computation device.
    """

    def __init__(
        self,
        x_range: tuple[float, float] = (-2.0, 8.0),
        y_range: tuple[float, float] = (-2.0, 2.0),
        t_range: tuple[float, float] = (0.0, 5.0),
        cx: float = 0.0,
        cy: float = 0.0,
        radius: float = 0.5,
        device: torch.device | str = "cpu",
    ):
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.t_min, self.t_max = t_range
        self.cx = cx
        self.cy = cy
        self.radius = radius
        self.device = torch.device(device)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _rand(self, n: int, lo: float, hi: float) -> torch.Tensor:
        return (lo + (hi - lo) * torch.rand(n, 1, device=self.device)).requires_grad_(True)

    def _is_outside_cylinder(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return ((x - self.cx) ** 2 + (y - self.cy) ** 2) > self.radius ** 2

    # ------------------------------------------------------------------
    # collocation points
    # ------------------------------------------------------------------

    def sample_interior(self, n: int):
        """Sample points inside the channel but outside the cylinder.

        Returns ``(x, y, t)`` each of shape ``(m, 1)`` where ``m <= n``.
        """
        # oversample then reject points inside cylinder
        factor = 1.2
        n_sample = int(n * factor)
        x = self._rand(n_sample, self.x_min, self.x_max)
        y = self._rand(n_sample, self.y_min, self.y_max)
        mask = self._is_outside_cylinder(x, y).squeeze()
        x = x[mask][:n]
        y = y[mask][:n]
        t = self._rand(x.shape[0], self.t_min, self.t_max)
        return x, y, t

    def sample_cylinder_surface(self, n: int):
        """Sample points on the cylinder boundary.

        Returns ``(x, y, t)``.
        """
        theta = torch.linspace(0, 2 * torch.pi, n, device=self.device).unsqueeze(1)
        x = (self.cx + self.radius * torch.cos(theta)).requires_grad_(True)
        y = (self.cy + self.radius * torch.sin(theta)).requires_grad_(True)
        t = self._rand(n, self.t_min, self.t_max)
        return x, y, t

    def sample_inlet(self, n: int):
        """Left boundary: x = x_min."""
        x = (torch.full((n, 1), self.x_min, device=self.device)).requires_grad_(True)
        y = self._rand(n, self.y_min, self.y_max)
        t = self._rand(n, self.t_min, self.t_max)
        return x, y, t

    def sample_outlet(self, n: int):
        """Right boundary: x = x_max."""
        x = (torch.full((n, 1), self.x_max, device=self.device)).requires_grad_(True)
        y = self._rand(n, self.y_min, self.y_max)
        t = self._rand(n, self.t_min, self.t_max)
        return x, y, t

    def sample_top_bottom(self, n: int):
        """Top and bottom walls.

        Returns ``(x, y, t)`` with *n* points on each wall (2*n* total).
        """
        x_top = self._rand(n, self.x_min, self.x_max)
        y_top = (torch.full((n, 1), self.y_max, device=self.device)).requires_grad_(True)
        t_top = self._rand(n, self.t_min, self.t_max)

        x_bot = self._rand(n, self.x_min, self.x_max)
        y_bot = (torch.full((n, 1), self.y_min, device=self.device)).requires_grad_(True)
        t_bot = self._rand(n, self.t_min, self.t_max)

        x = torch.cat([x_top, x_bot])
        y = torch.cat([y_top, y_bot])
        t = torch.cat([t_top, t_bot])
        return x, y, t

    def sample_initial(self, n: int):
        """Initial condition at t = 0.

        Returns ``(x, y, t)`` with *t* = 0.
        """
        n_sample = int(n * 1.2)
        x = self._rand(n_sample, self.x_min, self.x_max)
        y = self._rand(n_sample, self.y_min, self.y_max)
        mask = self._is_outside_cylinder(x, y).squeeze()
        x = x[mask][:n]
        y = y[mask][:n]
        t = torch.zeros(x.shape[0], 1, device=self.device, requires_grad=True)
        return x, y, t

    # ------------------------------------------------------------------
    # visualization grid
    # ------------------------------------------------------------------

    def mesh_grid(self, nx: int = 200, ny: int = 100, t_val: float = 0.0):
        """Create a regular mesh grid for plotting.

        Points inside the cylinder are kept (caller can mask them).

        Returns ``(x, y, t, X, Y)`` where *X, Y* are 2-D mesh arrays.
        """
        xs = torch.linspace(self.x_min, self.x_max, nx, device=self.device)
        ys = torch.linspace(self.y_min, self.y_max, ny, device=self.device)
        X, Y = torch.meshgrid(xs, ys, indexing="ij")
        x = X.reshape(-1, 1).requires_grad_(True)
        y = Y.reshape(-1, 1).requires_grad_(True)
        t = torch.full_like(x, t_val).requires_grad_(True)
        return x, y, t, X, Y
