"""Tests for the cylinder-flow PINN components."""

import pytest
import torch

from src.equations import compute_velocities, navier_stokes_residuals
from src.geometry import CylinderDomain
from src.model import PINNModel
from src.train import compute_loss


# ---- Model tests ----

class TestPINNModel:
    def test_output_shapes(self):
        model = PINNModel(hidden_layers=[32, 32])
        x = torch.randn(10, 1)
        y = torch.randn(10, 1)
        t = torch.randn(10, 1)
        psi, p = model(x, y, t)
        assert psi.shape == (10, 1)
        assert p.shape == (10, 1)

    def test_default_hidden_layers(self):
        model = PINNModel()
        x = torch.randn(5, 1)
        y = torch.randn(5, 1)
        t = torch.randn(5, 1)
        psi, p = model(x, y, t)
        assert psi.shape == (5, 1)
        assert p.shape == (5, 1)

    def test_gradient_flow(self):
        """Gradients should flow through the stream function."""
        model = PINNModel(hidden_layers=[32, 32])
        x = torch.randn(10, 1, requires_grad=True)
        y = torch.randn(10, 1, requires_grad=True)
        t = torch.randn(10, 1, requires_grad=True)
        psi, p = model(x, y, t)
        loss = psi.sum() + p.sum()
        loss.backward()
        assert x.grad is not None
        assert y.grad is not None


# ---- Geometry tests ----

class TestCylinderDomain:
    def setup_method(self):
        self.domain = CylinderDomain()

    def test_sample_interior(self):
        x, y, t = self.domain.sample_interior(100)
        assert x.shape[0] <= 100
        assert x.shape[1] == 1
        # No point should be inside the cylinder
        dist = (x - self.domain.cx) ** 2 + (y - self.domain.cy) ** 2
        assert torch.all(dist > self.domain.radius ** 2 - 1e-6)

    def test_sample_cylinder_surface(self):
        x, y, t = self.domain.sample_cylinder_surface(50)
        dist = torch.sqrt((x - self.domain.cx) ** 2 + (y - self.domain.cy) ** 2)
        torch.testing.assert_close(dist, torch.full_like(dist, self.domain.radius), atol=1e-5, rtol=1e-5)

    def test_sample_inlet(self):
        x, y, t = self.domain.sample_inlet(50)
        torch.testing.assert_close(x, torch.full_like(x, self.domain.x_min))

    def test_sample_outlet(self):
        x, y, t = self.domain.sample_outlet(50)
        torch.testing.assert_close(x, torch.full_like(x, self.domain.x_max))

    def test_sample_initial(self):
        x, y, t = self.domain.sample_initial(100)
        torch.testing.assert_close(t, torch.zeros_like(t))

    def test_mesh_grid(self):
        x, y, t, X, Y = self.domain.mesh_grid(20, 10, t_val=1.0)
        assert x.shape == (200, 1)
        assert X.shape == (20, 10)


# ---- Equations tests ----

class TestEquations:
    def test_compute_velocities_shapes(self):
        model = PINNModel(hidden_layers=[32, 32])
        x = torch.randn(10, 1, requires_grad=True)
        y = torch.randn(10, 1, requires_grad=True)
        t = torch.randn(10, 1, requires_grad=True)
        u, v, p, psi = compute_velocities(model, x, y, t)
        assert u.shape == (10, 1)
        assert v.shape == (10, 1)

    def test_ns_residuals_shapes(self):
        model = PINNModel(hidden_layers=[32, 32])
        x = torch.randn(10, 1, requires_grad=True)
        y = torch.randn(10, 1, requires_grad=True)
        t = torch.randn(10, 1, requires_grad=True)
        res_x, res_y, u, v, p = navier_stokes_residuals(model, x, y, t, Re=100.0)
        assert res_x.shape == (10, 1)
        assert res_y.shape == (10, 1)


# ---- Training tests ----

class TestTraining:
    def test_compute_loss_runs(self):
        device = "cpu"
        domain = CylinderDomain(device=device)
        model = PINNModel(hidden_layers=[32, 32]).to(device)
        total_loss, loss_dict = compute_loss(
            model, domain, Re=100.0, U_inf=1.0,
            n_interior=50, n_boundary=20, n_initial=30,
        )
        assert total_loss.item() > 0
        assert "total" in loss_dict
        assert "pde_x" in loss_dict

    def test_loss_is_differentiable(self):
        device = "cpu"
        domain = CylinderDomain(device=device)
        model = PINNModel(hidden_layers=[32, 32]).to(device)
        total_loss, _ = compute_loss(
            model, domain, Re=100.0, U_inf=1.0,
            n_interior=50, n_boundary=20, n_initial=30,
        )
        total_loss.backward()
        # The last layer bias does not contribute to the loss because
        # we only use spatial/temporal derivatives of psi and p (the
        # bias is a constant that vanishes under differentiation).
        params_with_grad = [p for p in model.parameters() if p.grad is not None]
        assert len(params_with_grad) > 0
        total_params = sum(1 for _ in model.parameters())
        assert len(params_with_grad) >= total_params - 1
