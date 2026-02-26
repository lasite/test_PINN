# PINN for 2D Cylinder Flow — Kármán Vortex Street

Physics-Informed Neural Network (PINN) for simulating 2D incompressible
flow around a circular cylinder and visualising the resulting
[Kármán vortex street](https://en.wikipedia.org/wiki/K%C3%A1rm%C3%A1n_vortex_street).

## Method

The network takes spatial-temporal coordinates **(x, y, t)** as input and
outputs a **stream function ψ** and **pressure p**.  Velocity components are
derived via automatic differentiation:

```
u =  ∂ψ/∂y
v = −∂ψ/∂x
```

This formulation **automatically satisfies the continuity equation**.  The
loss function enforces:

| Term | Description |
|------|-------------|
| PDE residual | 2D incompressible Navier–Stokes momentum equations |
| Cylinder BC | No-slip condition on the cylinder surface |
| Inlet BC | Uniform free-stream velocity |
| Wall BC | Free-stream condition on top/bottom boundaries |
| Initial condition | Uniform flow at t = 0 |

## Project Structure

```
├── main.py              # Entry point (training + visualisation)
├── requirements.txt     # Python dependencies
├── src/
│   ├── __init__.py
│   ├── model.py         # PINN neural network architecture
│   ├── equations.py     # Navier–Stokes residuals via autograd
│   ├── geometry.py      # Domain geometry & collocation sampling
│   ├── train.py         # Training loop with weighted loss
│   └── visualize.py     # Plotting (velocity, vorticity, pressure)
└── tests/
    └── test_model.py    # Unit tests
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Quick demo (~50 s on CPU)
python main.py --quick

# Full training (recommended for clear vortex street)
python main.py --epochs 10000 --Re 100

# Custom run
python main.py --epochs 20000 --Re 200 --lr 5e-4 --hidden 256 256 256 256 256
```

Results (velocity, vorticity, pressure plots and loss curves) are saved
to the `outputs/` directory.

### Command-line Options

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 5000 | Training iterations |
| `--lr` | 1e-3 | Learning rate |
| `--Re` | 100 | Reynolds number |
| `--U_inf` | 1.0 | Free-stream velocity |
| `--n_interior` | 5000 | Interior collocation points |
| `--n_boundary` | 500 | Boundary points per edge |
| `--n_initial` | 1000 | Initial-condition points |
| `--hidden` | 128×5 | Hidden layer sizes |
| `--output_dir` | `outputs/` | Where to save plots |
| `--save_model` | — | Path to save model checkpoint |
| `--quick` | — | Reduced settings for fast demo |
| `--device` | auto | `cpu` or `cuda` |

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## Requirements

* Python ≥ 3.10
* PyTorch ≥ 2.0
* NumPy ≥ 1.24
* Matplotlib ≥ 3.7