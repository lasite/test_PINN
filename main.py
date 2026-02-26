"""Main entry point for the cylinder-flow PINN project.

Usage
-----
Quick demo (fewer epochs, for testing)::

    python main.py --epochs 500 --quick

Full training::

    python main.py --epochs 10000 --Re 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.geometry import CylinderDomain
from src.model import PINNModel
from src.train import train
from src.visualize import plot_all


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PINN for 2D cylinder flow (Kármán vortex street)")
    p.add_argument("--epochs", type=int, default=5000, help="Number of training epochs")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--Re", type=float, default=100.0, help="Reynolds number")
    p.add_argument("--U_inf", type=float, default=1.0, help="Free-stream velocity")
    p.add_argument("--n_interior", type=int, default=5000, help="Interior collocation points")
    p.add_argument("--n_boundary", type=int, default=500, help="Boundary collocation points")
    p.add_argument("--n_initial", type=int, default=1000, help="Initial-condition points")
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 128, 128, 128, 128],
                   help="Hidden layer sizes")
    p.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    p.add_argument("--save_model", type=str, default=None, help="Path to save model checkpoint")
    p.add_argument("--quick", action="store_true", help="Quick demo with reduced settings")
    p.add_argument("--device", type=str, default=None,
                   help="Device (cpu / cuda). Auto-detected if omitted")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Auto-detect device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Quick-demo overrides
    if args.quick:
        args.epochs = min(args.epochs, 500)
        args.n_interior = min(args.n_interior, 2000)
        args.n_boundary = min(args.n_boundary, 200)
        args.n_initial = min(args.n_initial, 500)
        args.hidden = [64, 64, 64, 64]

    # Domain
    domain = CylinderDomain(
        x_range=(-2.0, 8.0),
        y_range=(-2.0, 2.0),
        t_range=(0.0, 5.0),
        cx=0.0,
        cy=0.0,
        radius=0.5,
        device=device,
    )

    # Model
    model = PINNModel(hidden_layers=args.hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Train
    print(f"Training for {args.epochs} epochs  (Re = {args.Re}, U_inf = {args.U_inf})")
    history = train(
        model,
        domain,
        Re=args.Re,
        U_inf=args.U_inf,
        epochs=args.epochs,
        lr=args.lr,
        n_interior=args.n_interior,
        n_boundary=args.n_boundary,
        n_initial=args.n_initial,
        print_every=max(1, args.epochs // 20),
        device=device,
    )

    # Save model
    if args.save_model:
        path = Path(args.save_model)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
        print(f"Model saved to {path}")

    # Visualize
    print("Generating visualizations …")
    plot_all(model, domain, history, output_dir=args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
