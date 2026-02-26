"""Neural network model for Physics-Informed Neural Networks (PINN).

The network takes spatial-temporal coordinates (x, y, t) as input and
outputs the stream function psi and pressure p. Velocity components are
derived from psi via automatic differentiation:
    u =  dpsi/dy
    v = -dpsi/dx
This formulation automatically satisfies the continuity equation.
"""

import torch
import torch.nn as nn


class PINNModel(nn.Module):
    """Fully-connected neural network for PINN.

    Parameters
    ----------
    hidden_layers : list[int]
        Number of neurons in each hidden layer.
    """

    def __init__(self, hidden_layers: list[int] | None = None):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [128, 128, 128, 128, 128]

        layers: list[nn.Module] = []
        in_features = 3  # (x, y, t)
        for h in hidden_layers:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.Tanh())
            in_features = h
        layers.append(nn.Linear(in_features, 2))  # (psi, p)

        self.net = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Xavier initialization for better convergence."""
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor):
        """Forward pass.

        Parameters
        ----------
        x, y, t : torch.Tensor
            Spatial and temporal coordinates, each of shape ``(N, 1)``.

        Returns
        -------
        psi, p : torch.Tensor
            Stream function and pressure, each of shape ``(N, 1)``.
        """
        inputs = torch.cat([x, y, t], dim=1)
        out = self.net(inputs)
        psi = out[:, 0:1]
        p = out[:, 1:2]
        return psi, p
