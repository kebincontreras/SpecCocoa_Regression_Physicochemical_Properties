import torch
import numpy as np
import torch.nn as nn

class SpectralNet(nn.Module):
    def __init__(self, input_dim=100, num_classes=10, architecture=[10, 10]):
        super(SpectralNet, self).__init__()

        architecture.append(num_classes)

        self.architecture = architecture
        self.layers = nn.ModuleList()
        self.input_dim = input_dim

        current_dim = self.input_dim
        for i, layer in enumerate(self.architecture):
            next_dim = layer
            if i == len(self.architecture) - 1:
                self.layers.append(
                    nn.Sequential(nn.Linear(current_dim, next_dim), nn.Identity())
                )
            else:
                self.layers.append(
                    nn.Sequential(nn.Linear(current_dim, next_dim), nn.BatchNorm1d(next_dim), nn.ReLU())
                )
                current_dim = next_dim

    def _make_orthonorm_weights(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Orthonormalize the output of the network using the Cholesky decomposition.

        Parameters
        ----------
        Y : torch.Tensor
            The output of the network.

        Returns
        -------
        torch.Tensor
            The orthonormalized output.

        Notes
        -----
        This function applies QR decomposition to orthonormalize the output (`Y`) of the network.
        The inverse of the R matrix is returned as the orthonormalization weights.
        """

        m = Y.shape[0]
        _, R = torch.linalg.qr(Y)
        R = R + 1e-8  # Add a small value to the diagonal to avoid division by zero
        orthonorm_weights = np.sqrt(m) * torch.inverse(R)
        return orthonorm_weights

    def forward(
            self, x: torch.Tensor, should_update_orth_weights: bool = False
    ) -> torch.Tensor:
        """
        Perform the forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        should_update_orth_weights : bool, optional
            Whether to update the orthonormalization weights using the Cholesky decomposition or not.

        Returns
        -------
        torch.Tensor
            The output tensor.

        Notes
        -----
        This function takes an input tensor `x` and computes the forward pass of the model.
        If `should_update_orth_weights` is set to True, the orthonormalization weights are updated
        using the QR decomposition. The output tensor is returned.
        """

        for layer in self.layers:
            x = layer(x)

        Y_tilde = x

        if should_update_orth_weights:
            self.orthonorm_weights = self._make_orthonorm_weights(Y_tilde)

        Y = Y_tilde  # @ self.orthonorm_weights
        return Y
