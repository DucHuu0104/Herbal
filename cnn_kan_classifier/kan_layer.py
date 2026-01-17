"""
Kolmogorov-Arnold Network (KAN) Layer Implementation
Based on the paper: "KAN: Kolmogorov-Arnold Networks"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SplineLinear(nn.Linear):
    """Linear layer with spline-based initialization"""
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))


class RadialBasisFunction(nn.Module):
    """Radial Basis Function for KAN"""
    
    def __init__(self, grid_min=-2.0, grid_max=2.0, num_grids=8, denominator=None):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)
        
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.register_buffer("grid", grid)
    
    def forward(self, x):
        # x: [batch, in_features]
        # output: [batch, in_features, num_grids]
        return torch.exp(-((x[..., None] - self.grid) ** 2) / (2 * self.denominator ** 2))


class KANLinear(nn.Module):
    """
    KAN Linear Layer using B-spline basis functions
    
    Implements: f(x) = w_b * b(x) + w_s * spline(x)
    where b(x) is the base function (SiLU) and spline(x) is learned
    """
    
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # Grid for B-splines
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1) * h
            + grid_range[0]
        ).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)
        
        # Spline weights
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, grid_size + spline_order)
            * scale_noise / (grid_size + spline_order)
        )
        
        # Base weights (for residual connection)
        self.base_weight = nn.Parameter(
            torch.randn(out_features, in_features) * scale_base / math.sqrt(in_features)
        )
        
        # Scaling factors
        self.spline_scaler = nn.Parameter(
            torch.ones(out_features, in_features) * scale_spline
        )
        
        # Base activation
        self.base_activation = base_activation()
        
    def forward(self, x):
        # x: [batch_size, in_features]
        batch_size = x.size(0)
        
        # Base function (residual)
        base_output = F.linear(self.base_activation(x), self.base_weight)
        
        # B-spline computation
        x_expanded = x.unsqueeze(-1)  # [batch, in_features, 1]
        
        # Compute B-spline bases
        bases = self._compute_bspline_basis(x)  # [batch, in_features, grid_size + spline_order]
        
        # Apply spline weights
        # spline_weight: [out_features, in_features, num_bases]
        # bases: [batch, in_features, num_bases]
        spline_output = torch.einsum('bik,oik->bo', bases, self.spline_weight * self.spline_scaler.unsqueeze(-1))
        
        return base_output + spline_output
    
    def _compute_bspline_basis(self, x):
        """
        Compute B-spline basis functions
        Uses Cox-de Boor recursion formula
        """
        # x: [batch, in_features]
        x = x.unsqueeze(-1)  # [batch, in_features, 1]
        grid = self.grid  # [in_features, num_grid_points]
        
        # Order 0 bases
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).float()
        
        # Recursive computation for higher orders
        for k in range(1, self.spline_order + 1):
            left_num = x - grid[:, :-(k + 1)]
            left_den = grid[:, k:-1] - grid[:, :-(k + 1)]
            left = left_num / (left_den + 1e-8) * bases[:, :, :-1]
            
            right_num = grid[:, k + 1:] - x
            right_den = grid[:, k + 1:] - grid[:, 1:-k]
            right = right_num / (right_den + 1e-8) * bases[:, :, 1:]
            
            bases = left + right
        
        return bases


class KAN(nn.Module):
    """
    Kolmogorov-Arnold Network
    A stack of KAN layers
    """
    
    def __init__(
        self,
        layers_dims,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        """
        Args:
            layers_dims: List of layer dimensions [in_dim, hidden1, hidden2, ..., out_dim]
            grid_size: Number of grid intervals for splines
            spline_order: Order of B-splines (typically 3 for cubic)
        """
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        for i in range(len(layers_dims) - 1):
            self.layers.append(
                KANLinear(
                    layers_dims[i],
                    layers_dims[i + 1],
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class EfficientKANLinear(nn.Module):
    """
    Efficient KAN layer using Radial Basis Functions
    Faster than B-spline version while maintaining expressiveness
    """
    
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=8,
        base_activation=nn.SiLU,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Base linear transformation
        self.base_linear = nn.Linear(in_features, out_features)
        
        # RBF transformation
        self.rbf = RadialBasisFunction(num_grids=grid_size)
        self.spline_linear = nn.Linear(in_features * grid_size, out_features)
        
        # Base activation
        self.base_activation = base_activation()
        
        # Initialize
        nn.init.zeros_(self.spline_linear.weight)
    
    def forward(self, x):
        # Base output
        base_output = self.base_linear(self.base_activation(x))
        
        # Spline output
        spline_basis = self.rbf(x)  # [batch, in_features, grid_size]
        spline_basis = spline_basis.view(x.size(0), -1)  # [batch, in_features * grid_size]
        spline_output = self.spline_linear(spline_basis)
        
        return base_output + spline_output


class EfficientKAN(nn.Module):
    """
    Efficient KAN using RBF basis functions
    """
    
    def __init__(
        self,
        layers_dims,
        grid_size=8,
        base_activation=nn.SiLU,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        for i in range(len(layers_dims) - 1):
            self.layers.append(
                EfficientKANLinear(
                    layers_dims[i],
                    layers_dims[i + 1],
                    grid_size=grid_size,
                    base_activation=base_activation,
                )
            )
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    # Test KAN
    batch_size = 4
    in_features = 512
    out_features = 20
    
    # Test B-spline KAN
    kan = KAN([in_features, 64, 32, out_features])
    x = torch.randn(batch_size, in_features)
    out = kan(x)
    print(f"B-spline KAN output shape: {out.shape}")
    
    # Test Efficient KAN
    efficient_kan = EfficientKAN([in_features, 64, 32, out_features])
    out = efficient_kan(x)
    print(f"Efficient KAN output shape: {out.shape}")
    
    # Count parameters
    kan_params = sum(p.numel() for p in kan.parameters())
    efficient_kan_params = sum(p.numel() for p in efficient_kan.parameters())
    print(f"B-spline KAN params: {kan_params:,}")
    print(f"Efficient KAN params: {efficient_kan_params:,}")
