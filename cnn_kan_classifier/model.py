"""
CNN + KAN Hybrid Model for Image Classification
Uses ResNet18 as backbone and KAN as classifier
"""
import torch
import torch.nn as nn
from torchvision import models

from kan_layer import KAN, EfficientKAN
import config


class CNN_KAN(nn.Module):
    """
    CNN + KAN Hybrid Model
    
    Architecture:
        CNN backbone (ResNet18 or EfficientNet-B0) -> KAN (classifier)
        
    The backbone extracts features, which are
    then classified by a KAN network.
    """
    
    def __init__(
        self,
        num_classes=config.NUM_CLASSES,
        backbone_features=config.BACKBONE_FEATURES,
        kan_hidden_dims=config.KAN_HIDDEN_DIMS,
        freeze_backbone=config.FREEZE_BACKBONE,
        use_efficient_kan=True,
        grid_size=config.GRID_SIZE,
        spline_order=config.SPLINE_ORDER,
        backbone=config.BACKBONE,
    ):
        super().__init__()
        
        # Load backbone based on config
        if backbone == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            # Remove the final classifier
            self.backbone.classifier = nn.Identity()
            backbone_name = "EfficientNet-B0"
        else:  # Default to resnet18
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            # Remove the final FC layer
            self.backbone.fc = nn.Identity()
            backbone_name = "ResNet18"
        
        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Build KAN classifier
        kan_dims = [backbone_features] + kan_hidden_dims + [num_classes]
        
        if use_efficient_kan:
            self.classifier = EfficientKAN(
                layers_dims=kan_dims,
                grid_size=grid_size,
            )
        else:
            self.classifier = KAN(
                layers_dims=kan_dims,
                grid_size=grid_size,
                spline_order=spline_order,
            )
        
        print(f"CNN_KAN Model initialized:")
        print(f"  - Backbone: {backbone_name} (frozen={freeze_backbone})")
        print(f"  - KAN dims: {kan_dims}")
        print(f"  - Using {'Efficient' if use_efficient_kan else 'B-spline'} KAN")
    
    def forward(self, x):
        # Extract features using backbone
        features = self.backbone(x)  # [batch, 512]
        
        # Classify using KAN
        logits = self.classifier(features)  # [batch, num_classes]
        
        return logits
    
    def get_features(self, x):
        """Get backbone features without classification"""
        return self.backbone(x)


class CNN_MLP(nn.Module):
    """
    Baseline model: CNN + MLP (for comparison)
    """
    
    def __init__(
        self,
        num_classes=config.NUM_CLASSES,
        backbone_features=config.BACKBONE_FEATURES,
        hidden_dims=config.KAN_HIDDEN_DIMS,
        freeze_backbone=config.FREEZE_BACKBONE,
    ):
        super().__init__()
        
        # Load pretrained ResNet18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Build MLP classifier
        layers = []
        in_dim = backbone_features
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
        print(f"CNN_MLP Model initialized:")
        print(f"  - Backbone: ResNet18 (frozen={freeze_backbone})")
        print(f"  - MLP dims: {[backbone_features] + hidden_dims + [num_classes]}")
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


def count_parameters(model):
    """Count trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    # Test models
    batch_size = 4
    x = torch.randn(batch_size, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    
    # Test CNN_KAN
    print("=" * 50)
    model_kan = CNN_KAN()
    out = model_kan(x)
    print(f"Output shape: {out.shape}")
    total, trainable = count_parameters(model_kan)
    print(f"Total params: {total:,} | Trainable: {trainable:,}")
    
    # Test CNN_MLP
    print("=" * 50)
    model_mlp = CNN_MLP()
    out = model_mlp(x)
    print(f"Output shape: {out.shape}")
    total, trainable = count_parameters(model_mlp)
    print(f"Total params: {total:,} | Trainable: {trainable:,}")
