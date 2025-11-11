"""
Keypoint detection model using CNN backbone with heatmap regression.

This module implements a keypoint detection architecture that:
1. Uses a pre-trained CNN backbone for feature extraction
2. Applies decoder layers to upsample features
3. Predicts a heatmap where peaks represent keypoint locations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple, Optional

from config import MODEL_CONFIG


class KeypointDetector(nn.Module):
    """
    Keypoint detection model using encoder-decoder architecture.
    
    The model predicts a heatmap where Gaussian peaks indicate keypoint locations.
    This is particularly suitable for detecting multiple keypoints (object centers)
    in aerial imagery.
    
    Architecture:
        - Encoder: Pre-trained ResNet backbone
        - Decoder: Upsampling layers with skip connections
        - Head: Final convolution to produce single-channel heatmap
    
    Args:
        backbone: Name of backbone architecture ('resnet18', 'resnet34', etc.)
        pretrained: Whether to use ImageNet pre-trained weights
        heatmap_size: Output heatmap spatial dimensions (H, W)
    """
    
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        heatmap_size: Tuple[int, int] = (64, 64)
    ):
        super().__init__()
        
        self.heatmap_size = heatmap_size
        
        # Initialize backbone
        self.encoder = self._create_encoder(backbone, pretrained)
        
        # Get encoder output channels
        self.encoder_channels = self._get_encoder_channels(backbone)
        
        # Decoder layers for upsampling
        self.decoder = self._create_decoder()
        
        # Final prediction head
        self.head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()  # Output in [0, 1] range
        )
        
        # Initialize weights for decoder and head
        self._initialize_weights()
    
    def _create_encoder(self, backbone: str, pretrained: bool) -> nn.Module:
        """
        Create encoder from pre-trained backbone.
        
        Args:
            backbone: Backbone architecture name
            pretrained: Whether to load ImageNet weights
        
        Returns:
            Encoder module (backbone without final FC layer)
        """
        # Use weights parameter (newer API) instead of pretrained
        weights = "DEFAULT" if pretrained else None
        
        if backbone == "resnet18":
            model = models.resnet18(weights=weights)
        elif backbone == "resnet34":
            model = models.resnet34(weights=weights)
        elif backbone == "resnet50":
            model = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove the final fully connected layer and average pooling
        encoder = nn.Sequential(*list(model.children())[:-2])
        
        return encoder
    
    def _get_encoder_channels(self, backbone: str) -> int:
        """Get number of output channels from encoder."""
        channels_map = {
            "resnet18": 512,
            "resnet34": 512,
            "resnet50": 2048,
        }
        return channels_map.get(backbone, 512)
    
    def _create_decoder(self) -> nn.Module:
        """
        Create decoder for upsampling feature maps.
        
        The decoder progressively upsamples features from the encoder
        to match the desired heatmap size.
        
        Returns:
            Decoder module
        """
        decoder_layers = []
        
        # Decoder path: progressively upsample
        in_channels = self.encoder_channels
        out_channels_list = [256, 128, 64]
        
        for out_channels in out_channels_list:
            decoder_layers.extend([
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])
            in_channels = out_channels
        
        return nn.Sequential(*decoder_layers)
    
    def _initialize_weights(self):
        """Initialize weights for decoder and head layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.weight.requires_grad:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None and m.bias.requires_grad:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight.requires_grad:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None and m.bias.requires_grad:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input image tensor of shape (B, 3, H, W)
        
        Returns:
            Heatmap tensor of shape (B, 1, H_out, W_out)
        """
        # Encoder
        features = self.encoder(x)
        
        # Decoder
        upsampled = self.decoder(features)
        
        # Prediction head
        heatmap = self.head(upsampled)
        
        # Ensure output matches target heatmap size
        if heatmap.shape[2:] != self.heatmap_size:
            heatmap = F.interpolate(
                heatmap,
                size=self.heatmap_size,
                mode='bilinear',
                align_corners=False
            )
        
        return heatmap
    
    def extract_keypoints(
        self,
        heatmap: torch.Tensor,
        threshold: float = 0.3,
        nms_threshold: int = 10
    ) -> list:
        """
        Extract keypoint coordinates from predicted heatmap.
        
        This method finds local maxima in the heatmap that exceed a confidence
        threshold and applies non-maximum suppression.
        
        Args:
            heatmap: Predicted heatmap tensor (B, 1, H, W)
            threshold: Minimum confidence threshold for detection
            nms_threshold: Minimum distance between keypoints (pixels)
        
        Returns:
            List of keypoint coordinates for each image in batch
        """
        batch_size = heatmap.shape[0]
        keypoints_batch = []
        
        for b in range(batch_size):
            hm = heatmap[b, 0].detach().cpu().numpy()
            
            # Find peaks above threshold
            peaks = []
            h, w = hm.shape
            
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    value = hm[y, x]
                    if value < threshold:
                        continue
                    
                    # Check if local maximum
                    neighborhood = hm[y-1:y+2, x-1:x+2]
                    if value == neighborhood.max():
                        peaks.append((x, y, value))
            
            # Apply non-maximum suppression
            peaks = self._nms(peaks, nms_threshold)
            
            keypoints_batch.append(peaks)
        
        return keypoints_batch
    
    def _nms(self, peaks: list, distance: int) -> list:
        """
        Apply non-maximum suppression to remove nearby peaks.
        
        Args:
            peaks: List of (x, y, confidence) tuples
            distance: Minimum distance threshold
        
        Returns:
            Filtered list of peaks
        """
        if len(peaks) == 0:
            return []
        
        # Sort by confidence
        peaks = sorted(peaks, key=lambda p: p[2], reverse=True)
        
        kept_peaks = []
        for peak in peaks:
            x, y, conf = peak
            
            # Check distance to all kept peaks
            keep = True
            for kept_x, kept_y, _ in kept_peaks:
                dist = ((x - kept_x) ** 2 + (y - kept_y) ** 2) ** 0.5
                if dist < distance:
                    keep = False
                    break
            
            if keep:
                kept_peaks.append(peak)
        
        return kept_peaks


class KeypointLoss(nn.Module):
    """
    Loss function for keypoint heatmap prediction.
    
    Combines MSE loss with optional focal loss weighting to handle
    class imbalance (most pixels are background).
    """
    
    def __init__(self, use_focal: bool = False, alpha: float = 2.0, beta: float = 4.0):
        super().__init__()
        self.use_focal = use_focal
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute loss between predicted and target heatmaps.
        
        Args:
            pred: Predicted heatmap (B, 1, H, W)
            target: Target heatmap (B, 1, H, W)
        
        Returns:
            Scalar loss value
        """
        if self.use_focal:
            # Focal loss variant for better handling of easy negatives
            pos_mask = target.eq(1).float()
            neg_mask = target.lt(1).float()
            
            pos_loss = torch.pow(1 - pred, self.alpha) * torch.pow(pred - target, 2) * pos_mask
            neg_loss = torch.pow(pred, self.alpha) * torch.pow(target, self.beta) * \
                       torch.pow(pred - target, 2) * neg_mask
            
            loss = (pos_loss + neg_loss).sum() / (pos_mask.sum() + neg_mask.sum() + 1e-6)
        else:
            # Standard MSE loss
            loss = F.mse_loss(pred, target)
        
        return loss


def create_model(device: Optional[str] = None) -> Tuple[nn.Module, nn.Module]:
    """
    Create model and loss function from configuration.
    
    Args:
        device: Device to place model on ('cuda' or 'cpu')
    
    Returns:
        Tuple of (model, loss_function)
    """
    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create model
    model = KeypointDetector(
        backbone=MODEL_CONFIG["backbone"],
        pretrained=MODEL_CONFIG["pretrained"],
        heatmap_size=MODEL_CONFIG["heatmap_size"]
    )
    
    # Move to device
    model = model.to(device)
    
    # Create loss function
    criterion = KeypointLoss(use_focal=True)
    
    return model, criterion

