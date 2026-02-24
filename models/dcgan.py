"""
DCGAN (Deep Convolutional Generative Adversarial Network) Models

This module contains the convolutional generator and discriminator implementations
used in Lesson 11 to improve image quality using convolutional architectures.

Architecture:
- Generator: ConvTranspose2d layers for upsampling (100-dim noise → 32×32 RGB image)
- Discriminator: Conv2d layers for downsampling (32×32 RGB → real/fake probability)
- Dataset: CIFAR-10 (32×32 color images)

Key Improvements over BasicGAN (Lesson 3-4):
1. Convolutional architecture instead of fully-connected
2. Batch normalization for training stability
3. Fractional-strided convolutions (ConvTranspose2d) for upsampling
4. Works with color (3-channel) images instead of grayscale
5. Better quality generated images

Training Patterns:
- Generator: Maps noise to realistic CIFAR-10 images
- Discriminator: Learns to distinguish real CIFAR-10 from generated images
- Both networks use Conv2d/ConvTranspose2d with BatchNorm
- LeakyReLU activations for better gradient flow

References:
- Lesson 11: Building DCGAN with PyTorch
- Paper: Radford et al., "Unsupervised Representation Learning with Deep Convolutional GANs" (2015)
"""

from typing import Tuple

import torch
import torch.nn as nn


class DCGANGenerator(nn.Module):
    """
    DCGAN Generator: Converts noise to realistic CIFAR-10 images via upsampling.

    Architecture:
    - Input: noise vector (batch_size, latent_dim)
    - FC layer: Project to spatial features (batch_size, 256, 4, 4)
    - Upsample using ConvTranspose2d:
      - (256, 4, 4) → (128, 8, 8)
      - (128, 8, 8) → (64, 16, 16)
      - (64, 16, 16) → (3, 32, 32)
    - Output: RGB image in [-1, 1]

    Design Principles:
    - ConvTranspose2d with stride=2 and padding=1 doubles spatial dimensions
    - BatchNorm after each transposed convolution for training stability
    - ReLU activations in hidden layers
    - Tanh output layer for [-1, 1] pixel range
    - No bias in conv layers (BatchNorm has learnable bias)

    Why Fractional-Strided Convolutions?
    - ConvTranspose2d reverses a convolution operation
    - With stride > 1, it upsamples the feature maps
    - Better gradient flow than upsampling followed by convolution
    """

    def __init__(
        self,
        latent_dim: int = 100,
        num_channels: int = 3,
        feature_maps: int = 64,
    ):
        """
        Args:
            latent_dim: Dimension of input noise vector (typically 100)
            num_channels: Number of output channels (3 for RGB)
            feature_maps: Base number of feature maps (64)
        """
        super(DCGANGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        self.feature_maps = feature_maps

        # Fully-connected layer: noise -> spatial features
        # Input: (batch, latent_dim)
        # Output: (batch, feature_maps * 4, 4, 4) = (batch, 256, 4, 4)
        self.fc = nn.Linear(latent_dim, feature_maps * 4 * 4 * 4)

        # Convolutional layers: Upsample from 4x4 to 32x32
        self.layers = nn.Sequential(
            # Layer 1: (256, 4, 4) → (128, 8, 8)
            nn.ConvTranspose2d(
                feature_maps * 4,
                feature_maps * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(inplace=True),
            # Layer 2: (128, 8, 8) → (64, 16, 16)
            nn.ConvTranspose2d(
                feature_maps * 2,
                feature_maps,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(inplace=True),
            # Layer 3: (64, 16, 16) → (3, 32, 32)
            nn.ConvTranspose2d(
                feature_maps,
                num_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.Tanh(),  # Output in [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generate image from noise vector.

        Args:
            z: Noise tensor of shape (batch_size, latent_dim)

        Returns:
            img: Generated image of shape (batch_size, num_channels, 32, 32)
        """
        # Project noise to spatial features
        x = self.fc(z)  # (batch_size, latent_dim) -> (batch_size, 256*4)
        x = x.view(x.size(0), self.feature_maps * 4, 4, 4)  # Reshape to spatial

        # Upsample through conv layers
        x = self.layers(x)  # (batch_size, 3, 32, 32)

        return x


class DCGANDiscriminator(nn.Module):
    """
    DCGAN Discriminator: Classifies images as real/fake via downsampling.

    Architecture:
    - Input: RGB image (batch_size, 3, 32, 32)
    - Downsample using Conv2d:
      - (3, 32, 32) → (64, 16, 16)
      - (64, 16, 16) → (128, 8, 8)
      - (128, 8, 8) → (256, 4, 4)
    - FC layer: Flatten and classify (batch_size, 256*4*4) → (batch_size, 1)
    - Output: Real/fake probability in [0, 1]

    Design Principles:
    - Conv2d with stride=2 and padding=1 halves spatial dimensions
    - BatchNorm after convolutions for training stability
    - LeakyReLU (slope=0.2) activations for better GAN training
    - Sigmoid output layer for [0, 1] probability
    - No bias in conv layers (BatchNorm has learnable bias)

    Why LeakyReLU?
    - Standard ReLU can cause "dying ReLU" problem in GANs
    - LeakyReLU allows small negative gradients, improving training stability
    - Small positive slope (0.2) recommended for GANs

    Why Stride-2 Convolutions?
    - Replaces maxpooling for downsampling
    - Learned downsampling is more effective than fixed pooling
    - Better gradient flow through the network
    """

    def __init__(
        self,
        num_channels: int = 3,
        feature_maps: int = 64,
    ):
        """
        Args:
            num_channels: Number of input channels (3 for RGB)
            feature_maps: Base number of feature maps (64)
        """
        super(DCGANDiscriminator, self).__init__()
        self.num_channels = num_channels
        self.feature_maps = feature_maps

        self.layers = nn.Sequential(
            # Layer 1: (3, 32, 32) → (64, 16, 16)
            nn.Conv2d(
                num_channels,
                feature_maps,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(feature_maps),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 2: (64, 16, 16) → (128, 8, 8)
            nn.Conv2d(
                feature_maps,
                feature_maps * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 3: (128, 8, 8) → (256, 4, 4)
            nn.Conv2d(
                feature_maps * 2,
                feature_maps * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(feature_maps * 4 * 4 * 4, 1),
            nn.Sigmoid(),  # Output probability
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify image as real (1) or fake (0).

        Args:
            x: Image tensor of shape (batch_size, num_channels, 32, 32)

        Returns:
            prob: Classification probability of shape (batch_size, 1)
        """
        # Downsample through conv layers
        x = self.layers(x)  # (batch_size, 256, 4, 4)

        # Flatten
        x = x.view(x.size(0), -1)  # (batch_size, 256*4*4)

        # Classify
        x = self.fc(x)  # (batch_size, 1)

        return x


def create_dcgan_models(
    latent_dim: int = 100,
    num_channels: int = 3,
    device: str = "cpu",
) -> Tuple[DCGANGenerator, DCGANDiscriminator]:
    """
    Helper function to create DCGAN generator and discriminator.

    Args:
        latent_dim: Dimension of noise vector
        num_channels: Number of image channels (3 for RGB)
        device: 'cpu' or 'cuda'

    Returns:
        Tuple of (generator, discriminator) initialized and moved to device

    Example:
        >>> generator, discriminator = create_dcgan_models(device='cuda')
        >>> print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    """
    generator = DCGANGenerator(
        latent_dim=latent_dim,
        num_channels=num_channels,
        feature_maps=64,
    ).to(device)

    discriminator = DCGANDiscriminator(
        num_channels=num_channels,
        feature_maps=64,
    ).to(device)

    return generator, discriminator


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(
    generator: nn.Module, discriminator: nn.Module, latent_dim: int = 100
):
    """Print model architecture and parameter counts."""
    print("\n" + "=" * 80)
    print("DCGAN Model Summary")
    print("=" * 80)

    print("\n🎨 GENERATOR (Noise → Image)")
    print("-" * 80)
    print(generator)
    gen_params = count_parameters(generator)
    print(f"\nTotal Parameters: {gen_params:,}")

    print("\n\n🔍 DISCRIMINATOR (Image → Classification)")
    print("-" * 80)
    print(discriminator)
    disc_params = count_parameters(discriminator)
    print(f"\nTotal Parameters: {disc_params:,}")

    print("\n" + "=" * 80)
    print(f"Total DCGAN Parameters: {gen_params + disc_params:,}")
    print(f"  Generator:     {gen_params:,}")
    print(f"  Discriminator: {disc_params:,}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    import sys

    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Create models
    generator, discriminator = create_dcgan_models(
        latent_dim=100, num_channels=3, device=str(device)
    )

    # Print summary
    print_model_summary(generator, discriminator, latent_dim=100)

    # Test forward passes with shape verification
    print("\n✓ Shape Verification:")
    print("-" * 80)

    batch_size = 4
    latent_dim = 100

    # Generator forward pass
    z = torch.randn(batch_size, latent_dim, device=device)
    fake_images = generator(z)
    print(f"Generator Input (noise):     {z.shape}")
    print(f"Generator Output (images):   {fake_images.shape}")
    assert fake_images.shape == (
        batch_size,
        3,
        32,
        32,
    ), "Generator output shape mismatch!"

    # Discriminator forward pass
    D_output = discriminator(fake_images)
    print(f"Discriminator Input (images): {fake_images.shape}")
    print(f"Discriminator Output (prob):  {D_output.shape}")
    assert D_output.shape == (batch_size, 1), "Discriminator output shape mismatch!"

    print("\n✓ All shape assertions passed!")
    print("=" * 80)
