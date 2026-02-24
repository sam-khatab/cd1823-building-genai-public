"""
Conditional GAN (cGAN) Models

This module contains the conditional generator and discriminator implementations
used in Lesson 13 to generate class-specific images.

Architecture:
- Generator: Takes noise + class label, generates class-specific CIFAR-10 images
- Discriminator: Takes image + class label, classifies as real/fake
- Dataset: CIFAR-10 (32×32 color images with 10 classes)

Key Feature: Class Conditioning
- Both G and D receive the class label as additional input
- Generator: Label embedding concatenated with noise before upsampling
- Discriminator: Label embedding concatenated with image features before classification
- Enforces label-image consistency through conditional losses

Applications:
- Generate images of specific classes (e.g., only dogs, only cars)
- Class-aware discriminator improves training stability
- More control over generation than unconditional GAN

References:
- Lesson 13: Implementing Conditional GAN
- Paper: Mirza & Osindero, "Conditional Generative Adversarial Nets" (2014)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class ConditionalGenerator(nn.Module):
    """
    Conditional Generator: Generates class-specific images.

    Architecture:
    - Input: z-vector (latent_dim,) + class label (num_classes,)
    - Label embedding: One-hot → learnable embedding
    - Concatenation: [z, embedded_label]
    - FC layer: project to (256, 4, 4)
    - ConvTranspose2d layers with stride=2 for upsampling
    - Batch Normalization and ReLU between layers
    - Output: (3, 32, 32) RGB image with Tanh

    Conditioning Strategy:
    - Label embedding reduces dimensionality (num_classes → label_dim)
    - Concatenated with noise early (at FC input)
    - Allows generator to use label throughout network
    - Ensures generated image matches the requested class

    Design Principles:
    - Label information is preserved throughout the network
    - Early concatenation allows label to influence all layers
    - Embedding reduces parameter explosion vs one-hot encoding
    """

    def __init__(
        self,
        latent_dim: int = 100,
        num_classes: int = 10,
        label_dim: int = 50,
        num_channels: int = 3,
    ):
        """
        Args:
            latent_dim: Dimension of latent noise vector
            num_classes: Number of classes (10 for CIFAR-10)
            label_dim: Dimension of label embedding
            num_channels: Number of output channels (3 for RGB)
        """
        super(ConditionalGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.label_dim = label_dim
        self.num_channels = num_channels

        # Label embedding: one-hot class → continuous embedding
        self.label_embedding = nn.Embedding(num_classes, label_dim)

        # FC layer: concatenated [z + embedded_label] → spatial features
        # Input: latent_dim + label_dim
        # Output: 256 * 4 * 4 = 4096
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + label_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256 * 4 * 4),
            nn.ReLU(inplace=True),
        )

        # Convolutional layers with upsampling
        self.conv_layers = nn.Sequential(
            # Layer 1: (256, 4, 4) → (128, 8, 8)
            nn.ConvTranspose2d(
                256, 128, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Layer 2: (128, 8, 8) → (64, 16, 16)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Layer 3: (64, 16, 16) → (32, 32, 32)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Output layer: (32, 32, 32) → (num_channels, 32, 32)
            nn.ConvTranspose2d(
                32, num_channels, kernel_size=4, stride=1, padding=1, bias=False
            ),
            nn.Tanh(),  # Output in range [-1, 1]
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Generate class-specific image.

        Args:
            z: Noise vector (batch_size, latent_dim)
            labels: Class labels (batch_size,) - LongTensor with values 0-9

        Returns:
            Generated image (batch_size, num_channels, 32, 32)
        """
        # Embed class labels
        embedded_labels = self.label_embedding(labels)  # (batch_size, label_dim)

        # Concatenate noise and embedded label
        z_label = torch.cat(
            [z, embedded_labels], dim=1
        )  # (batch_size, latent_dim + label_dim)

        # Project through FC layers
        x = self.fc(z_label)

        # Reshape to spatial features
        x = x.view(x.size(0), 256, 4, 4)

        # Upsample through convolutional layers
        x = self.conv_layers(x)

        return x


class ConditionalDiscriminator(nn.Module):
    """
    Conditional Discriminator: Classifies image authenticity given class label.

    Architecture:
    - Input: image (3, 32, 32) + class label (num_classes,)
    - Image path: Conv2d layers with stride=2 for downsampling
    - Label embedding: One-hot class → learnable embedding
    - Concatenation: [flattened_features, embedded_label]
    - FC layers: classify as real/fake given the label
    - Output: (1,) probability

    Conditioning Strategy:
    - Label embedded separately from image features
    - Concatenated after convolutions (spatial info preserved)
    - Discriminator must match image to label for real classification
    - Forces generator to respect class conditioning
    - Improves training stability by providing additional context

    Design Principles:
    - Image features extracted first through convolutions
    - Label embedding combined at the end for classification
    - Allows discriminator to learn image-label matching
    - LeakyReLU for better GAN training dynamics
    """

    def __init__(
        self,
        num_classes: int = 10,
        label_dim: int = 50,
        num_channels: int = 3,
    ):
        """
        Args:
            num_classes: Number of classes (10 for CIFAR-10)
            label_dim: Dimension of label embedding
            num_channels: Number of input channels (3 for RGB)
        """
        super(ConditionalDiscriminator, self).__init__()
        self.num_classes = num_classes
        self.label_dim = label_dim
        self.num_channels = num_channels

        # Label embedding
        self.label_embedding = nn.Embedding(num_classes, label_dim)

        # Convolutional layers for image feature extraction
        self.conv_layers = nn.Sequential(
            # Input: (num_channels, 32, 32)
            # Layer 1: (num_channels, 32, 32) → (32, 16, 16)
            nn.Conv2d(num_channels, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 2: (32, 16, 16) → (64, 8, 8)
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 3: (64, 8, 8) → (128, 4, 4)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 4: (128, 4, 4) → (256, 2, 2)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # FC layers: combine image features and label
        # Input: 256 * 2 * 2 (image features) + label_dim
        self.fc = nn.Sequential(
            nn.Linear(256 * 2 * 2 + label_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Classify image as real/fake (conditioned on label).

        Args:
            x: Image tensor (batch_size, num_channels, 32, 32)
            labels: Class labels (batch_size,) - LongTensor with values 0-9

        Returns:
            Classification probability (batch_size, 1)
        """
        # Extract image features
        features = self.conv_layers(x)  # (batch_size, 256, 2, 2)

        # Flatten image features
        features_flat = features.view(features.size(0), -1)  # (batch_size, 256 * 2 * 2)

        # Embed class labels
        embedded_labels = self.label_embedding(labels)  # (batch_size, label_dim)

        # Concatenate features and label
        combined = torch.cat([features_flat, embedded_labels], dim=1)

        # Classify
        output = self.fc(combined)

        return output


def create_cgan_models(
    latent_dim: int = 100,
    num_classes: int = 10,
    label_dim: int = 50,
    num_channels: int = 3,
    device: torch.device = None,
) -> Tuple[ConditionalGenerator, ConditionalDiscriminator]:
    """
    Helper function to create cGAN generator and discriminator.

    Args:
        latent_dim: Dimension of latent noise vector
        num_classes: Number of classes
        label_dim: Dimension of label embedding
        num_channels: Number of image channels (3 for RGB)
        device: Device to move models to

    Returns:
        Tuple of (generator, discriminator)

    Example:
        >>> gen, disc = create_cgan_models(device=torch.device('cuda'))
        >>> labels = torch.randint(0, 10, (4,), device='cuda')
        >>> z = torch.randn(4, 100, device='cuda')
        >>> images = gen(z, labels)
    """
    if device is None:
        # Auto-detect device
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    generator = ConditionalGenerator(
        latent_dim=latent_dim,
        num_classes=num_classes,
        label_dim=label_dim,
        num_channels=num_channels,
    ).to(device)

    discriminator = ConditionalDiscriminator(
        num_classes=num_classes,
        label_dim=label_dim,
        num_channels=num_channels,
    ).to(device)

    return generator, discriminator


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(
    generator: nn.Module,
    discriminator: nn.Module,
    latent_dim: int = 100,
    num_classes: int = 10,
):
    """Print cGAN model architecture and parameter counts."""
    print("\n" + "=" * 80)
    print("Conditional GAN (cGAN) Model Summary")
    print("=" * 80)

    print("\n🎨 CONDITIONAL GENERATOR (Noise + Label → Image)")
    print("-" * 80)
    print(generator)
    gen_params = count_parameters(generator)
    print(f"\nTotal Parameters: {gen_params:,}")

    print("\n\n🔍 CONDITIONAL DISCRIMINATOR (Image + Label → Classification)")
    print("-" * 80)
    print(discriminator)
    disc_params = count_parameters(discriminator)
    print(f"\nTotal Parameters: {disc_params:,}")

    print("\n" + "=" * 80)
    print(f"Total cGAN Parameters: {gen_params + disc_params:,}")
    print(f"  Generator:     {gen_params:,}")
    print(f"  Discriminator: {disc_params:,}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Create models
    generator, discriminator = create_cgan_models(
        latent_dim=100,
        num_classes=10,
        label_dim=50,
        num_channels=3,
        device=device,
    )

    # Print summary
    print_model_summary(generator, discriminator, latent_dim=100, num_classes=10)

    # Test forward passes with shape verification
    print("\n✓ Shape Verification:")
    print("-" * 80)

    batch_size = 4
    latent_dim = 100
    num_classes = 10

    # Generator forward pass
    z = torch.randn(batch_size, latent_dim, device=device)
    labels = torch.randint(0, num_classes, (batch_size,), device=device)
    fake_images = generator(z, labels)
    print(f"Generator Input (noise):     {z.shape}")
    print(f"Generator Labels:            {labels.shape}")
    print(f"Generator Output (images):   {fake_images.shape}")
    assert fake_images.shape == (
        batch_size,
        3,
        32,
        32,
    ), "Generator output shape mismatch!"

    # Discriminator forward pass
    D_output = discriminator(fake_images, labels)
    print(f"Discriminator Input (images): {fake_images.shape}")
    print(f"Discriminator Labels:        {labels.shape}")
    print(f"Discriminator Output (prob):  {D_output.shape}")
    assert D_output.shape == (batch_size, 1), "Discriminator output shape mismatch!"

    print("\n✓ All shape assertions passed!")
    print("=" * 80)
