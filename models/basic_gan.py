"""
Basic GAN Models - Foundation for Generative Adversarial Networks

This module contains the simple generator and discriminator implementations
used in Lessons 3-4 to introduce GAN fundamentals.

Architecture:
- Generator: Fully-connected network (noise → 28×28 image)
- Discriminator: Fully-connected network (image → real/fake probability)
- Dataset: MNIST (28×28 grayscale images)

Key Concepts:
- Generator learns to map random noise to realistic images
- Discriminator learns to distinguish real from fake images
- Training is adversarial: Generator vs Discriminator in competition
- Output: BasicGenerator, SimpleDiscriminator, create_generator(), create_discriminator()

References:
- Lesson 3: Building a Simple Generator Network
- Lesson 5: Building a Discriminator
"""

import torch
import torch.nn as nn


class BasicGenerator(nn.Module):
    """
    A simple generator network for MNIST.

    Takes a random noise vector and progressively upsamples it through
    fully-connected layers to produce a 28x28 grayscale image.

    Architecture:
        Input: noise vector of size latent_dim (typically 100)
        → Dense layers with LeakyReLU activations
        → Output: 28x28 image (flattened to 784 values)
        → Reshape to (1, 28, 28)

    Why LeakyReLU? Standard ReLU can suffer from "dying ReLU" problem during
    training where neurons permanently output zero. LeakyReLU allows a small
    negative gradient to flow through, preventing this.

    Why Tanh output? We want pixel values in [-1, 1]. Tanh naturally outputs
    in this range, which is easier for the discriminator to work with than [0, 1].
    """

    def __init__(self, latent_dim=100, hidden_dim=256):
        """
        Args:
            latent_dim: Size of the input noise vector (default: 100)
            hidden_dim: Size of hidden layers (default: 256)
        """
        super(BasicGenerator, self).__init__()

        # We'll build a 3-layer network:
        # noise (100) -> hidden (256) -> hidden (256) -> output (784)

        self.latent_dim = latent_dim

        self.network = nn.Sequential(
            # First dense layer: noise_dim -> hidden_dim
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),  # Small negative slope for x < 0
            # Second dense layer: hidden_dim -> hidden_dim
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            # Output layer: hidden_dim -> 784 (28*28 for MNIST)
            nn.Linear(hidden_dim, 784),
            nn.Tanh(),  # Map to [-1, 1] for pixel values
        )

    def forward(self, z):
        """
        Generate an image from a noise vector.

        Args:
            z: Tensor of shape (batch_size, latent_dim) containing noise vectors

        Returns:
            img: Tensor of shape (batch_size, 784) containing flattened images

        Note: The output is flattened. You'll reshape it to (batch_size, 1, 28, 28)
              for visualization or to feed into a discriminator.
        """
        return self.network(z)


class SimpleDiscriminator(nn.Module):
    """
    A simple discriminator for binary classification (real vs fake) on MNIST.

    This is the foundation. In the project, we'll extend it to be conditional
    (conditioned on class labels), but the core logic stays the same.

    Architecture:
        Input: flattened image (784 dimensions for 28x28)
        → Hidden layer (512) with LeakyReLU
        → Hidden layer (256) with LeakyReLU
        → Output (1) with Sigmoid → probability between 0 and 1

    Training Strategy (adversarial):
        - Real images → should output ~1 (high probability of being real)
        - Fake images → should output ~0 (high probability of being fake)

    Why LeakyReLU?
        Standard ReLU can cause "dying ReLU" problem in GANs where gradients
        become zero and neurons stop learning. LeakyReLU allows a small
        negative gradient (negative_slope=0.2) to flow through, keeping training stable.

    Why Sigmoid?
        We want probabilities in [0, 1]. Sigmoid naturally maps any input to this range.
        This matches Binary Cross-Entropy loss expectations.
    """

    def __init__(self, img_dim=784, hidden_dim=512):
        """
        Args:
            img_dim: Size of flattened image (28*28 = 784 for MNIST)
            hidden_dim: Size of hidden layers
        """
        super(SimpleDiscriminator, self).__init__()

        self.network = nn.Sequential(
            # First hidden layer
            nn.Linear(img_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            # Second hidden layer
            nn.Linear(hidden_dim, hidden_dim // 2),  # Reduce dimensionality
            nn.LeakyReLU(0.2, inplace=True),
            # Output layer: single probability
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # Map to [0, 1]
        )

    def forward(self, x):
        """
        Classify an image as real (1) or fake (0).

        Args:
            x: Image tensor of shape:
               - (batch_size, 1, 28, 28) → will be flattened to (batch_size, 784)
               - OR (batch_size, 784) → already flattened

        Returns:
            prob: Probability tensor of shape (batch_size, 1)
                  Values close to 1 → model thinks image is real
                  Values close to 0 → model thinks image is fake
        """
        # Flatten if needed (handles both 4D and 2D inputs)
        if x.dim() == 4:
            x = x.view(x.size(0), -1)  # Flatten to (batch_size, 784)

        # Pass through network
        return self.network(x)


def create_generator(latent_dim=100, device="cpu"):
    """
    Convenience function to create and initialize a generator.

    Args:
        latent_dim: Size of noise vector
        device: 'cpu' or 'cuda'

    Returns:
        generator: BasicGenerator instance on the specified device
    """
    generator = BasicGenerator(latent_dim=latent_dim)
    generator = generator.to(device)
    return generator


def create_discriminator(img_dim=784, device="cpu"):
    """
    Convenience function to create and initialize a discriminator.

    Args:
        img_dim: Size of flattened image
        device: 'cpu' or 'cuda'

    Returns:
        discriminator: SimpleDiscriminator instance on the specified device
    """
    discriminator = SimpleDiscriminator(img_dim=img_dim)
    discriminator = discriminator.to(device)
    return discriminator


if __name__ == "__main__":
    # Test generator
    print("Testing BasicGenerator...")
    generator = create_generator(latent_dim=100, device="cpu")

    z = torch.randn(4, 100)
    fake_images = generator(z)

    print(f"Input noise shape: {z.shape}")
    print(f"Generated images shape: {fake_images.shape}")
    print(
        f"Image pixel range: [{fake_images.min().item():.3f}, {fake_images.max().item():.3f}]"
    )
    print("✓ Generator working correctly!\n")

    # Test discriminator
    print("Testing SimpleDiscriminator...")
    device = "cpu"
    discriminator = create_discriminator(device=device)
    discriminator.eval()

    fake_images = torch.randn(4, 1, 28, 28).to(device)

    with torch.no_grad():
        predictions = discriminator(fake_images)

    print(f"Input shape: {fake_images.shape}")
    print(f"Output shape: {predictions.shape}")
    print(
        f"Predictions range: [{predictions.min().item():.3f}, {predictions.max().item():.3f}]"
    )
    print("✓ Discriminator working correctly!")
