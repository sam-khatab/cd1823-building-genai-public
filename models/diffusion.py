"""
Diffusion Model Implementation (DDPM)

This module contains the diffusion model components used in Lessons 16-17
to implement conditional diffusion models for image generation.

Architecture:
- NoiseScheduler: Defines forward diffusion process (clean image → noise)
- SimpleUNet: U-Net architecture for reverse diffusion (noise → clean image)
- Supporting components: TimeEmbedding, ResidualBlock, AttentionBlock

Key Concepts:
- Forward Diffusion: Gradually add Gaussian noise to images
- Reverse Diffusion: Train network to predict and remove noise step-by-step
- Denoising: Iteratively denoise from pure noise to generate new images
- U-Net: Encoder-decoder with skip connections for noise prediction

Applications:
- Image generation from pure noise
- Conditional generation (with class labels)
- Superior sample quality compared to GANs
- More stable training dynamics

References:
- Lesson 16: Implementing Simple Diffusion Model
- Lesson 17: Sampling and Image Generation with Diffusion
- Paper: Ho et al., "Denoising Diffusion Probabilistic Models" (DDPM, 2020)

Dataset: MNIST (28×28 grayscale images)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# PART 1: NOISE SCHEDULER
# ============================================================================


class NoiseScheduler:
    """
    Fixed variance schedule for forward diffusion process.

    Key Parameters:
    - num_timesteps: Number of diffusion steps (T)
    - beta_start: Initial noise variance
    - beta_end: Final noise variance
    - schedule_type: 'linear' or 'cosine'

    The scheduler defines how much noise to add at each timestep.
    Critical for training stability and generation quality.

    Mathematical Foundation:
    - Forward diffusion: x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε
    - ᾱ_t: Cumulative product of (1 - β_t)
    - β_t: Noise variance at timestep t
    - sqrt(ᾱ_t): Decays from 1 to 0 (signal → noise)
    - sqrt(1 - ᾱ_t): Grows from 0 to 1 (noise → signal)
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule_type: str = "linear",
    ):
        """
        Initialize noise scheduler with variance schedule.

        Args:
            num_timesteps: T, number of diffusion steps
            beta_start: Initial noise variance β_start
            beta_end: Final noise variance β_T
            schedule_type: 'linear' or 'cosine'

        Example:
            scheduler = NoiseScheduler(num_timesteps=1000)
            # Returns schedules for all 1000 timesteps
        """
        self.num_timesteps = num_timesteps

        if schedule_type == "linear":
            # Linear schedule: β_t = β_start + t * (β_end - β_start) / T
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == "cosine":
            # Cosine schedule: Better empirical results
            # Based on OpenAI's improved DDPM
            s = 0.008
            steps = torch.arange(num_timesteps + 1)
            alphas_cumprod = (
                torch.cos(((steps / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            )
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            self.betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(self.betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        # Pre-compute useful quantities to avoid repeated computation
        # These are used in forward diffusion and reverse process

        # α_t = 1 - β_t
        self.alphas = 1.0 - self.betas

        # ᾱ_t = ∏_{s=1}^{t} α_s (cumulative product of alphas)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # ᾱ_{t-1} = cumulative product up to t-1
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])

        # Useful quantities for forward process
        # sqrt(ᾱ_t): Coefficient for original image
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)

        # sqrt(1 - ᾱ_t): Coefficient for noise
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Posterior variance (for reverse process)
        # β̃_t = (1 - ᾱ_{t-1}) / (1 - ᾱ_t) * β_t
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def get_coefficients(
        self, timestep: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get forward diffusion coefficients for a given timestep.

        Args:
            timestep: Tensor of shape (batch_size,) with timestep indices

        Returns:
            sqrt_alphas_cumprod: Coefficient for original image x_0
            sqrt_one_minus_alphas_cumprod: Coefficient for noise ε

        Forward diffusion formula:
            x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε

        where ε ~ N(0, I)
        """
        # Move coefficients to same device as timestep
        device = timestep.device
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)[timestep]
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)[
            timestep
        ]

        # Reshape for broadcasting with image batch
        # Shape: (batch_size, 1, 1, 1) for MNIST (1, 28, 28)
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.reshape(
            -1, 1, 1, 1
        )

        return sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod


# ============================================================================
# PART 2: U-NET ARCHITECTURE FOR NOISE PREDICTION
# ============================================================================


class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding for timestep conditioning.

    Converts integer timestep t into a continuous embedding.
    Based on Transformer positional encodings.

    Key idea: Sine/cosine at different frequencies capture t's position.
    """

    def __init__(self, embedding_dim: int = 128):
        """
        Args:
            embedding_dim: Dimension of embedding (typically 128)
        """
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert timestep t to embedding.

        Args:
            timestep: Tensor of shape (batch_size,)

        Returns:
            embedding: Tensor of shape (batch_size, embedding_dim)
        """
        device = timestep.device
        half_dim = self.embedding_dim // 2

        # Frequency schedule: 10000^(2i/d) for i in [0, d/2)
        # This creates different frequencies to encode position
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=device) / half_dim
        )

        # Multiply timestep by frequencies
        args = timestep[:, None].float() * freqs[None, :]

        # Use sine and cosine (like transformer positional encoding)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        return embedding


class ResidualBlock(nn.Module):
    """
    Residual block with timestep conditioning.

    Structure:
    - Conv2D (groups=channels for depth-wise)
    - Time embedding MLP conditioning
    - GroupNorm + ReLU + Conv
    - Skip connection

    Key feature: Timestep modulates the convolution via FiLM (Feature-wise Linear Modulation)
    """

    def __init__(
        self, in_channels: int, out_channels: int, time_embedding_dim: int = 128
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            time_embedding_dim: Dimension of time embedding
        """
        super().__init__()

        # Main path
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Time conditioning via FiLM (Feature-wise Linear Modulation)
        # Maps time embedding to scale and shift for output features
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, out_channels),
            nn.SiLU(),  # Sigmoid Linear Unit
            nn.Linear(out_channels, out_channels),
        )

        # Skip connection (1x1 conv if dimensions don't match)
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with time conditioning.

        Args:
            x: Input tensor (batch, in_channels, height, width)
            time_embedding: Time embedding (batch, time_embedding_dim)

        Returns:
            Output tensor (batch, out_channels, height, width)
        """
        # Residual path
        h = self.norm1(x)
        h = F.silu(h)  # Sigmoid Linear Unit
        h = self.conv1(h)

        # FiLM conditioning: Modulate by time
        # time_mlp outputs (batch, out_channels)
        time_scale_shift = self.time_mlp(time_embedding)
        # Reshape for broadcasting: (batch, out_channels) -> (batch, out_channels, 1, 1)
        time_scale_shift = time_scale_shift[:, :, None, None]
        h = h * time_scale_shift  # Scale features by time

        # Second convolution
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        # Skip connection
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """
    Multi-head self-attention block for modeling long-range dependencies.

    Used in later layers of U-Net to capture global patterns.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        """
        Args:
            channels: Number of channels
            num_heads: Number of attention heads
        """
        super().__init__()

        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.mha = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention.

        Args:
            x: Input (batch, channels, height, width)

        Returns:
            Output (batch, channels, height, width)
        """
        batch, channels, height, width = x.shape

        # Normalize
        h = self.norm(x)

        # Reshape to sequence: (batch, height*width, channels)
        h = h.reshape(batch, channels, -1).permute(0, 2, 1)

        # Self-attention
        h, _ = self.mha(h, h, h)

        # Reshape back
        h = h.permute(0, 2, 1).reshape(batch, channels, height, width)

        return x + h  # Skip connection


class SimpleUNet(nn.Module):
    """
    Simplified U-Net for MNIST noise prediction.

    Architecture:
    - Encoder: Conv downsampling
    - Middle: Residual blocks with attention
    - Decoder: ConvTranspose upsampling (symmetric to encoder)
    - Time conditioning: Applied at each residual block

    Purpose: Predict noise ε given (x_t, t)
    Output: Same shape as input x_t, represents predicted noise
    """

    def __init__(
        self,
        image_channels: int = 1,
        base_channels: int = 64,
        time_embedding_dim: int = 128,
        num_residual_blocks: int = 2,
    ):
        """
        Args:
            image_channels: 1 for MNIST (grayscale)
            base_channels: Base number of filters (64)
            time_embedding_dim: Dimension of time embedding (128)
            num_residual_blocks: Residual blocks per resolution (2)
        """
        super().__init__()

        self.image_channels = image_channels
        self.base_channels = base_channels
        self.time_embedding_dim = time_embedding_dim
        self.num_residual_blocks = num_residual_blocks

        # Time embedding
        self.time_embedding = TimeEmbedding(time_embedding_dim)

        # ========== ENCODER (Downsampling) ==========
        # Input: (batch, 1, 28, 28)

        # Initial conv: 1 channel -> base_channels
        self.init_conv = nn.Conv2d(
            image_channels, base_channels, kernel_size=3, padding=1
        )

        # Downsampling blocks with increasing channels
        # 28x28 -> 14x14 -> 7x7 (2 levels for MNIST)
        self.down_blocks = nn.ModuleList()

        in_channels = base_channels
        for level in range(2):  # 2 downsampling levels
            out_channels = base_channels * (2**level)

            # Residual blocks at current resolution
            for _ in range(num_residual_blocks):
                self.down_blocks.append(
                    ResidualBlock(in_channels, out_channels, time_embedding_dim)
                )
                in_channels = out_channels

            # Downsample (except last level)
            if level < 1:
                self.down_blocks.append(
                    nn.Conv2d(
                        in_channels, in_channels, kernel_size=4, stride=2, padding=1
                    )
                )

        # ========== MIDDLE (Bottleneck) ==========
        # At 7x7 resolution with base_channels*2 channels
        self.middle_blocks = nn.ModuleList()
        for _ in range(num_residual_blocks):
            self.middle_blocks.append(
                ResidualBlock(in_channels, in_channels, time_embedding_dim)
            )

        # Attention at bottleneck
        self.middle_attention = AttentionBlock(in_channels)

        # ========== DECODER (Upsampling) ==========
        # Mirror the encoder structure
        self.up_blocks = nn.ModuleList()

        for level in range(1, -1, -1):  # Reverse: 7x7 -> 14x14 -> 28x28
            out_channels = base_channels * (2**level)

            # Upsample (except first level)
            if level < 1:
                self.up_blocks.append(
                    nn.ConvTranspose2d(
                        in_channels, in_channels, kernel_size=4, stride=2, padding=1
                    )
                )

            # Residual blocks at current resolution
            for _ in range(num_residual_blocks):
                self.up_blocks.append(
                    ResidualBlock(in_channels, out_channels, time_embedding_dim)
                )
                in_channels = out_channels

        # ========== OUTPUT ==========
        # Final normalization and conv
        self.final_norm = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.final_conv = nn.Conv2d(
            in_channels, image_channels, kernel_size=3, padding=1
        )

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Predict noise from noisy image and timestep.

        Args:
            x: Noisy image (batch, 1, 28, 28)
            timestep: Timestep indices (batch,)

        Returns:
            Predicted noise (batch, 1, 28, 28)
        """
        # Create time embedding
        time_emb = self.time_embedding(timestep)  # (batch, time_embedding_dim)

        # Initial convolution
        h = self.init_conv(x)  # (batch, base_channels, 28, 28)

        # Store activations for skip connections
        skip_connections = [h]

        # Encoder (downsampling)
        for block in self.down_blocks:
            if isinstance(block, ResidualBlock):
                h = block(h, time_emb)
            else:  # Conv downsampling
                h = block(h)
            skip_connections.append(h)

        # Middle (bottleneck)
        for block in self.middle_blocks:
            h = block(h, time_emb)
        h = self.middle_attention(h)

        # Decoder (upsampling)
        for block in self.up_blocks:
            if isinstance(block, ResidualBlock):
                h = block(h, time_emb)
            else:  # ConvTranspose upsampling
                h = block(h)

        # Final output
        h = self.final_norm(h)
        h = F.silu(h)
        output = self.final_conv(h)  # (batch, 1, 28, 28)

        return output


# ============================================================================
# PART 3: FORWARD DIFFUSION PROCESS
# ============================================================================


def add_noise(
    x_0: torch.Tensor,
    timestep: torch.Tensor,
    scheduler: NoiseScheduler,
    noise: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward diffusion: Add noise to image at timestep t.

    Mathematical formula:
        x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε

    where:
    - x_0: Original image
    - ε: Standard Gaussian noise N(0, I)
    - ᾱ_t: Cumulative product of alphas (determines noise amount)
    - sqrt(ᾱ_t): Weight on original image (decreases with t)
    - sqrt(1 - ᾱ_t): Weight on noise (increases with t)

    Args:
        x_0: Original images (batch, channels, height, width)
        timestep: Timestep indices (batch,) from 0 to T-1
        scheduler: NoiseScheduler instance
        noise: Optional pre-sampled noise (for reproducibility)

    Returns:
        x_t: Noisy image at timestep t
        noise: The noise that was added (used for training)

    Example:
        scheduler = NoiseScheduler(num_timesteps=1000)
        t = torch.randint(0, 1000, (batch_size,))
        x_t, noise = add_noise(x_0, t, scheduler)
        # x_t: Noisy version of x_0
        # noise: Ground truth noise for this timestep
    """
    # Sample random noise if not provided
    if noise is None:
        noise = torch.randn_like(x_0)

    # Get coefficients for this timestep
    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = scheduler.get_coefficients(
        timestep
    )

    # Forward diffusion formula
    x_t = (sqrt_alphas_cumprod * x_0) + (sqrt_one_minus_alphas_cumprod * noise)

    return x_t, noise


# ============================================================================
# PART 4: UTILITY FUNCTIONS
# ============================================================================


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module, name: str = "Model"):
    """Print model architecture and parameter count."""
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(model)
    print(f"Total Parameters: {count_parameters(model):,}")
    print(f"{'='*60}\n")


# ============================================================================
# PART 5: INITIALIZATION FUNCTIONS
# ============================================================================


def initialize_weights(model: nn.Module):
    """
    Initialize model weights using Kaiming normal initialization.

    This is different from DCGAN's normal initialization.
    Works better for U-Net architectures.
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


# ============================================================================
# PART 6: MODEL CREATION HELPER
# ============================================================================


def create_diffusion_model(
    image_channels: int = 1,
    base_channels: int = 64,
    time_embedding_dim: int = 128,
    num_timesteps: int = 1000,
    device: str = "cpu",
) -> Tuple[nn.Module, NoiseScheduler]:
    """
    Create and initialize diffusion model with scheduler.

    Args:
        image_channels: 1 for MNIST (grayscale)
        base_channels: Base filter count (64)
        time_embedding_dim: Time embedding dimension (128)
        num_timesteps: Number of diffusion steps (1000)
        device: 'cpu', 'cuda', or 'mps'

    Returns:
        model: SimpleUNet noise prediction model
        scheduler: NoiseScheduler for forward diffusion

    Example:
        model, scheduler = create_diffusion_model(device='cuda')
        print(f"Parameters: {count_parameters(model):,}")
    """
    # Create scheduler
    scheduler = NoiseScheduler(
        num_timesteps=num_timesteps,
        beta_start=0.0001,
        beta_end=0.02,
        schedule_type="linear",
    )

    # Create model
    model = SimpleUNet(
        image_channels=image_channels,
        base_channels=base_channels,
        time_embedding_dim=time_embedding_dim,
    )

    # Initialize weights
    initialize_weights(model)

    # Move to device
    model = model.to(device)

    return model, scheduler


if __name__ == "__main__":
    """Test script for diffusion model."""

    print("Testing Diffusion Model Components...\n")

    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}\n")

    # Create model and scheduler
    print("Creating model and scheduler...")
    model, scheduler = create_diffusion_model(
        image_channels=1,
        base_channels=64,
        time_embedding_dim=128,
        num_timesteps=1000,
        device=str(device),
    )

    print_model_summary(model, "SimpleUNet for MNIST Denoising")

    # Test forward pass
    print("Testing forward pass...")
    batch_size = 4
    x_0 = torch.randn(batch_size, 1, 28, 28, device=device)
    t = torch.randint(0, 1000, (batch_size,), device=device)

    # Add noise
    x_t, noise = add_noise(x_0, t, scheduler)
    print(f"x_0 (original) shape: {x_0.shape}")
    print(f"x_t (noisy) shape: {x_t.shape}")
    print(f"noise shape: {noise.shape}")

    # Predict noise
    predicted_noise = model(x_t, t)
    print(f"predicted_noise shape: {predicted_noise.shape}")

    # Verify shapes
    assert predicted_noise.shape == x_t.shape, "Output shape mismatch!"
    print("✓ All shapes correct!")

    # Test MSE loss (main training loss)
    print("\nTesting MSE loss...")
    mse_loss = F.mse_loss(predicted_noise, noise)
    print(f"MSE Loss: {mse_loss.item():.6f}")
    print(f"✓ MSE Loss computation works!")

    print("\n✓ All tests passed!")
