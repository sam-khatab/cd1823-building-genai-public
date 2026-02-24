"""
Centralized Models Repository for Generative Deep Learning Course

This module aggregates all model definitions from individual lessons,
enabling students to import from a single location without depending on
previous lessons.

Available Models:
- basic_gan: BasicGenerator, SimpleDiscriminator (Lessons 3-4)
- dcgan: DCGANGenerator, DCGANDiscriminator (Lesson 11)
- conditional_gan: ConditionalGenerator, ConditionalDiscriminator (Lesson 13)
- diffusion: SimpleUNet, NoiseScheduler, add_noise (Lessons 16-17)

Usage Examples:

    # Import basic GAN components
    from models.basic_gan import create_generator, create_discriminator

    # Import DCGAN components
    from models.dcgan import create_dcgan_models

    # Import conditional GAN components
    from models.conditional_gan import create_cgan_models

    # Import diffusion components
    from models.diffusion import create_diffusion_model, add_noise

This centralized approach ensures:
1. Students can work independently without accessing previous lessons
2. Models are maintained in one location for consistency
3. Easy updates propagate to all exercises
4. Clear import paths for all model definitions
"""

# Import for convenient access
from .basic_gan import (
    BasicGenerator,
    SimpleDiscriminator,
    create_discriminator,
    create_generator,
)
from .cgan import (
    ConditionalDiscriminator,
    ConditionalGenerator,
    create_cgan_models,
)
from .dcgan import DCGANDiscriminator, DCGANGenerator, create_dcgan_models
from .diffusion import (
    AttentionBlock,
    NoiseScheduler,
    ResidualBlock,
    SimpleUNet,
    TimeEmbedding,
    add_noise,
    create_diffusion_model,
)

__all__ = [
    # Basic GAN
    "BasicGenerator",
    "SimpleDiscriminator",
    "create_generator",
    "create_discriminator",
    # DCGAN
    "DCGANGenerator",
    "DCGANDiscriminator",
    "create_dcgan_models",
    # Conditional GAN
    "ConditionalGenerator",
    "ConditionalDiscriminator",
    "create_cgan_models",
    # Diffusion
    "SimpleUNet",
    "NoiseScheduler",
    "add_noise",
    "create_diffusion_model",
    "TimeEmbedding",
    "ResidualBlock",
    "AttentionBlock",
]
