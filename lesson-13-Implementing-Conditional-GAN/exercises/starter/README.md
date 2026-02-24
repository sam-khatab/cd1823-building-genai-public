# Exercise: Implementing Conditional GANs



## Learning Objectives

 **Understand Conditioning:** How to add class information to neural networks

 **Implement cGAN Models:** Build Generator and Discriminator with class control

 **Train Conditional Models:** Implement training loop with class labels

 **Analyze Results:** Interpret 10×10 class grids to evaluate quality

 **Apply to Augmentation:** Use generated images for data augmentation

---

## TODO Structure

Both files have **10 TODO sections** organized by difficulty:

### Beginner TODOs (1-3): Setup

**TODO 1: Load CIFAR-10 Dataset**
```python
# What to do:
# - Define transforms (ToTensor + Normalize to [-1, 1])
# - Load CIFAR-10 with download=True
# - Create DataLoader with batch_size=64

# Why it matters:
# - Normalization affects training stability
# - Proper dataloader ensures batch consistency
# - Shuffling enables stochastic gradient descent
```

**TODO 2: Create cGAN Models**
```python
# What to do:
# - Use create_cgan_models() function
# - Set latent_dim=100 (noise vector)
# - Set num_classes=10 (CIFAR-10)
# - Initialize weights using DCGAN guidelines

# Why it matters:
# - Proper weight initialization speeds up convergence
# - Correct dimensions ensure clean tensor operations
```

**TODO 3: Create Trainer**
```python
# What to do:
# - Instantiate ConditionalGANTrainer
# - Pass generator and discriminator
# - Set learning rates to 0.0002
# - Configure Adam optimizer with beta1=0.5, beta2=0.999

# Why it matters:
# - Trainer manages loss computation and updates
# - These hyperparameters are DCGAN standards
```

### Intermediate TODOs (4-6): Training and Visualization

**TODO 4: Train the cGAN**
```python
# What to do:
# - Call trainer.train() for 20 epochs
# - Pass train_loader, latent_dim, num_classes
# - Let it run (will take 20-30 minutes)

# What to expect:
# - D loss should stabilize around 0.5-0.7
# - G loss should decrease over time
# - Loss curves show training progress

# Common questions:
# Q: Why so long?
# A: Training GANs requires many iterations for convergence

# Q: Can I interrupt?
# A: Yes, you can stop early and see partial results
```

**TODO 5: Plot Loss Curves**
```python
# What to do:
# - Create 1x2 subplot (D loss, G loss)
# - Plot results['d_losses'] on left
# - Plot results['g_losses'] on right
# - Add reference line at y=0.5 (ideal D loss)

# What to look for:
# ✓ D loss: Should be around 0.5 (random guessing)
# ✓ G loss: Should decrease over time
# ✗ Both losses high: Training instability
# ✗ D loss → 0: D overpowering G
```

**TODO 6: Generate 10×10 Class Grid**
```python
# What to do:
# - Use trainer.generate_all_classes_grid()
# - Generate 10 classes × 10 samples = 100 images
# - Same noise z, different class labels y
# - Denormalize from [-1, 1] to [0, 1]
# - Display as 10×10 subplot grid

# What to look for:
# ✓ Each row is clearly a different class
# ✓ Within rows: style variation visible
# ✓ Across rows: clear separation
# ✗ All rows look similar: class control failing
# ✗ Blurry/noisy: model underfitted

# This is the MOST IMPORTANT visualization!
```

### Advanced TODOs (7-8): Analysis

**TODO 7: Generate Single-Class Samples**
```python
# What to do:
# - Call trainer.generate_class_samples() for target_class=5 (dogs)
# - Generate 16 samples with different noise
# - Denormalize and display as 2×8 grid

# Key insight:
# - Same class, different noise = style variation
# - Shows the noise dimension creates diversity
# - All 16 should look like dogs, different appearances
```


