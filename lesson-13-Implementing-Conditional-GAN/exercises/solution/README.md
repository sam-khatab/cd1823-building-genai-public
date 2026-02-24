# Solution: Conditional GAN Implementation

---

## Implementation Walkthrough

### TODO 1: Load CIFAR-10 Dataset

**Solution Pattern:**

```python
# Step 1a: Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to [0, 1]
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    # Normalizes to [-1, 1]: (x - 0.5) / 0.5 = 2x - 1
])

# Step 1b: Load dataset
train_dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,  # Auto-download if not present
    transform=transform
)

# Step 1c: Create DataLoader
batch_size = 64
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,  # Critical for SGD
    num_workers=0,  # MPS compatibility
)
```

**Key Points:**
- Normalize to [-1, 1] because Generator uses Tanh (outputs in [-1, 1])
- Shuffling ensures random sampling (helps with convergence)
- num_workers=0 for MPS/Apple Silicon compatibility

**Verification:**
```python
print(f\"Dataset: {len(train_dataset)} images\")
print(f\"Batches: {len(train_loader)}\")
print(f\"Batch size: {batch_size}\")
# Output: Dataset: 50000 images, Batches: 781, Batch size: 64
```

---

### TODO 2: Create cGAN Models

**Solution Pattern:**

```python
latent_dim = 100      # Noise vector size
num_classes = 10      # CIFAR-10 has 10 classes
label_dim = 50        # Embedding dimension for labels

# Factory function creates both
generator, discriminator = create_cgan_models(
    latent_dim=latent_dim,
    num_classes=num_classes,
    label_dim=label_dim,
    num_channels=3,  # RGB
    device=device,
)

# Initialize weights (DCGAN style)
initialize_weights(generator)
initialize_weights(discriminator)

# Verify
print(f\"Generator: {sum(p.numel() for p in generator.parameters()):,} params\")
print(f\"Discriminator: {sum(p.numel() for p in discriminator.parameters()):,} params\")
```

**What's Happening Inside:**

1. **Generator Creation:**
   - Label embedding: 10 classes → 50-dim continuous
   - FC layers: (z + label) → initial feature maps
   - ConvTranspose: Upsampling layers
   - Tanh activation: Output in [-1, 1]

2. **Discriminator Creation:**
   - Conv layers: Downsampling
   - Feature extraction: Image → 1024-dim features
   - Label embedding: Same as Generator
   - FC layers: Concatenate features + embedding → probability
   - Sigmoid activation: Output in [0, 1]

3. **Weight Initialization:**
   - Conv/ConvTranspose: Normal(mean=0, std=0.02)
   - BatchNorm: Normal(mean=1, std=0.02)
   - This is the DCGAN standard
   - Speeds up convergence significantly

---

### TODO 3: Create Trainer

**Solution Pattern:**

```python
trainer = ConditionalGANTrainer(
    generator=generator,
    discriminator=discriminator,
    device=device,
    lr_g=0.0002,    # Generator learning rate
    lr_d=0.0002,    # Discriminator learning rate
    beta1=0.5,      # Adam parameter (not default 0.9)
    beta2=0.999,    # Adam parameter (default)
)
```

**What This Does:**

1. Creates Adam optimizers for both networks
2. Sets up BCE loss function
3. Initializes loss tracking lists
4. Moves models to specified device

**Why These Hyperparameters:**

| Parameter | Value | Reasoning |
|---|---|---|
| lr_g | 0.0002 | DCGAN standard, slower learning = more stable |
| lr_d | 0.0002 | Match G, prevents D overpowering |
| beta1 | 0.5 | DCGAN standard (not default 0.9) |
| beta2 | 0.999 | Standard for GANs, momentum for gradients |

---

### TODO 4: Train the cGAN

**Solution Pattern:**

```python
num_epochs = 20

results = trainer.train(
    train_loader=train_loader,
    num_epochs=num_epochs,
    latent_dim=latent_dim,
    num_classes=num_classes,
    log_interval=50,  # Log every 50 batches
)

print(f\"Training complete after {num_epochs} epochs\")
```

**What Happens Internally:**

```python
# For each epoch:
for epoch in range(num_epochs):
    # For each batch:
    for real_images, real_labels in train_loader:
        # === Discriminator Step ===
        # Want: D(real_img, real_label) → 1.0
        #       D(fake_img, fake_label) → 0.0
        
        # === Generator Step ===
        # Want: D(G(z, target_label), target_label) → 1.0
        # (fool the discriminator)
        
        # Update both networks
```

**Expected Behavior:**

- Loss decreases initially
- D loss stabilizes around 0.5 (random guessing)
- G loss continues decreasing
- Progress printed every 50 batches

**Typical Output:**
```
Epoch 1/20:
  Batch 50: D Loss = 0.8532, G Loss = 2.1234
  Batch 100: D Loss = 0.6234, G Loss = 1.5678
  ...
  Epoch Loss: D = 0.6123, G = 1.2345

Epoch 2/20:
  ...
```

---

### TODO 5: Plot Loss Curves

**How to Interpret:**

**Good Training (What We Want):**
```
D Loss Graph:
- Starts high (~0.7)
- Decreases to ~0.5
- Stays around 0.5 (random guessing)
- Maybe oscillates slightly

G Loss Graph:
- Starts very high (~2.0)
- Steadily decreases
- After epoch 5+: maybe increases again
- Generally trending downward
```

**Bad Training (What We Don't Want):**
```
D Loss → 0: Discriminator overpowering
  Solution: Reduce D learning rate

Both losses → ∞: Exploding gradients
  Solution: Use smaller learning rates

D Loss oscillating wildly: Instability
  Solution: Add batch norm, reduce LR
```

