# Simple MLP GAN for MNIST Generation 🎨

A **PyTorch implementation** of a **Generative Adversarial Network (GAN)** built using **Multi-Layer Perceptrons (MLP)** to generate handwritten digits similar to those in the **MNIST dataset**.  
This project demonstrates the fundamental working of GANs — from training the generator and discriminator to evaluating sample quality — in a clean and minimal design.

---

## 🧠 What is a GAN?

A **Generative Adversarial Network (GAN)** consists of two neural networks — a **Generator** and a **Discriminator** — that compete against each other in a zero-sum game:

- The **Generator** learns to produce fake images that resemble real data.  
- The **Discriminator** learns to distinguish real images from generated (fake) ones.

Through this adversarial training, the generator improves its ability to create realistic samples.

---

## 🧩 What is an MLP GAN?

Unlike CNN-based GANs that use convolutional layers for image synthesis, an **MLP GAN** relies entirely on **fully connected (dense) layers**.  
This makes it simpler and easier to understand for beginners, especially for small datasets like **MNIST (28×28 grayscale images)**.

The MLP GAN used here is composed of:
- **Generator**: A stack of linear layers with ReLU/Tanh activations that map noise vectors (latent space) to image pixels.  
- **Discriminator**: A similar network that classifies inputs as *real* or *fake* using a sigmoid output.

---


## ⚙️ Key Functions and Their Roles

### `load_mnist_dataset()`
Loads and preprocesses the **MNIST dataset** from `torchvision.datasets`, normalizing pixel values and preparing PyTorch dataloaders for training.

---

### `train_generator(generator, discriminator, optimizer, loss_function, batch_size, noise_dim)`
Trains the **Generator** network for one step.  
It generates fake images from random noise, passes them through the discriminator, computes the generator’s loss, and updates its weights.

---

### `train_discriminator(generator, discriminator, real_images, optimizer, loss_function, batch_size, noise_dim)`
Trains the **Discriminator** to differentiate between real and fake samples.  
It uses both real MNIST images and generated images to compute a binary classification loss and updates the discriminator’s parameters accordingly.

---

### `save_generated_samples(generator, epoch, noise_dim)`
Generates and saves synthetic images at specific epochs.  
Helps visualize how the generator improves over time.

---

### `overfit_single_batch_gan()`
**Testing function:**  
Attempts to **overfit a single batch** of MNIST images to verify if the GAN setup and architecture are capable of learning.  
If the generator and discriminator cannot overfit one batch, they are unlikely to work effectively on the full dataset.

---

### `train_gan(epochs, generator, discriminator, dataloader, optimizer_g, optimizer_d, loss_function, noise_dim)`
Main training loop that orchestrates the training of both networks over multiple epochs.  
Displays loss curves and periodically generates samples for visual inspection.

---

### `visualize_single_batch()`
Displays a batch of real MNIST images for quick visualization and comparison with generated outputs.

---

### `main()`
The entry point of the script — loads data, initializes models, sets optimizers, and starts the training process.  
It also ensures reproducibility and proper logging of training progress.

---

## 🧮 Model Overview

| Component | Type | Description |
|------------|------|-------------|
| **Generator** | MLP | Maps random noise → fake MNIST-like images |
| **Discriminator** | MLP | Classifies real vs fake images |
| **Loss Function** | Binary Cross Entropy (BCE) | Used by both generator and discriminator |
| **Optimizer** | Adam | Adaptive optimization for both networks |
| **Dataset** | MNIST (28×28 grayscale digits) | Standard benchmark for generative tasks |

---

## 📊 Results

- Generates realistic MNIST-style digits after sufficient training.  
- Overfitting test helps ensure network correctness before full-scale training.  
- Saved samples show progressive improvement across epochs.
