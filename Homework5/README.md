# CS5720: Neural Networks and Deep Learning

### Home Assignment 5 - Spring 2025

Student Name:  Ravi Teja Reddy Nomula

---

## 1. GAN Architecture

**Generative Adversarial Networks (GANs)** involve two neural networks — a Generator and a Discriminator — that compete in a zero-sum game.

- **Generator (G)**: Learns to create data resembling the training distribution from random noise.
- **Discriminator (D)**: Learns to distinguish between real data and fake data produced by the Generator.

### Training Objective:

- The Generator improves by **fooling** the Discriminator.
- The Discriminator improves by **catching** the Generator’s fakes.

### Architecture Diagram:

*The diagram file is included in the project directory as ****`gan_architecture.png`****.*

---

## 2. Ethics and AI Harm

**Chosen Harm**: *Misinformation in Generative AI*

### Example:

A generative AI tool creates realistic-looking fake news articles or deepfakes, leading to widespread misinformation.

### Mitigation Strategies:

1. **Content Provenance**: Tag AI-generated content with metadata to trace its source.
2. **Safety Alignment**: Fine-tune AI models to avoid producing harmful or misleading content.

---

## 3. Programming Task – GAN (TensorFlow)

### Description:

Implemented a simple GAN using TensorFlow and Keras to generate handwritten digits from the MNIST dataset.

### Features:

- Generator and Discriminator networks
- Training loop with alternating updates
- Outputs sample images at epochs 0, 50, 100
- Plots Generator and Discriminator loss over time

### Output Files:

- `gan_mnist_epoch_0.png`, `gan_mnist_epoch_50.png`, `gan_mnist_epoch_100.png`
- `gan_loss_plot.png`

---

## 4. Programming Task – Data Poisoning Simulation

### Description:

Simulated a data poisoning attack on a sentiment analysis model trained on IMDB reviews.

### Process:

- Trained a binary classifier on clean IMDB data
- Flipped labels for 500 samples to simulate poisoning
- Re-trained and compared model performance

### Output:

- `accuracy_comparison.png`
- `confusion_matrix_clean.png`
- `confusion_matrix_poisoned.png`

### Impact:

The poisoned model showed decreased validation accuracy and a more confused classification boundary in the confusion matrix.

---

## 5. Legal and Ethical Implications of GenAI

### Issues:

- **Privacy**: Models like GPT-2 may memorize and leak personal data.
- **Copyright**: Generated text may replicate copyrighted material (e.g., Harry Potter excerpts).

### Opinion:

Yes, models **should be restricted** from certain datasets.

- Ensures privacy compliance
- Reduces legal risk
- Promotes ethical AI development

---

## 6. Bias & Fairness Tools

### Tool: [Aequitas Bias Audit Tool](http://aequitas.dssg.io/)

### Metric: **False Negative Rate Parity**

- **Definition**: Measures if one group receives more false negatives than another.
- **Importance**: High disparity can result in underserved populations being wrongly rejected (e.g., in loan approvals).
- **Failure Example**: A model disproportionately denying loans to a racial minority while accepting similar profiles from others.

---

**Repository Contents**:

- `gan_mnist_tensorflow.py`
- `data_poisoning_sentiment.py`
- Images: `.png` outputs
- README: This file

---

*End of README*

