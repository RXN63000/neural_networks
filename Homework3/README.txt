# Deep Learning Experiments with Keras

This repository contains four deep learning experiments implemented using Keras and TensorFlow:

## 1. Autoencoder for Image Reconstruction

- **Description:** This experiment demonstrates the implementation of a fully connected autoencoder for image reconstruction using the MNIST dataset.
- **Functionality:**
    - Loads and preprocesses the MNIST dataset.
    - Defines a fully connected autoencoder architecture with an encoder and a decoder.
    - Compiles and trains the autoencoder with binary cross-entropy loss.
    - Visualizes the original and reconstructed images for comparison.
    - Allows experimentation with different latent dimension sizes.
- **File:** `autoencoder.py`
- **Answers to Questions:**
    - **How to change the latent dimension size?** Modify the `encoded` layer's units (e.g., 16, 64) in the code to experiment with different latent dimension sizes. Re-run the training and visualization steps to observe the impact on reconstruction quality.


## 2. Denoising Autoencoder

- **Description:** This experiment builds upon the basic autoencoder and introduces a denoising autoencoder to remove noise from images.
- **Functionality:**
    - Adds Gaussian noise to input images.
    - Trains the denoising autoencoder to reconstruct clean images from noisy inputs.
    - Visualizes noisy vs. reconstructed images.
    - Compares the performance of a basic vs. denoising autoencoder in reconstructing images.
- **File:** `denoising_autoencoder.py`
- **Answers to Questions:**
    - **Compare the performance of a basic vs. denoising autoencoder in reconstructing images:** To compare the performance, train both models on the same dataset and evaluate their reconstruction errors on a separate test set. The denoising autoencoder is expected to have a lower reconstruction error on noisy images compared to the basic autoencoder.
    - **Explain one real-world scenario where denoising autoencoders can be useful:** Denoising autoencoders can be useful in medical imaging, for example, to remove noise from MRI or X-ray images, which can improve the accuracy of diagnosis. They can also be used in security systems for image enhancement and noise reduction in surveillance footage.


## 3. Text Generation with RNN

- **Description:** This experiment utilizes a Recurrent Neural Network (RNN) with LSTM layers to generate sequences of text.
- **Functionality:**
    - Loads a text dataset (e.g., Shakespeare Sonnets).
    - Converts text into a sequence of characters.
    - Defines an RNN model using LSTM layers to predict the next character.
    - Trains the model and generates new text by sampling characters one at a time.
    - Explores the role of temperature scaling in text generation and its effect on randomness.
- **File:** `text_generation.py`
- **Answers to Questions:**
    - **Explain the role of temperature scaling in text generation and its effect on randomness:** Temperature scaling controls the randomness of the generated text. Lower temperatures make the model more confident and deterministic, resulting in repetitive and predictable text. Higher temperatures increase randomness and creativity, potentially leading to more diverse but potentially nonsensical text.


## 4. Sentiment Analysis with LSTM

- **Description:** This experiment focuses on sentiment analysis, determining if a given text expresses a positive or negative emotion, using an LSTM-based classifier.
- **Functionality:**
    - Loads the IMDB sentiment dataset.
    - Preprocesses the text data by tokenization and padding sequences.
    - Trains an LSTM-based model to classify reviews as positive or negative.
    - Generates a confusion matrix and classification report (accuracy, precision, recall, F1-score).
    - Interprets the precision-recall tradeoff in sentiment classification.
- **File:** `sentiment_analysis.py`
- **Answers to Questions:**
    - **Interpret why precision-recall tradeoff is important in sentiment classification:** In sentiment classification, the precision-recall tradeoff is important because it reflects the balance between correctly identifying positive sentiments (precision) and capturing all positive sentiments (recall). High precision means that when the model predicts a positive sentiment, it is very likely to be correct. High recall means that the model is able to identify most of the positive sentiments in the data. The optimal balance between precision and recall depends on the specific application and the relative importance of false positives and false negatives.


## Requirements

- Python 3.6 or higher
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- Scikit-learn

## Usage

1. **Install required libraries:**

bash pip install tensorflow numpy matplotlib scikit-learn

2. **Run each experiment:**
   Execute the corresponding Python script (e.g., `python autoencoder.py`) in a Google Colab environment or Jupyter Notebook.

