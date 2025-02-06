# neural_networks
 
# Tensor Manipulation and Reshaping with TensorFlow

   This Colab notebook demonstrates basic tensor manipulation and reshaping operations using TensorFlow.

   ## Task 1: Tensor Manipulations & Reshaping

   The notebook covers the following operations:

   1. Creating a random tensor of shape (4, 6).
   2. Finding the rank and shape of the tensor.
   3. Reshaping the tensor into (2, 3, 4) and transposing it to (3, 2, 4).
   4. Broadcasting a smaller tensor (1, 4) to match the larger tensor and adding them.

   ## Usage

   To run this notebook, simply open it in Google Colab and execute the cells sequentially. Ensure that you have TensorFlow installed in your environment.

	# Loss Functions and Hyperparameter Tuning

This code demonstrates the calculation and comparison of different loss functions commonly used in machine learning, particularly for hyperparameter tuning.

## Functionality

The code performs the following steps:

1. **Defines true values (`y_true`) and model predictions (`y_pred`):** These are represented as TensorFlow constants.
2. **Computes loss values:**
   - **Mean Squared Error (MSE):** Using `tf.keras.losses.MeanSquaredError()`.
   - **Categorical Cross-Entropy (CCE):** Using `tf.keras.losses.CategoricalCrossentropy()`. Note: CCE may require one-hot encoding for `y_true`.
3. **Prints loss values:** Displays the calculated MSE and CCE losses.
4. **Modifies predictions and recalculates losses:** Slightly adjusts the predictions and recomputes the losses to illustrate the impact of prediction changes.
5. **Creates a bar chart:** Visualizes the comparison between MSE and CCE losses using Matplotlib.

## Usage

1. **Prerequisites:** Ensure you have TensorFlow and Matplotlib installed. You can install them using:
	
   ## Author

   Ravi Teja Reddy Nomula (700756300)