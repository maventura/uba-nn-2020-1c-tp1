# Custom Multilayer Perceptron & Linear Regression

This repository contains a from-scratch implementation of a **Multilayer Perceptron (MLP)** and a **Linear Regression** model using Python and NumPy. The project was built for educational purposes to understand the inner workings of backpropagation, gradient descent, momentum, and weight decay without relying on high-level machine learning libraries like TensorFlow or scikit-learn.

##  Project Structure

* multilayer_perceptron.py: 
  The core Object-Oriented implementation of the Multilayer Perceptron. It supports configurable layer architectures, customizable activation functions (Sigmoid, Step, Tanh), and backpropagation with weight decay.
* tp1ej1.py: 
  Command-line script for training and evaluating the MLP on a classification dataset. It includes data normalization and model serialization (saving/loading trained weights using Pickle).
* tp1ej2.py: 
  Command-line script for training and evaluating the Linear Regression model. It utilizes gradient descent with momentum and a two-stage learning rate scheduler.
* linear_regression.ipynb & multi_layer_perceptron.ipynb: 
  Jupyter Notebooks used for experimentation, hyperparameter tuning, and visualizing the Mean Squared Error (MSE) across training epochs.

##  Key Features

* **From-Scratch Implementation:** Only relies on Numpy for matrix multiplications and mathematical operations.
* **Multilayer Perceptron:**
  * Configurable network topology (number of layers and neurons).
  * Activation functions: sigmoid, tanh, and step.
  * Training loop with customizable learning rate ($\nu$), weight decay, and early stopping based on $\epsilon$.
* **Linear Regression:**
  * Multi-variable linear regression using gradient descent.
  * Momentum implementation to accelerate convergence and escape local minima.
  * Adaptive learning rate mechanism (increases learning rate in the second half of epochs).
* **Model Serialization:** Easily save trained network weights to a .pkl file and load them later to perform predictions without retraining.

##  Requirements

To run the scripts and notebooks, you need:
* Python 3.x
* NumPy
* Pandas
* Matplotlib (for running the Jupyter notebooks)

##  Usage

Both execution scripts ("tp1ej1.py" and "tp1ej2.py") follow the same command-line argument structure. 

If the specified ".pkl" file exists, the script will load the pre-trained weights and evaluate the dataset (outputting Accuracy or Mean Squared Error). If the file does not exist, the script will train a new model from scratch and save the resulting weights to the specified filename.

### 1. Multilayer Perceptron (Classification)

To train or test the MLP model, use "tp1ej1.py". This script expects the target classes to be in the first column of the CSV (e.g., 'M' for positive class).

    python tp1ej1.py <model_weights.pkl> <path_to_dataset.csv>

Example:

    python tp1ej1.py my_mlp_model.pkl ./data/dataset1.csv

### 2. Linear Regression

To train or test the Linear Regression model, use "tp1ej2.py". This script assumes the first 8 columns are input features and the remaining columns are the continuous target variables.

    python tp1ej2.py <model_weights.pkl> <path_to_dataset.csv>

Example:

    python tp1ej2.py my_linreg_model.pkl ./data/dataset2.csv

##  Notes on Optimization

During the development of the Linear Regression model, several optimization techniques were explored:
* **Data Normalization:** Mandatory for the model to converge properly without diverging at reasonable learning rates.
* **Momentum:** Adding a momentum factor significantly smoothed out training oscillations and improved convergence speed.
* **Learning Rate Scheduling:** Splitting the training process to use a smaller learning rate initially and a larger one later yielded better results than a static learning rate.
