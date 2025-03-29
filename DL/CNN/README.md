# Fashion MNIST Classification with TensorFlow and Keras

This project demonstrates the implementation of a neural network model using TensorFlow and Keras to classify images from the **Fashion MNIST** dataset. The project includes data preprocessing, model building, training, evaluation, and visualization using TensorBoard.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset Description](#dataset-description)
3. [Technologies Used](#technologies-used)
4. [Project Workflow](#project-workflow)
5. [Model Architecture](#model-architecture)
6. [Training and Evaluation](#training-and-evaluation)
7. [TensorBoard Visualization](#tensorboard-visualization)
8. [How to Run the Project](#how-to-run-the-project)
9. [References](#references)

---

## Introduction

The **Fashion MNIST** dataset is a collection of grayscale images of 28x28 pixels, representing 10 categories of clothing items. This project uses a neural network model to classify these images into their respective categories. The training process is monitored using **TensorBoard**, which provides interactive visualizations of metrics like loss, accuracy, and weight histograms.

---

## Dataset Description

The **Fashion MNIST** dataset contains:
- **Training Set**: 60,000 images with corresponding labels.
- **Test Set**: 10,000 images with corresponding labels.

### Class Labels:
The dataset includes the following 10 categories:
1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

---

## Technologies Used

- **Programming Language**: Python
- **Libraries**:
  - `TensorFlow` and `Keras`: For building and training the neural network.
  - `NumPy`: For numerical computations.
  - `Pandas`: For data manipulation (optional).
  - `Matplotlib` and `Seaborn`: For data visualization.
  - `TensorBoard`: For monitoring and visualizing the training process.

---

## Project Workflow

1. **Data Loading**:
   - Load the Fashion MNIST dataset using `keras.datasets.fashion_mnist`.

2. **Data Preprocessing**:
   - Normalize the pixel values to a range of 0 to 1 for better model performance.

3. **Model Building**:
   - Define a neural network model using the Keras Sequential API.

4. **Model Compilation**:
   - Compile the model with the Adam optimizer, Sparse Categorical Crossentropy loss, and accuracy as the evaluation metric.

5. **Model Training**:
   - Train the model for 10 epochs using the training data.
   - Use a TensorBoard callback to log training metrics.

6. **Model Evaluation**:
   - Evaluate the model on the test dataset to measure its performance.

7. **Visualization**:
   - Use TensorBoard to visualize training and validation metrics.

---

## Model Architecture

The neural network model consists of the following layers:

1. **Flatten Layer**:
   - Converts the 2D input images (28x28) into a 1D array of 784 pixels.

2. **Dense Layer (Hidden Layer)**:
   - Fully connected layer with 128 neurons and ReLU activation.

3. **Dense Layer (Output Layer)**:
   - Fully connected layer with 10 neurons (one for each class) and Softmax activation.

### Summary of the Model:
| Layer (type)         | Output Shape   | Parameters |
|----------------------|----------------|------------|
| **Flatten**          | (None, 784)   | 0          |
| **Dense (Hidden)**   | (None, 128)   | 100,480    |
| **Dense (Output)**   | (None, 10)    | 1,290      |

---

## Training and Evaluation

1. **Compilation**:
   - Optimizer: Adam
   - Loss Function: Sparse Categorical Crossentropy
   - Metric: Accuracy

2. **Training**:
   - Train the model for 10 epochs using the training data.
   - Use a TensorBoard callback to log metrics.

3. **Evaluation**:
   - Evaluate the model on the test dataset to measure accuracy.

---

## TensorBoard Visualization

**TensorBoard** is used to monitor the training process. It provides visualizations for:
- Training and validation loss.
- Training and validation accuracy.
- Histograms of model weights.

### How to Launch TensorBoard:
1. After training, run the following command in the terminal:
   ```bash
   tensorboard --logdir=logs
   ```
2. Open the provided URL in a web browser to view the TensorBoard dashboard.

### Prerequisites
- Install Python 3.x.
- Install the required libraries

## Steps to Run
1. Clone the repository or download the project files.
2. Run the script in a Python environment or Jupyter Notebook.
3. Monitor the training process using TensorBoard.

## References
1. TensorFlow Documentation
2. Keras Documentation
3. Fashion MNIST Dataset

## Author
This project was created to demonstrate the implementation of a neural network for image classification using TensorFlow and Keras.