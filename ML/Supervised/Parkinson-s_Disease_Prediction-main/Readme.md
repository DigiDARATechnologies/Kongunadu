# Parkinson's Disease Prediction using Machine Learning and Flask

## Overview
This project is a **machine learning-based web application** that predicts whether a patient has **Parkinson's disease** using voice measurements. The application is built using **Flask** for the backend and **multiple machine learning models**, including:
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**
- **Random Forest Classifier**
- **Logistic Regression**
- **Naive Bayes Classifier**

The user inputs voice-related attributes, and the model predicts whether the patient has Parkinson's disease or not.

## Dataset
The dataset used in this project is `parkinsons.csv`, which contains various features extracted from voice recordings.

## Features
The dataset includes:
- **MDVP:Fo(Hz)** - Fundamental frequency
- **MDVP:Fhi(Hz)** - Maximum fundamental frequency
- **MDVP:Flo(Hz)** - Minimum fundamental frequency
- **MDVP:Jitter(%)** - Variation in pitch
- **MDVP:Shimmer** - Variation in amplitude
- **NHR & HNR** - Measures of noise and harmonic-to-noise ratio
- **RPDE & DFA** - Nonlinear dynamic measures
- **spread1, spread2, D2, PPE** - Measures of fundamental frequency variation
- **status** - 1 for Parkinson's, 0 for healthy

## Installation
To run the project, follow these steps:

### Prerequisites
Ensure you have **Python 3.x** installed.

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-repository-url.git
cd parkinsons-prediction
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Application
```bash
python app.py
```
The Flask server will start at `http://127.0.0.1:5000/`.

## Usage
1. Open the browser and go to `http://127.0.0.1:5000/`
2. Enter the required patient voice measurement values.
3. Click **Predict** to get the result.
4. The model will predict whether the patient has **Parkinson's disease** or not.

## Machine Learning Models Used
### **1. Data Preprocessing:**
- Features are scaled using `StandardScaler`.
- Data is split into **training (80%)** and **testing (20%)** sets.

### **2. Model Training:**
- Various models are trained and evaluated for accuracy.
- The **Random Forest Classifier** is used for final predictions.

### **3. Evaluation Metrics:**
- **Accuracy Score**
- **Confusion Matrix**
- **Cross-validation**

## Visualization
The project includes several visualizations:
- **Boxplots** to show feature distributions.
- **Histograms** for feature distribution comparisons.
- **Heatmap** to show feature correlation.

## Technologies Used
- **Python**
- **Flask** (Web framework)
- **Pandas** (Data processing)
- **NumPy** (Numerical computation)
- **Matplotlib & Seaborn** (Data visualization)
- **Scikit-Learn** (Machine learning models)

## Future Improvements
- **Deploy on a cloud platform (AWS/GCP/Heroku)**
- **Improve accuracy with deep learning (Neural Networks)**
- **Enhance UI for better user experience**

## Author
Developed by **Premkumar**

## License
This project is licensed under the MIT License.

