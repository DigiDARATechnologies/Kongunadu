# Clustering Analysis with K-Means and Hierarchical Clustering

This project demonstrates clustering techniques using K-Means and Hierarchical Clustering on a dataset. The goal is to group data points into clusters based on their similarities and evaluate the clustering performance.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Technologies Used](#technologies-used)
4. [Steps Performed](#steps-performed)
5. [Results](#results)
6. [How to Run the Code](#how-to-run-the-code)
7. [References](#references)

---

## Introduction

Clustering is an unsupervised machine learning technique used to group data points into clusters. This project applies:
- **K-Means Clustering**: A centroid-based clustering algorithm.
- **Hierarchical Clustering**: A tree-based clustering approach visualized using dendrograms.

The project also evaluates clustering performance using the **Silhouette Score** and visualizes the results.

---

## Dataset

The project uses two datasets:
1. **`userbehaviour.csv`**: Contains user behavior data.
2. **`shopping_data.csv`**: Contains customer shopping data.

### Key Features:
- Numerical columns for clustering.
- Standardized data for better clustering performance.

---

## Technologies Used

- **Python**: Programming language.
- **Libraries**:
  - `pandas`: For data manipulation.
  - `numpy`: For numerical computations.
  - `matplotlib` and `seaborn`: For data visualization.
  - `scikit-learn`: For clustering and evaluation.
  - `scipy`: For hierarchical clustering.

---

## Steps Performed

### 1. Data Loading and Preprocessing
- Loaded datasets using `pandas`.
- Standardized the data for clustering using `StandardScaler`.

### 2. Hierarchical Clustering
- Generated a dendrogram using `scipy.cluster.hierarchy`.
- Visualized the hierarchical clustering process.

### 3. K-Means Clustering
- Used the **Elbow Method** to determine the optimal number of clusters.
- Applied K-Means clustering with the optimal number of clusters.
- Adjusted cluster numbering to start from 1.

### 4. Cluster Analysis
- Added cluster labels to the dataset.
- Calculated the mean of each cluster for analysis.

### 5. Clustering Evaluation
- Calculated the **Silhouette Score** to evaluate clustering performance.

### 6. Visualization
- Visualized clusters using scatter plots.

---

## Results

1. **Optimal Number of Clusters**:
   - Determined using the Elbow Method.

2. **Cluster Analysis**:
   - Mean values of features for each cluster were calculated.

3. **Silhouette Score**:
   - Provided a quantitative measure of clustering quality.

4. **Visualizations**:
   - Dendrogram for hierarchical clustering.
   - Scatter plot for K-Means clusters.

---

## How to Run the Code

1. **Prerequisites**:
   - Install Python 3.x.
   - Install required libraries using:
     ```bash
     pip install pandas numpy matplotlib seaborn scikit-learn scipy
     ```

2. **Run the Code**:
   - Place the datasets (`userbehaviour.csv` and `shopping_data.csv`) in the same directory as the script.
   - Execute the script in a Python environment or Jupyter Notebook.

3. **Output**:
   - Dendrogram for hierarchical clustering.
   - Scatter plot for K-Means clusters.
   - Silhouette Score for clustering evaluation.

---

## References

1. [Scikit-learn Documentation](https://scikit-learn.org/)
2. [Matplotlib Documentation](https://matplotlib.org/)
3. [Pandas Documentation](https://pandas.pydata.org/)
4. [SciPy Documentation](https://scipy.org/)

---

## Author

This project was created to demonstrate clustering techniques and their evaluation.