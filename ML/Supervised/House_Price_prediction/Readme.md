# House Price Prediction

## Overview
This project aims to predict house prices using machine learning algorithms, specifically Linear Regression and RandomForestRegressor. The dataset used for training and testing is `house_price.csv`, which contains various features related to house properties.

## Requirements
The following Python libraries are required to run the project:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

## Dataset
The dataset contains 81 columns, including:

### Key Features:

- **Id**: Unique identifier for each property.
- **MSSubClass**: The building class of the house.
- **MSZoning**: General zoning classification.
- **LotFrontage**: Linear feet of street connected to the property.
- **LotArea**: Lot size in square feet.
- **Street**: Type of road access to the property.
- **Alley**: Type of alley access (if available).
- **LotShape**: General shape of the property.
- **LandContour**: Flatness of the property.
- **Utilities**: Type of utilities available.
- **PoolArea**: Pool area in square feet.
- **PoolQC**: Pool quality.
- **Fence**: Quality of the fence.
- **MiscFeature**: Miscellaneous feature not covered in other categories.
- **MiscVal**: Value of miscellaneous feature.
- **MoSold**: Month in which the property was sold.
- **YrSold**: Year in which the property was sold.
- **SaleType**: Type of sale.
- **SaleCondition**: Condition of sale.
- **SalePrice**: The target variable representing the sale price of the house.

## Steps to Run the Project

### 1. Import Libraries
The necessary libraries are imported for data manipulation, visualization, and machine learning.

### 2. Load the Dataset
The dataset is loaded using pandas and the first few rows are displayed.

### 3. Data Exploration
Missing values in the dataset are checked to identify columns requiring preprocessing.

### 4. Data Preprocessing
- Missing values are handled by filling them with the median.
- Categorical variables are converted to numerical using one-hot encoding.

### 5. Model Selection
- The dataset is split into training and testing sets (80% training, 20% testing).
- Two models are used for prediction:
  - **Linear Regression**
  - **Random Forest Regressor**

### 6. Model Training and Evaluation
- Linear Regression:
  - Training Score: 0.936
  - Testing Score: 0.684

- RandomForestRegressor:
  - Training Score: 0.976
  - Testing Score: 0.880

### 7. Visualization
A scatter plot is generated to compare actual vs. predicted sale prices using the Random Forest model.

## Results
- The Random Forest model outperformed Linear Regression with a higher testing score, indicating better predictive accuracy.

## Usage
To run the script, execute the following command in a Python environment:

```python
python house_price_prediction.py
```

## Future Improvements
- Feature selection to improve model efficiency.
- Hyperparameter tuning for better performance.
- Exploring deep learning models for further accuracy.

## License
This project is open-source and available for modification and use.

