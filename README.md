# House Price Prediction using Linear Regression

This project demonstrates the use of **Linear Regression** machine learning algorithm to predict house prices based on square footage. It's a fundamental example of supervised learning in the field of real estate analytics.

## Overview

Linear Regression is a statistical method used to model the relationship between a dependent variable (house price) and one or more independent variables (square footage). In this project, we use a simple linear regression model to predict house prices based on the size of the house.

## Features

- **Data Analysis**: Load and explore house price data
- **Model Training**: Train a linear regression model on historical data
- **Price Prediction**: Predict house prices for new square footage values
- **Model Evaluation**: Assess model performance using MSE and R² metrics
- **Data Visualization**: Plot actual vs predicted values with regression line

## Dataset

The project uses a sample dataset containing:
- **Square Footage**: House size in square feet (1500 - 4500 sq ft)
- **Price**: Corresponding house prices ($200,000 - $600,000)

## Dependencies

Install the required packages using:

```bash
pip install -r requirements.txt
```

The project requires:
- `numpy` - Numerical computing
- `pandas` - Data manipulation and analysis
- `scikit-learn` - Machine learning library
- `matplotlib` - Data visualization

## Usage

1. **Clone or download** this repository
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the script**:
   ```bash
   python linear_regression.py
   ```

## How It Works

### 1. Data Preparation
- Load house price data into a pandas DataFrame
- Split data into features (X) and target variable (y)
- Divide data into training (80%) and testing (20%) sets

### 2. Model Training
- Initialize a Linear Regression model
- Train the model using the training dataset
- The model learns the relationship: `Price = intercept + coefficient × SquareFootage`

### 3. Prediction & Evaluation
- Make predictions on the test dataset
- Calculate evaluation metrics:
  - **Mean Squared Error (MSE)**: Average squared difference between actual and predicted values
  - **R-squared (R²)**: Proportion of variance in the dependent variable explained by the model

### 4. Visualization
- Plot actual data points vs predicted values
- Display the regression line showing the learned relationship

## Model Interpretation

The linear regression model provides:
- **Intercept**: Base price when square footage is zero
- **Coefficient**: Price increase per additional square foot
- **Equation**: `Predicted Price = Intercept + (Coefficient × Square Footage)`

## Example Output

```
Training data: (5, 1), (5,)
Testing data: (2, 1), (2,)
Intercept: 50000.0
Coefficient: 125.0
Mean Squared Error: 0.0
R-squared: 1.0
```

## Applications

This linear regression approach can be extended for:
- **Real Estate Valuation**: Predicting property values
- **Investment Analysis**: Assessing property investment potential
- **Market Research**: Understanding price trends in different areas
- **Feature Engineering**: Adding more variables (bedrooms, location, age, etc.)

## Future Enhancements

- Add more features (bedrooms, bathrooms, location, age)
- Implement multiple linear regression
- Add data preprocessing and feature scaling
- Include cross-validation for better model evaluation
- Use real estate datasets from APIs or CSV files

## Learning Objectives

This project helps understand:
- Basic concepts of supervised learning
- Linear regression algorithm implementation
- Model training and evaluation techniques
- Data visualization in machine learning
- Real-world application of ML in real estate

## License

This project is for educational purposes and learning machine learning concepts.
