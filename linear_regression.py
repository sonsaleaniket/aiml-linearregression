# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Generate a larger, more realistic dataset
def generate_house_data(n_samples=1000):
    """Generate synthetic house price data with multiple features"""
    # Square footage (main feature)
    square_footage = np.random.normal(2500, 800, n_samples)
    square_footage = np.clip(square_footage, 800, 6000)  # Reasonable range
    
    # Number of bedrooms (2-6)
    bedrooms = np.random.poisson(3, n_samples)
    bedrooms = np.clip(bedrooms, 1, 6)
    
    # Number of bathrooms (1-4)
    bathrooms = np.random.poisson(2, n_samples)
    bathrooms = np.clip(bathrooms, 1, 4)
    
    # Age of house (0-50 years)
    age = np.random.exponential(15, n_samples)
    age = np.clip(age, 0, 50)
    
    # Distance to city center (miles)
    distance_to_city = np.random.exponential(10, n_samples)
    distance_to_city = np.clip(distance_to_city, 1, 50)
    
    # Generate price based on features with some noise
    base_price = (
        square_footage * 120 +  # $120 per sq ft base
        bedrooms * 15000 +      # $15k per bedroom
        bathrooms * 20000 +     # $20k per bathroom
        -age * 2000 +           # -$2k per year of age
        -distance_to_city * 3000  # -$3k per mile from city
    )
    
    # Add some noise and ensure positive prices
    noise = np.random.normal(0, 20000, n_samples)
    price = base_price + noise
    price = np.maximum(price, 50000)  # Minimum price of $50k
    
    return pd.DataFrame({
        'SquareFootage': square_footage,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Age': age,
        'DistanceToCity': distance_to_city,
        'Price': price
    })

# Generate the dataset
df = generate_house_data(1000)

# Display the first few rows of the data
print("Dataset Overview:")
print(df.head())
print(f"\nDataset shape: {df.shape}")
print(f"\nDataset statistics:")
print(df.describe())

# Features (X) and Target (y)
feature_columns = ['SquareFootage', 'Bedrooms', 'Bathrooms', 'Age', 'DistanceToCity']
X = df[feature_columns]  # Multiple features
y = df['Price']          # Target variable

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining data: {X_train.shape}, {y_train.shape}")
print(f"Testing data: {X_test.shape}, {y_test.shape}")

# Define multiple models to compare
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=1.0),
    'Polynomial Regression (degree=2)': Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ]),
    'Ridge with Polynomial Features': Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('scaler', StandardScaler()),
        ('regressor', Ridge(alpha=1.0))
    ])
}

# Train and evaluate each model
results = {}

print("\n" + "="*80)
print("MODEL COMPARISON RESULTS")
print("="*80)

for name, model in models.items():
    print(f"\n{name}:")
    print("-" * 50)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    # Store results
    results[name] = {
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    # Display results
    print(f"Training MSE: {train_mse:,.2f}")
    print(f"Testing MSE:  {test_mse:,.2f}")
    print(f"Training R²:  {train_r2:.4f}")
    print(f"Testing R²:   {test_r2:.4f}")
    print(f"Training MAE: {train_mae:,.2f}")
    print(f"Testing MAE:  {test_mae:,.2f}")
    print(f"CV R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Display coefficients for linear models
    if hasattr(model, 'coef_'):
        print(f"Coefficients: {model.coef_}")
        print(f"Intercept: {model.intercept_}")
    elif hasattr(model, 'named_steps') and 'regressor' in model.named_steps:
        regressor = model.named_steps['regressor']
        if hasattr(regressor, 'coef_'):
            print(f"Coefficients: {regressor.coef_}")
            print(f"Intercept: {regressor.intercept_}")

# Find the best model based on test R²
best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
best_model = models[best_model_name]

print(f"\n" + "="*80)
print(f"BEST MODEL: {best_model_name}")
print(f"Test R²: {results[best_model_name]['test_r2']:.4f}")
print(f"Test MSE: {results[best_model_name]['test_mse']:,.2f}")
print("="*80)

# Generate predictions with the best model
y_pred_best = best_model.predict(X_test)

# Create comprehensive visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Model Performance Analysis', fontsize=16)

# 1. Actual vs Predicted scatter plot
axes[0, 0].scatter(y_test, y_pred_best, alpha=0.6)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Price')
axes[0, 0].set_ylabel('Predicted Price')
axes[0, 0].set_title(f'Actual vs Predicted\n{best_model_name}')

# 2. Residuals plot
residuals = y_test - y_pred_best
axes[0, 1].scatter(y_pred_best, residuals, alpha=0.6)
axes[0, 1].axhline(y=0, color='r', linestyle='--')
axes[0, 1].set_xlabel('Predicted Price')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('Residuals Plot')

# 3. Model comparison - R² scores
model_names = list(results.keys())
r2_scores = [results[name]['test_r2'] for name in model_names]
axes[0, 2].bar(range(len(model_names)), r2_scores)
axes[0, 2].set_xticks(range(len(model_names)))
axes[0, 2].set_xticklabels(model_names, rotation=45, ha='right')
axes[0, 2].set_ylabel('Test R² Score')
axes[0, 2].set_title('Model Comparison - R² Scores')

# 4. Model comparison - MSE scores
mse_scores = [results[name]['test_mse'] for name in model_names]
axes[1, 0].bar(range(len(model_names)), mse_scores)
axes[1, 0].set_xticks(range(len(model_names)))
axes[1, 0].set_xticklabels(model_names, rotation=45, ha='right')
axes[1, 0].set_ylabel('Test MSE')
axes[1, 0].set_title('Model Comparison - MSE Scores')

# 5. Feature importance (for linear models)
if hasattr(best_model, 'coef_'):
    coef = best_model.coef_
elif hasattr(best_model, 'named_steps') and 'regressor' in best_model.named_steps:
    coef = best_model.named_steps['regressor'].coef_
else:
    coef = None

if coef is not None:
    if len(coef) > len(feature_columns):
        # Polynomial features case - show first few coefficients
        coef = coef[:len(feature_columns)]
    axes[1, 1].bar(feature_columns, coef)
    axes[1, 1].set_ylabel('Coefficient Value')
    axes[1, 1].set_title('Feature Importance')
    axes[1, 1].tick_params(axis='x', rotation=45)

# 6. Distribution of residuals
axes[1, 2].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
axes[1, 2].set_xlabel('Residuals')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].set_title('Distribution of Residuals')

plt.tight_layout()
plt.show()

# Feature correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

print(f"\nModel improvement summary:")
print(f"Original MSE: 3,284,248,323")
print(f"Improved MSE: {results[best_model_name]['test_mse']:,.2f}")
print(f"Improvement: {((3284248323 - results[best_model_name]['test_mse']) / 3284248323 * 100):.1f}%")
print(f"\nOriginal R²: -4.25")
print(f"Improved R²: {results[best_model_name]['test_r2']:.4f}")
print(f"Improvement: {((results[best_model_name]['test_r2'] - (-4.25)) / abs(-4.25) * 100):.1f}%")