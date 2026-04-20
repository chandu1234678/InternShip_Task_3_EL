# Task 3: Linear Regression Implementation

## 🎯 Objective
Implement and understand simple & multiple linear regression with comprehensive analysis and evaluation.

## 📊 Dataset
California Housing Dataset (built-in scikit-learn dataset)
- **Features**: 8 numerical features including median income, house age, average rooms, etc.
- **Target**: Median house value for California districts
- **Samples**: 20,640 instances

## 🛠️ Tools & Libraries Used
- **Python 3.x**
- **Scikit-learn**: Model training and evaluation
- **Pandas**: Data manipulation
- **Matplotlib & Seaborn**: Visualizations
- **NumPy**: Numerical computations
- **SciPy**: Statistical tests

## 📁 Project Structure
```
task3_linear_regression/
├── linear_regression.py      # Main implementation
├── README.md                  # This file
├── requirements.txt           # Dependencies
├── eda_analysis.png          # Exploratory data analysis plots
├── model_evaluation.png      # Model performance visualizations
├── feature_importance.png    # Feature coefficient analysis
└── assumptions_check.png     # Regression assumptions validation
```

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Analysis
```bash
python linear_regression.py
```

## 📈 Implementation Steps

### 1. Data Loading & Preprocessing
- Load California Housing dataset
- Check for missing values
- Explore data distribution
- Analyze feature correlations

### 2. Exploratory Data Analysis (EDA)
- Distribution plots
- Correlation heatmap
- Scatter plots for relationships
- Box plots for outlier detection

### 3. Data Preparation
- Split data into training (80%) and testing (20%) sets
- Feature-target separation
- Data validation

### 4. Model Training
- Fit Linear Regression model
- Extract coefficients and intercept
- Analyze feature importance

### 5. Model Evaluation
- **Metrics Used**:
  - MAE (Mean Absolute Error)
  - MSE (Mean Squared Error)
  - RMSE (Root Mean Squared Error)
  - R² Score (Coefficient of Determination)
- Cross-validation (5-fold)
- Training vs Testing performance comparison

### 6. Visualization
- Actual vs Predicted plots
- Residual analysis
- Feature importance charts
- Assumption validation plots

### 7. Assumption Checking
- Linearity verification
- Normality of residuals
- Homoscedasticity check
- Q-Q plot analysis

## 📊 Results

### Model Performance
- **Training R² Score**: ~0.60
- **Testing R² Score**: ~0.58
- **MAE**: ~0.53 ($53,000)
- **RMSE**: ~0.73 ($73,000)

### Key Insights
1. **Most Important Features**:
   - Median Income (strongest positive correlation)
   - Average Occupancy (negative impact)
   - Latitude/Longitude (location matters)

2. **Model Interpretation**:
   - Model explains ~58% of price variance
   - Reasonable performance for baseline model
   - Some non-linear relationships exist

3. **Assumptions**:
   - ✅ Linearity: Generally satisfied
   - ✅ Independence: Satisfied
   - ⚠️ Homoscedasticity: Some variance patterns
   - ⚠️ Normality: Slight deviations in residuals

## 💡 What I Learned

### Technical Skills
- Linear regression implementation from scratch
- Model evaluation using multiple metrics
- Statistical assumption validation
- Feature importance interpretation
- Cross-validation techniques

### Concepts Mastered
- Regression modeling fundamentals
- Coefficient interpretation
- R² score significance
- MSE vs MAE trade-offs
- Multicollinearity detection
- Simple vs Multiple regression differences

## 📚 Interview Questions Answered

### 1. What assumptions does linear regression make?
- **Linearity**: Linear relationship between features and target
- **Independence**: Observations are independent
- **Homoscedasticity**: Constant variance of errors
- **Normality**: Residuals are normally distributed
- **No Multicollinearity**: Features are not highly correlated

### 2. How do you interpret the coefficients?
Each coefficient represents the change in the target variable for a one-unit change in the corresponding feature, holding all other features constant. Positive coefficients indicate positive relationships, negative indicate inverse relationships.

### 3. What is R² score and its significance?
R² (Coefficient of Determination) measures the proportion of variance in the target variable explained by the model. It ranges from 0 to 1:
- **1.0**: Perfect predictions
- **0.7-0.9**: Strong model
- **0.5-0.7**: Moderate model
- **<0.5**: Weak model

### 4. When would you prefer MSE over MAE?
- **Use MSE when**: Large errors are particularly problematic (squared penalty)
- **Use MAE when**: All errors should be treated equally, or data has outliers
- MSE is more sensitive to outliers due to squaring

### 5. How do you detect multicollinearity?
- **Correlation Matrix**: Check pairwise correlations (>0.8 is concerning)
- **VIF (Variance Inflation Factor)**: VIF > 10 indicates multicollinearity
- **Condition Number**: High values suggest multicollinearity

### 6. What is the difference between simple and multiple regression?
- **Simple**: One independent variable (y = mx + b)
- **Multiple**: Two or more independent variables (y = b₀ + b₁x₁ + b₂x₂ + ...)

### 7. Can linear regression be used for classification?
No, linear regression is not suitable for classification because:
- It predicts continuous values, not probabilities
- Predictions can be outside [0,1] range
- Use Logistic Regression for binary classification instead

### 8. What happens if you violate regression assumptions?
- **Biased coefficients**: Incorrect feature importance
- **Unreliable predictions**: Poor generalization
- **Invalid statistical tests**: Confidence intervals meaningless
- **Reduced model performance**: Lower accuracy

## 🎓 Advanced Concepts Explored
- Cross-validation for robust evaluation
- Residual analysis for model diagnostics
- Feature scaling considerations
- Regularization awareness (Ridge/Lasso)

## 🔧 Possible Improvements
1. Feature engineering (polynomial features, interactions)
2. Regularization (Ridge/Lasso) for better generalization
3. Feature selection to reduce multicollinearity
4. Non-linear transformations for better fit
5. Ensemble methods for improved accuracy