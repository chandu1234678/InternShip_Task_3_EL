"""
Task 3: Linear Regression Implementation
AI & ML Internship - National Level Quality

This script demonstrates both Simple and Multiple Linear Regression
with comprehensive analysis, visualization, and model evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class LinearRegressionAnalysis:
    """
    Comprehensive Linear Regression Analysis Class
    Handles data loading, preprocessing, model training, and evaluation
    """
    
    def __init__(self, data_path=None):
        """Initialize the analysis with optional data path"""
        self.data_path = data_path
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load and prepare the dataset"""
        # Using California Housing dataset as example
        # You can replace this with any dataset
        from sklearn.datasets import fetch_california_housing
        
        print(" Loading California Housing Dataset...")
        housing = fetch_california_housing()
        
        # Create DataFrame
        self.df = pd.DataFrame(housing.data, columns=housing.feature_names)
        self.df['Price'] = housing.target
        
        print(f" Dataset loaded successfully!")
        print(f"   Shape: {self.df.shape}")
        print(f"   Features: {list(self.df.columns[:-1])}")
        
        return self.df
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        print("\n" + "="*60)
        print(" EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        # Basic statistics
        print("\n1. Dataset Info:")
        print(self.df.info())
        
        print("\n2. Statistical Summary:")
        print(self.df.describe())
        
        print("\n3. Missing Values:")
        print(self.df.isnull().sum())
        
        print("\n4. Correlation with Target:")
        correlations = self.df.corr()['Price'].sort_values(ascending=False)
        print(correlations)
        
        # Visualizations
        self._plot_eda()
        
    def _plot_eda(self):
        """Create EDA visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Distribution of target variable
        axes[0, 0].hist(self.df['Price'], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Distribution of House Prices', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Price (in $100,000s)')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Correlation heatmap
        corr_matrix = self.df.corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    ax=axes[0, 1], cbar_kws={'label': 'Correlation'})
        axes[0, 1].set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        
        # 3. Scatter plot - Most correlated feature
        most_corr_feature = self.df.corr()['Price'].abs().sort_values(ascending=False).index[1]
        axes[1, 0].scatter(self.df[most_corr_feature], self.df['Price'], alpha=0.5)
        axes[1, 0].set_title(f'Price vs {most_corr_feature}', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel(most_corr_feature)
        axes[1, 0].set_ylabel('Price')
        
        # 4. Box plot for outlier detection
        self.df['Price'].plot(kind='box', ax=axes[1, 1])
        axes[1, 1].set_title('Box Plot - Price Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Price')
        
        plt.tight_layout()
        plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
        print("\n EDA plots saved as 'eda_analysis.png'")
        plt.close()
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        print("\n" + "="*60)
        print(" DATA PREPARATION")
        print("="*60)
        
        # Separate features and target
        X = self.df.drop('Price', axis=1)
        y = self.df['Price']
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"\n Data split completed:")
        print(f"   Training set: {self.X_train.shape[0]} samples")
        print(f"   Testing set: {self.X_test.shape[0]} samples")
        print(f"   Features: {self.X_train.shape[1]}")
        
    def train_model(self):
        """Train the Linear Regression model"""
        print("\n" + "="*60)
        print(" MODEL TRAINING")
        print("="*60)
        
        print("\n Training Linear Regression model...")
        self.model.fit(self.X_train, self.y_train)
        print(" Model training completed!")
        
        # Display model parameters
        print("\n Model Coefficients:")
        coef_df = pd.DataFrame({
            'Feature': self.X_train.columns,
            'Coefficient': self.model.coef_
        }).sort_values('Coefficient', key=abs, ascending=False)
        print(coef_df.to_string(index=False))
        print(f"\nIntercept: {self.model.intercept_:.4f}")
        
    def evaluate_model(self):
        """Evaluate model performance"""
        print("\n" + "="*60)
        print(" MODEL EVALUATION")
        print("="*60)
        
        # Make predictions
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        
        # Calculate metrics for training set
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        train_mse = mean_squared_error(self.y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_r2 = r2_score(self.y_train, y_train_pred)
        
        # Calculate metrics for testing set
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        test_mse = mean_squared_error(self.y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(self.y_test, y_test_pred)
        
        # Display results
        print("\n Training Set Performance:")
        print(f"   MAE:  {train_mae:.4f}")
        print(f"   MSE:  {train_mse:.4f}")
        print(f"   RMSE: {train_rmse:.4f}")
        print(f"   R:   {train_r2:.4f}")
        
        print("\n Testing Set Performance:")
        print(f"   MAE:  {test_mae:.4f}")
        print(f"   MSE:  {test_mse:.4f}")
        print(f"   RMSE: {test_rmse:.4f}")
        print(f"   R:   {test_r2:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, 
                                     cv=5, scoring='r2')
        print(f"\n Cross-Validation R Scores: {cv_scores}")
        print(f"   Mean CV R: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Store predictions for plotting
        self.y_train_pred = y_train_pred
        self.y_test_pred = y_test_pred
        
        # Interpretation
        self._interpret_results(test_r2, test_mae, test_rmse)
        
    def _interpret_results(self, r2, mae, rmse):
        """Interpret the model results"""
        print("\n" + "="*60)
        print(" MODEL INTERPRETATION")
        print("="*60)
        
        print(f"\n1. R Score ({r2:.4f}):")
        if r2 > 0.9:
            print("    Excellent! Model explains >90% of variance")
        elif r2 > 0.7:
            print("    Good! Model explains >70% of variance")
        elif r2 > 0.5:
            print("     Moderate. Model explains >50% of variance")
        else:
            print("    Poor. Model explains <50% of variance")
        
        print(f"\n2. MAE ({mae:.4f}):")
        print(f"   On average, predictions are off by ${mae*100000:.2f}")
        
        print(f"\n3. RMSE ({rmse:.4f}):")
        print(f"   Root Mean Squared Error: ${rmse*100000:.2f}")
        print("   RMSE penalizes larger errors more than MAE")
        
    def plot_results(self):
        """Create comprehensive result visualizations"""
        print("\n" + "="*60)
        print(" CREATING VISUALIZATIONS")
        print("="*60)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Actual vs Predicted (Training)
        axes[0, 0].scatter(self.y_train, self.y_train_pred, alpha=0.5, s=20)
        axes[0, 0].plot([self.y_train.min(), self.y_train.max()], 
                        [self.y_train.min(), self.y_train.max()], 
                        'r--', lw=2, label='Perfect Prediction')
        axes[0, 0].set_xlabel('Actual Price', fontsize=12)
        axes[0, 0].set_ylabel('Predicted Price', fontsize=12)
        axes[0, 0].set_title('Training Set: Actual vs Predicted', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Actual vs Predicted (Testing)
        axes[0, 1].scatter(self.y_test, self.y_test_pred, alpha=0.5, s=20, color='green')
        axes[0, 1].plot([self.y_test.min(), self.y_test.max()], 
                        [self.y_test.min(), self.y_test.max()], 
                        'r--', lw=2, label='Perfect Prediction')
        axes[0, 1].set_xlabel('Actual Price', fontsize=12)
        axes[0, 1].set_ylabel('Predicted Price', fontsize=12)
        axes[0, 1].set_title('Testing Set: Actual vs Predicted', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Residual Plot (Training)
        train_residuals = self.y_train - self.y_train_pred
        axes[1, 0].scatter(self.y_train_pred, train_residuals, alpha=0.5, s=20)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('Predicted Price', fontsize=12)
        axes[1, 0].set_ylabel('Residuals', fontsize=12)
        axes[1, 0].set_title('Training Set: Residual Plot', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Residual Plot (Testing)
        test_residuals = self.y_test - self.y_test_pred
        axes[1, 1].scatter(self.y_test_pred, test_residuals, alpha=0.5, s=20, color='green')
        axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1, 1].set_xlabel('Predicted Price', fontsize=12)
        axes[1, 1].set_ylabel('Residuals', fontsize=12)
        axes[1, 1].set_title('Testing Set: Residual Plot', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        print("\n Evaluation plots saved as 'model_evaluation.png'")
        plt.close()
        
        # Feature importance plot
        self._plot_feature_importance()
        
    def _plot_feature_importance(self):
        """Plot feature importance based on coefficients"""
        coef_df = pd.DataFrame({
            'Feature': self.X_train.columns,
            'Coefficient': self.model.coef_
        }).sort_values('Coefficient', key=abs, ascending=True)
        
        plt.figure(figsize=(10, 8))
        colors = ['red' if x < 0 else 'green' for x in coef_df['Coefficient']]
        plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, alpha=0.7)
        plt.xlabel('Coefficient Value', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title('Feature Importance (Coefficient Magnitude)', fontsize=14, fontweight='bold')
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print(" Feature importance plot saved as 'feature_importance.png'")
        plt.close()
    
    def check_assumptions(self):
        """Check Linear Regression assumptions"""
        print("\n" + "="*60)
        print(" CHECKING LINEAR REGRESSION ASSUMPTIONS")
        print("="*60)
        
        residuals = self.y_test - self.y_test_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Linearity Check
        axes[0, 0].scatter(self.y_test_pred, self.y_test, alpha=0.5)
        axes[0, 0].plot([self.y_test.min(), self.y_test.max()], 
                        [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Actual Values')
        axes[0, 0].set_title('Linearity Check', fontweight='bold')
        
        # 2. Normality of Residuals
        axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Normality of Residuals', fontweight='bold')
        
        # 3. Homoscedasticity
        axes[1, 0].scatter(self.y_test_pred, residuals, alpha=0.5)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('Predicted Values')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].set_title('Homoscedasticity Check', fontweight='bold')
        
        # 4. Q-Q Plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('assumptions_check.png', dpi=300, bbox_inches='tight')
        print("\n Assumption check plots saved as 'assumptions_check.png'")
        plt.close()
        
        # Statistical tests
        print("\n Statistical Tests:")
        print(f"   Residuals Mean: {residuals.mean():.6f} (should be ~0)")
        print(f"   Residuals Std: {residuals.std():.4f}")
        
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("\n" + "="*70)
        print("LINEAR REGRESSION - COMPLETE ANALYSIS")
        print("="*70)
        
        self.load_data()
        self.explore_data()
        self.prepare_data()
        self.train_model()
        self.evaluate_model()
        self.plot_results()
        self.check_assumptions()
        
        print("\n" + "="*70)
        print(" ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\n Generated Files:")
        print("   1. eda_analysis.png - Exploratory Data Analysis")
        print("   2. model_evaluation.png - Model Performance")
        print("   3. feature_importance.png - Feature Coefficients")
        print("   4. assumptions_check.png - Regression Assumptions")


if __name__ == "__main__":
    # Run the complete analysis
    analyzer = LinearRegressionAnalysis()
    analyzer.run_complete_analysis()
    
    print("\n" + "="*70)
    print(" INTERVIEW QUESTIONS ANSWERS")
    print("="*70)
    print("""
1. Linear Regression Assumptions:
   - Linearity between features and target
   - Independence of errors
   - Homoscedasticity (constant variance)
   - Normality of residuals
   - No multicollinearity

2. Coefficient Interpretation:
   - Each coefficient shows the change in target for 1 unit change in feature
   - Sign indicates positive/negative relationship
   - Magnitude shows strength of relationship

3. R Score Significance:
   - Measures proportion of variance explained (0 to 1)
   - Higher R = better model fit
   - R = 1 means perfect predictions

4. MSE vs MAE:
   - MSE penalizes large errors more (squared)
   - Use MSE when large errors are particularly bad
   - MAE is more robust to outliers

5. Detecting Multicollinearity:
   - Check correlation matrix
   - Calculate VIF (Variance Inflation Factor)
   - VIF > 10 indicates multicollinearity

6. Simple vs Multiple Regression:
   - Simple: One independent variable
   - Multiple: Two or more independent variables

7. Linear Regression for Classification:
   - Not recommended - use Logistic Regression
   - Linear regression predicts continuous values
   - Classification needs probability/class labels

8. Violating Assumptions:
   - Biased coefficients
   - Unreliable predictions
   - Invalid statistical tests
   - Poor generalization
    """)

