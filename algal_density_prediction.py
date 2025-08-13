#GBDT
#--------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
# Read Excel Data
df = pd.read_excel('D:\\exp 1\\AVOCs.xlsx')

# Separate Features and Target
X = df.iloc[:, 1:-1]  # Features, assuming they are from the second column to the second-to-last column
y = df.iloc[:, -1]    # Target, assuming the last column is algal concentration

# Split Data into Training and Testing Sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Split the dataset into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=168)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=168)

# Print the sizes of the datasets
print("Training set size:", len(X_train))
print("Test set size:", len(X_test))
print("Validation set size:", len(X_val))

# Build and Train Gradient Boosting Decision Tree (GBDT) Regressor Model
gbdt_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)  # You can adjust parameters as needed
gbdt_model.fit(X_train, y_train)



# Make Predictions
y_train_pred = gbdt_model.predict(X_train)
y_test_pred = gbdt_model.predict(X_test)

# Evaluate the Model
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)
mape_train = mean_absolute_percentage_error(y_train, y_train_pred)
mape_test = mean_absolute_percentage_error(y_test, y_test_pred)
# Print Evaluation Metrics
print(f'R-squared score (Train): {r2_train}')
print(f'Mean Squared Error (Train): {mse_train}')
print(f'Root Mean Squared Error (Train): {rmse_train}')
print(f'Mean Absolute Percentage Error (Train): {mape_train}')
print(f'R-squared score (Test): {r2_test}')
print(f'Mean Squared Error (Test): {mse_test}')
print(f'Root Mean Squared Error (Test): {rmse_test}')
print(f'Mean Absolute Percentage Error (Test): {mape_test}')

#KNN
#--------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error  # Import mean_squared_error
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'


# Read Excel Data
df = pd.read_excel('D:\\exp 1\\AVOCs.xlsx')

# Separate Features and Target
X = df.iloc[:, 1:-1]  # Features, assuming they are from the second column to the second-to-last column
y = df.iloc[:, -1]    # Target, assuming the last column is algal concentration

# Split Data into Training and Testing Sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=168)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=168)


# Build and Train K-Nearest Neighbors (KNN) Regressor Model
knn_model = KNeighborsRegressor(n_neighbors=5)  # You can adjust n_neighbors as needed
knn_model.fit(X_train, y_train)

# Make Predictions
y_train_pred = knn_model.predict(X_train)
y_test_pred = knn_model.predict(X_test)

# Evaluate the Model
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)
mape_train = mean_absolute_percentage_error(y_train, y_train_pred)
mape_test = mean_absolute_percentage_error(y_test, y_test_pred)
# Print Evaluation Metrics
print(f'R-squared score (Train): {r2_train}')
print(f'Mean Squared Error (Train): {mse_train}')
print(f'Root Mean Squared Error (Train): {rmse_train}')
print(f'Mean Absolute Percentage Error (Train): {mape_train}')
print(f'R-squared score (Test): {r2_test}')
print(f'Mean Squared Error (Test): {mse_test}')
print(f'Root Mean Squared Error (Test): {rmse_test}')
print(f'Mean Absolute Percentage Error (Test): {mape_test}')

#SVM
#--------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'


# Read Excel Data
df = pd.read_excel('D:\\exp 1\\AVOCs.xlsx')

# Separate Features and Target
X = df.iloc[:, 1:-1]  # Features, assuming they are from the second column to the second-to-last column
y = df.iloc[:, -1]    # Target, assuming the last column is algal concentration

# Split Data into Training and Testing Sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=168)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=168)

# Build and Train Support Vector Machine (SVM) Regressor Model
model = SVR()
model.fit(X_train, y_train)

# Make Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluate the Model
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)
mape_train = mean_absolute_percentage_error(y_train, y_train_pred)
mape_test = mean_absolute_percentage_error(y_test, y_test_pred)
# Print Evaluation Metrics
print(f'R-squared score (Train): {r2_train}')
print(f'Mean Squared Error (Train): {mse_train}')
print(f'Root Mean Squared Error (Train): {rmse_train}')
print(f'Mean Absolute Percentage Error (Train): {mape_train}')
print(f'R-squared score (Test): {r2_test}')
print(f'Mean Squared Error (Test): {mse_test}')
print(f'Root Mean Squared Error (Test): {rmse_test}')
print(f'Mean Absolute Percentage Error (Test): {mape_test}')

#RF
#--------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

# Read Excel Data
df = pd.read_excel('D:\\exp 1\\AVOCs.xlsx')

# Separate Features and Target
X = df.iloc[:, 1:-1]  # Features, assuming they are from the second column to the second-to-last column
y = df.iloc[:, -1]    # Target, assuming the last column is algal concentration

# Split Data into Training and Testing Sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=168)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=168)
# Print the sizes of the datasets
print("Training set size:", len(X_train))
print("Test set size:", len(X_test))
print("Validation set size:", len(X_val))

model = RandomForestRegressor()

model.fit(X_train, y_train)

# Make Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluate the Model
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)
mape_train = mean_absolute_percentage_error(y_train, y_train_pred)
mape_test = mean_absolute_percentage_error(y_test, y_test_pred)
# Print Evaluation Metrics
print(f'R-squared score (Train): {r2_train}')
print(f'Mean Squared Error (Train): {mse_train}')
print(f'Root Mean Squared Error (Train): {rmse_train}')
print(f'Mean Absolute Percentage Error (Train): {mape_train}')
print(f'R-squared score (Test): {r2_test}')
print(f'Mean Squared Error (Test): {mse_test}')
print(f'Root Mean Squared Error (Test): {rmse_test}')
print(f'Mean Absolute Percentage Error (Test): {mape_test}')
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#XGB Adjustment of hyperparameters
#XGB prediction
# Feature importance
#--------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import time
import os


# Read Excel Data
file_path = 'D:\\exp 1\\AVOCs.xlsx'
df = pd.read_excel(file_path)
#
# Separate Features and Target
X = df.iloc[:, 1:-1]  # Features
y = df.iloc[:, -1]    # Target

# Split the dataset into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=168)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=168)

# Print the sizes of the datasets
print("Training set size:", len(X_train))
print("Validation set size:", len(X_val))
print("Test set size:", len(X_test))

# ======================================================================
# Hyperparameter Tuning with Grid Search
# ======================================================================
print("\nStarting hyperparameter tuning...")
start_time = time.time()

# Define the parameter grid
param_grid = {
    'n_estimators': np.arange(30, 310, 10),  # 30 to 300 with step 10
    'learning_rate': [0.1, 0.15, 0.2, 0.25, 0.3],
    'max_depth': [3, 4, 5, 6, 7, 8]
}

# Create base model with fixed parameters
xgb_base = XGBRegressor(
    min_child_weight=1,
    gamma=0,
    subsample=1,
    colsample_bytree=0.9,
    reg_lambda=0.2,
    random_state=168
)

# Set up GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    scoring=['r2', 'neg_mean_squared_error'],
    refit='r2',  # Use R2 for final model selection
    cv=5,
    verbose=3,
    n_jobs=-1,
    return_train_score=True
)

# Combine training and validation sets for cross-validation
X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

# Perform grid search
grid_search.fit(X_train_val, y_train_val)

# Calculate tuning duration
tuning_duration = time.time() - start_time
print(f"\nHyperparameter tuning completed in {tuning_duration/60:.2f} minutes")


results_df = pd.DataFrame(grid_search.cv_results_)

# Select and rename important columns
selected_columns = [
    'param_n_estimators', 'param_learning_rate', 'param_max_depth',
    'mean_train_r2', 'std_train_r2',
    'mean_test_r2', 'std_test_r2',
    'mean_train_neg_mean_squared_error', 'std_train_neg_mean_squared_error',
    'mean_test_neg_mean_squared_error', 'std_test_neg_mean_squared_error',
    'rank_test_r2'
]

# Create new column names mapping
new_column_names = {
    'param_n_estimators': 'n_estimators',
    'param_learning_rate': 'learning_rate',
    'param_max_depth': 'max_depth',
    'mean_train_r2': 'Train_R2_mean',
    'std_train_r2': 'Train_R2_std',
    'mean_test_r2': 'CV_R2_mean',
    'std_test_r2': 'CV_R2_std',
    'mean_train_neg_mean_squared_error': 'Train_NegMSE_mean',
    'std_train_neg_mean_squared_error': 'Train_NegMSE_std',
    'mean_test_neg_mean_squared_error': 'CV_NegMSE_mean',
    'std_test_neg_mean_squared_error': 'CV_NegMSE_std',
    'rank_test_r2': 'Rank'
}

# Create final results table
final_results = results_df[selected_columns].rename(columns=new_column_names)

# Calculate RMSE from negative MSE
final_results['Train_RMSE'] = np.sqrt(-final_results['Train_NegMSE_mean'])
final_results['CV_RMSE'] = np.sqrt(-final_results['CV_NegMSE_mean'])

# Reorder columns for better readability
final_results = final_results[[
    'n_estimators', 'learning_rate', 'max_depth', 'Rank',
    'Train_R2_mean', 'CV_R2_mean',
    'Train_RMSE', 'CV_RMSE',
    'Train_R2_std', 'CV_R2_std'
]]

# Sort by rank
final_results = final_results.sort_values(by='Rank')

# Create output directory if it doesn't exist
output_dir = 'D:\\exp 1\\result'
os.makedirs(output_dir, exist_ok=True)

# Save to Excel
timestamp = time.strftime("%Y%m%d-%H%M%S")
output_file = os.path.join(output_dir, f'tuning_results_{timestamp}.xlsx')
final_results.to_excel(output_file, index=False)
print(f"Tuning results saved to: {output_file}")

# ======================================================================
# Train Final Model with Best Parameters
# ======================================================================
# Get best parameters
best_params = grid_search.best_params_
print("\nBest parameters found:")
for param, value in best_params.items():
    print(f"{param}: {value}")

# Create final model with best parameters
final_model = XGBRegressor(
    **best_params,
    min_child_weight=1,
    gamma=0,
    subsample=1,
    colsample_bytree=0.9,
    reg_lambda=0.2
)

# Train final model on combined training and validation set
final_model.fit(X_train_val, y_train_val,
                eval_set=[(X_train_val, y_train_val), (X_test, y_test)],
                verbose=True)

# Make predictions
y_train_pred = final_model.predict(X_train_val)
y_test_pred = final_model.predict(X_test)

# Evaluate the final model
def evaluate_model(y_true, y_pred, set_name):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print(f"\n{set_name} Set Performance:")
    print(f"R-squared: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.4f}")
    return r2, rmse, mape

r2_train, rmse_train, mape_train = evaluate_model(y_train_val, y_train_pred, "Training")
r2_test, rmse_test, mape_test = evaluate_model(y_test, y_test_pred, "Test")

# ======================================================================
# Visualization
# ======================================================================
plt.figure(figsize=(6, 6))

# Training set
plt.scatter(y_train_val, y_train_pred, color='#52b7e8', label='Training set', marker='^', s=100)
# Test set
plt.scatter(y_test, y_test_pred, color='#e56a8a', label='Test set', marker='o', s=100)

# Reference lines
min_val = min(y.min(), y_train_pred.min(), y_test_pred.min())
max_val = max(y.max(), y_train_pred.max(), y_test_pred.max())
line_range = np.linspace(min_val, max_val, 100)
plt.plot(line_range, line_range, linestyle='-', color='teal', label='Perfect fit')
plt.plot(line_range, line_range + rmse_test, linestyle='--', color='teal', alpha=0.7)
plt.plot(line_range, line_range - rmse_test, linestyle='--', color='teal', alpha=0.7)

# Labels and title
plt.xlabel('Observation density', fontsize=28)
plt.ylabel('Prediction density', fontsize=28)
plt.tick_params(axis='both', which='major', labelsize=20)

# Legend and info box
plt.legend(fontsize=22, loc='upper left', frameon=False)
plt.text(0.95, 0.05,
         f'N={len(y_test)}\n$R^2$={r2_test:.2f}\nRMSE={rmse_test:.2f}',
         transform=plt.gca().transAxes, fontsize=24,
         verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# Save plot
plot_file = os.path.join(output_dir, f'final_model_performance_{timestamp}.png')
plt.tight_layout()
plt.savefig(plot_file, dpi=300)
print(f"Performance plot saved to: {plot_file}")
plt.show()

# ======================================================================
# Feature Importance Analysis
# ======================================================================
# Extract and save feature importances
feature_importances = final_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Save to Excel
importance_file = os.path.join(output_dir, f'feature_importances_{timestamp}.xlsx')
importance_df.to_excel(importance_file, index=False)
print(f"Feature importances saved to: {importance_file}")

# Create feature importance plot
plt.figure(figsize=(10, 6))
importance_df.plot.bar(x='Feature', y='Importance')
plt.title('Feature Importances', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save plot
feature_plot_file = os.path.join(output_dir, f'feature_importance_{timestamp}.png')
plt.savefig(feature_plot_file, dpi=300)
print(f"Feature importance plot saved to: {feature_plot_file}")
plt.show()

# ======================================================================
# Summary Report
# ======================================================================
print("\n" + "="*50)
print("HYPERPARAMETER TUNING SUMMARY")
print("="*50)
print(f"Total parameter combinations evaluated: {len(final_results)}")
print(f"Best parameters: {best_params}")
print(f"Best CV R2 score: {grid_search.best_score_:.4f}")
print(f"Tuning duration: {tuning_duration/60:.2f} minutes")
print("\nFINAL MODEL PERFORMANCE:")
print(f"Training R2: {r2_train:.4f}")
print(f"Test R2:     {r2_test:.4f}")
print(f"Test RMSE:   {rmse_test:.4f}")
print(f"Test MAPE:   {mape_test:.4f}")
print("="*50)

# SHAP
#--------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np


# Read Excel Data
df = pd.read_excel('D:\\exp 1\\AVOCs.xlsx')

# Separate Features and Target
X = df.iloc[:, 1:-1]  # Features, assuming they are from the second column to the second-to-last column
y = df.iloc[:, -1]    # Target, assuming the last column is algal concentration

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Normalize the features (X)
X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split the normalized dataset into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X_normalized, y, test_size=0.2, random_state=168)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=168)

# Build and Train XGBoost Regressor Model
xgb_model = XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=300,
        min_child_weight=1, gamma=0, subsample=1,
        colsample_bytree=0.9, reg_lambda=0.2)

# Define evaluation sets
eval_set = [(X_train, y_train), (X_val, y_val)]

# Train the model
xgb_model.fit(X_train, y_train, eval_set=eval_set, verbose=True)

# Make Predictions
y_train_pred = xgb_model.predict(X_train)
y_test_pred = xgb_model.predict(X_test)

# Evaluate the Model
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)

# Initialize the SHAP explainer with the trained model
explainer = shap.Explainer(xgb_model, X_train)

# Calculate SHAP values for the test set
shap_values = explainer(X_test)

# Get the mean absolute SHAP values for each feature
shap_abs_mean = np.abs(shap_values.values).mean(axis=0)

# Sort the features based on importance and take the top 10
sorted_indices = np.argsort(shap_abs_mean)[::-1][:10]  # Top 10 features
top_features = np.array(X.columns)[sorted_indices]
top_shap_values = shap_abs_mean[sorted_indices]

# Plot the bar chart for top 10 features
plt.figure(figsize=(6, 8))
plt.barh(top_features, top_shap_values, color="skyblue")

# Add text to each bar to display numerical values
for i, v in enumerate(top_shap_values):
    plt.text(v, i, f'{v:.4f}', va='center', fontweight='bold', fontsize=10)

# Customize the plot with adjusted font size for axis labels
plt.xlabel('Mean Absolute SHAP Value', fontsize=16, fontweight='bold')  # Adjust X-axis label font size
plt.ylabel('Top 10 Features', fontsize=16, fontweight='bold')  # Adjust Y-axis label font size
plt.title('Top 10 Feature Importance (SHAP Values)', fontsize=18, fontweight='bold')
plt.xticks(fontsize=14)  # Adjust X-axis tick label font size
plt.yticks(fontsize=14)  # Adjust Y-axis tick label font size
plt.gca().invert_yaxis()  # Invert y-axis for better alignment
plt.show()
# Generate SHAP summary plot for the top 10 features
# Extract SHAP values for the top 10 features
top_shap_values = shap_values.values[:, sorted_indices]  # SHAP values for the top 10 features
X_top_features = X_test.iloc[:, sorted_indices]  # Corresponding features from the test set

# Generate the SHAP summary plot for the top 10 features
plt.figure(figsize=(10, 8))
shap.summary_plot(top_shap_values, X_top_features, feature_names=top_features, show=True)

#Algal bloom risk assessment
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_save_model(df, model_path='xgb_model.joblib'):
   
    X = df.iloc[:, 1:-1] 
    y = df.iloc[:, -1] 

     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=168)

   
    n_bootstrap_samples = 100
    bootstrap_predictions = np.zeros((n_bootstrap_samples, len(X_test)))
      for i in range(n_bootstrap_samples):
             X_resampled, y_resampled = resample(X_train, y_train, random_state=i)

        xgb_model = XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=3)
        xgb_model.fit(X_resampled, y_resampled)
        y_pred = xgb_model.predict(X_test)
        bootstrap_predictions[i] = y_pred
  
    joblib.dump(xgb_model, model_path)
    print(f"Model saved as {model_path}")
   
    y_pred_mean = bootstrap_predictions.mean(axis=0)
    y_pred_std = bootstrap_predictions.std(axis=0)
    r2 = r2_score(y_test, y_pred_mean)
    mse = mean_squared_error(y_test, y_pred_mean)

    print(f'R-squared score (XGB): {r2}')
    print(f'Mean Squared Error (XGB): {mse}')

    prob_algal_bloom = np.mean(bootstrap_predictions.flatten() > 0.1)
    print(f'Probability of algal bloom (predicted value > 0.1): {prob_algal_bloom:.4f}')

    return xgb_model, r2, mse, prob_algal_bloom

def load_data(file_path):
    df = pd.read_excel(file_path)
    X = df.iloc[:, 1:] 
    return X

def predict_and_plot_distribution(model_path, new_data):
       loaded_model = joblib.load(model_path)
  
    n_bootstrap_samples = 100 #调整数量
    bootstrap_predictions = np.zeros((n_bootstrap_samples, len(new_data)))
   
    for i in range(n_bootstrap_samples):
        X_resampled = resample(new_data, random_state=i)
        y_pred = loaded_model.predict(X_resampled)
        bootstrap_predictions[i] = y_pred
    
    y_pred_mean = bootstrap_predictions.mean(axis=0)
    y_pred_std = bootstrap_predictions.std(axis=0)

    prob_algal_bloom = np.mean(bootstrap_predictions > 0.1, axis=0)


    for idx in range(len(new_data)):
        plt.figure(figsize=(8, 6))
        sns.histplot(bootstrap_predictions[:, idx], kde=True, bins=30, color='blue')
        plt.axvline(y_pred_mean[idx], color='red', linestyle='--', label=f'Mean Prediction: {y_pred_mean[idx]:.4f}')
        plt.xlabel('Predicted Algal Concentration', fontsize=20)
        plt.ylabel('Density', fontsize=20)
        plt.title(f'Time Point {idx + 1} Prediction Distribution', fontsize=24)
        plt.legend(fontsize=15)
        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.show()

    return y_pred_mean, prob_algal_bloom

df = pd.read_excel('D:exp 1\\AVOCs.xlsx')
trained_model, r2, mse, prob_algal_bloom_train = train_and_save_model(df)
new_data = load_data("D:\\sample_all_result.xlsx")
y_pred_mean, prob_algal_bloom = predict_and_plot_distribution('xgb_model.joblib', new_data)
for i, (mean, risk) in enumerate(zip(y_pred_mean, prob_algal_bloom)):
    print(f'Time Point {i + 1}: Mean Prediction = {mean:.4f}, Algal Bloom Risk = {risk:.4f}')

