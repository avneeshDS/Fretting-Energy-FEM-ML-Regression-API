import joblib
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split, learning_curve
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedKFold, RandomizedSearchCV

# Loading the prepared dataset and check for null values
path = Path("ML_fretting_screening.csv")
df= pd.read_csv(path, encoding="cp1252")
df_ml = df.dropna().reset_index(drop=True)
df_ml.isnull().sum()

# Prepare features and targets
X = df_ml[['P', 'D', 'µ', 'µP', 'µPD', 'a', 'D/a']]
Y1 = df_ml['DE(N-mm)']
Y2 = df_ml['Regime']

# Prpeparing data for training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y1, test_size=0.2, random_state=42)
print("X_test:", X_test)

# Define cross-validation strategy using repeated K-Fold validation method
cv_strategy = RepeatedKFold(n_splits=5,n_repeats=10,random_state=42)

# Selecting scoring metrices
scoring_metrics = {"R2": "r2","RMSE": "neg_root_mean_squared_error","MAE": "neg_mean_absolute_error"}

# Preparing pipeline for all selected models. Scaling is useful for Linear Regression and SVR. Tree-based models do not require scaling.
lr_pipeline = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
svr_pipeline = Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="rbf"))])
rf_pipeline = Pipeline([("model", RandomForestRegressor(random_state=42, n_jobs=-1))])
xgb_pipeline = Pipeline([("model", xgb.XGBRegressor(objective="reg:squarederror", 
                                                    eval_metric="rmse", random_state=42, n_jobs=-1))])

# Defining hyperparameter search space
# Linear Regression has very few meaningful hyperparameters. The intercept option is assessed.
lr_param_grid = {"model__fit_intercept": [True, False]}

svr_param_grid = {"model__C": [0.1, 1, 10, 100, 1000],"model__gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1.0],
    "model__epsilon": [0.001, 0.01, 0.05, 0.1, 0.2],"model__tol": [1e-4, 1e-3, 1e-2]}

rf_param_dist = {"model__n_estimators": [100, 200, 300, 500], "model__max_depth": [None, 3, 5, 8, 12],
    "model__min_samples_split": [2, 4, 6, 8], "model__min_samples_leaf": [1, 2, 3, 4],
    "model__max_features": ["sqrt", "log2", 0.5, 0.75, 1.0]}

xgb_param_dist = {"model__n_estimators": [50, 100, 200, 300, 500], "model__max_depth": [2, 3, 4, 5, 6],
    "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2], "model__subsample": [0.6, 0.8, 1.0],
    "model__colsample_bytree": [0.6, 0.8, 1.0], "model__min_child_weight": [1, 3, 5],
    "model__reg_alpha": [0, 0.001, 0.01, 0.1, 1.0], "model__reg_lambda": [0.1, 1.0, 5.0, 10.0]}

# Performing hyperparameter optimization.
# RMSE is minimized through the negative-RMSE score. The test set is not involved in optimization.
lr_search = GridSearchCV(estimator=lr_pipeline, param_grid=lr_param_grid, scoring=scoring_metrics, cv=cv_strategy,
    n_jobs=-1, refit="RMSE", return_train_score=True)

svr_search = GridSearchCV( estimator=svr_pipeline, param_grid=svr_param_grid, scoring=scoring_metrics, 
                          cv=cv_strategy, n_jobs=-1, refit="RMSE", return_train_score=True)

rf_search = RandomizedSearchCV(
    estimator=rf_pipeline, param_distributions=rf_param_dist, n_iter=50,
    scoring=scoring_metrics, cv=cv_strategy, random_state=42, n_jobs=-1, refit="RMSE", return_train_score=True)

xgb_search = RandomizedSearchCV(estimator=xgb_pipeline, param_distributions=xgb_param_dist, n_iter=60,
    scoring=scoring_metrics, cv=cv_strategy, random_state=42, n_jobs=-1, refit="RMSE", return_train_score=True)


# Fitting searches using training data
searches = {"Linear Regression (LR)": lr_search, "Support Vector Regression (SVR)": svr_search,
    "Random Forest Regression (RFR)": rf_search, "Extreme Gradient Boosting Regression (XGBoostR)": xgb_search}

for model_name, search in searches.items():
    print(f"\nOptimizing {model_name}...")
    search.fit(X_train, Y_train)
    print("Best parameters:")
    print(search.best_params_)
    print("Best Repeated-fold CV RMSE:", round(-search.best_score_, 6))

# Evaluating optimized models
def evaluate_regression_model(model_name,fitted_search,X_train,y_train,X_test,y_test):
    best_model = fitted_search.best_estimator_

    # Predictions from the refitted optimal model
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # Locating the selected hyperparameter combination
    best_idx = fitted_search.best_index_
    cv_results = fitted_search.cv_results_

    # Repeated five-fold CV statistics for the selected model
    cv_r2_mean = cv_results["mean_test_R2"][best_idx]
    cv_r2_sd = cv_results["std_test_R2"][best_idx]

    # Converting negative error scores back to positive values
    cv_rmse_mean = -cv_results["mean_test_RMSE"][best_idx]
    cv_rmse_sd = cv_results["std_test_RMSE"][best_idx]

    cv_mae_mean = -cv_results["mean_test_MAE"][best_idx]
    cv_mae_sd = cv_results["std_test_MAE"][best_idx]

    return {"Model": model_name, "Train_R2": r2_score(y_train, y_train_pred), "Test_R2": r2_score(y_test, y_test_pred),
        "CV_R2_Mean": cv_r2_mean, "CV_R2_SD": cv_r2_sd, "Test_RMSE": root_mean_squared_error( y_test, y_test_pred), "CV_RMSE_Mean": cv_rmse_mean, "CV_RMSE_SD": cv_rmse_sd,
        "Test_MAE": mean_absolute_error( y_test, y_test_pred), "CV_MAE_Mean": cv_mae_mean,"CV_MAE_SD": cv_mae_sd}

regression_results = []

for model_name, search in searches.items():
    result = evaluate_regression_model(model_name,search,X_train,Y_train,X_test,Y_test)
    regression_results.append(result)

results_df = pd.DataFrame(regression_results)
print("\nOptimized regression-model performance:")
print(results_df.round(4).to_string(index=False))

# Printing publication-ready mean ± SD results
for _, row in results_df.iterrows():
    print(f"\n{row['Model']}")
    print(f"Repeated-fold CV R2: "f"{row['CV_R2_Mean']:.3f} ± "f"{row['CV_R2_SD']:.3f}")
    print(f"Repeated-fold CV RMSE: "f"{row['CV_RMSE_Mean']:.3f} ± "f"{row['CV_RMSE_SD']:.3f}")
    print(f"Repeated-fold CV MAE: "f"{row['CV_MAE_Mean']:.3f} ± "f"{row['CV_MAE_SD']:.3f}")
    print(f"Independent test R2: "f"{row['Test_R2']:.3f}")
    print(f"Independent test RMSE: "f"{row['Test_RMSE']:.3f}")
    print(f"Independent test MAE: "f"{row['Test_MAE']:.3f}")


# Displaying best hyperparamters and metrics
hyperparameter_results = []
for model_name, search in searches.items():
    hyperparameter_results.append({"Model": model_name,"Selected parameters": str(search.best_params_)})
hyperparameter_df = pd.DataFrame(hyperparameter_results)
print("\nSelected hyperparameters:")
print(hyperparameter_df.to_string(index=False))

# Actual vs predicted plots for visualization
fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300)
panel_labels = ["(a)", "(b)", "(c)", "(d)"]
y_test_actual = np.asarray(Y_test).ravel()

for panel_label, ax, (model_name, search) in zip(panel_labels, axes.ravel(), searches.items()):
    best_model = search.best_estimator_
    y_test_pred = np.asarray(best_model.predict(X_test)).ravel()
    model_result = results_df.loc[results_df["Model"] == model_name].iloc[0]
    test_r2 = model_result["Test_R2"]
    train_r2 = model_result["Train_R2"]
    cv_r2 = model_result["CV_R2_Mean"]
    cv_r2_sd = model_result["CV_R2_SD"]
    test_rmse = model_result["Test_RMSE"]
    min_val = min(y_test_actual.min(), y_test_pred.min())
    max_val = max(y_test_actual.max(), y_test_pred.max())
    data_range = max_val - min_val
    margin = 0.075 * data_range if data_range > 0 else 0.05
    plot_min = min_val - margin
    plot_max = max_val + margin

    # Plot actual versus predicted values
    ax.scatter(y_test_actual, y_test_pred, color="blue", marker="x", s=45, linewidth=1.5)

    # Plot ideal prediction line
    ax.plot([plot_min, plot_max], [plot_min, plot_max], color="red", linewidth=1.5, label="Ideal prediction")
    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(plot_min, plot_max)
    ax.set_xlabel("FEM Dissipated Energy (mJ)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Predicted Dissipated Energy (mJ)", fontsize=14, fontweight="bold")
    ax.set_title(model_name, fontsize=15, fontweight="bold")

    # Performance annotation
    performance_text = f"$R^2_{{test}}$ = {test_r2:.3f}\n$R^2_{{train}}$ = {train_r2:.3f}"
    ax.text(0.05, 0.95, performance_text, transform=ax.transAxes, fontsize=14, verticalalignment="top", horizontalalignment="left", bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.80))
    ax.text(0.97, 0.97, panel_label, transform=ax.transAxes, fontsize=15, fontweight="bold", verticalalignment="top", horizontalalignment="right")
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, linestyle="--", alpha=0.30)

plt.suptitle("FEM versus Machine-Learning-Predicted Dissipated Energy", fontsize=15, fontweight="bold", y=0.96)
plt.tight_layout()
plt.show()

# Plotting learning curves
learning_curve_results = []
fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=300)

for ax, (model_name, search) in zip(axes.ravel(), searches.items()):

    print(f"\nGenerating learning curve for {model_name}")
    best_model = search.best_estimator_
    train_sizes, train_scores, valid_scores = learning_curve(estimator=best_model, X=X_train, y=Y_train, train_sizes=np.linspace(0.2, 1.0, 8), cv=cv_strategy, scoring="neg_root_mean_squared_error", n_jobs=-1)
    train_rmse = -train_scores
    valid_rmse = -valid_scores
    train_mean = train_rmse.mean(axis=1)
    train_std = train_rmse.std(axis=1)
    valid_mean = valid_rmse.mean(axis=1)
    valid_std = valid_rmse.std(axis=1)
    learning_curve_results.append(pd.DataFrame({"Model": model_name, "Training_Size": train_sizes, "Train_RMSE_Mean": train_mean, "Train_RMSE_SD": train_std, "Validation_RMSE_Mean": valid_mean, "Validation_RMSE_SD": valid_std}))
    ax.plot(train_sizes, train_mean, marker="o", linewidth=2, label="Training RMSE")
    ax.plot(train_sizes, valid_mean, marker="s", linewidth=2, label="Validation RMSE")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
    ax.fill_between(train_sizes, valid_mean - valid_std, valid_mean + valid_std, alpha=0.2)
    ax.set_title(model_name)
    ax.set_xlabel("Training Samples")
    ax.set_ylabel("RMSE")
    ax.grid(True)
    ax.legend()

plt.suptitle("Learning Curves of Optimized Regression Models", fontsize=14, weight="bold")
plt.tight_layout()
plt.show()

learning_curve_df = pd.concat(learning_curve_results, ignore_index=True)

print("\nLearning curve analysis completed.")


# Final comparison bar chart with cross-validation MAE mean and standard deviation
model_labels = ["LR", "SVR", "RFR", "XGBoostR"]
cv_mae_mean = results_df["CV_MAE_Mean"].to_numpy()
cv_mae_sd = results_df["CV_MAE_SD"].to_numpy()
x_positions = np.arange(len(model_labels))
fig, ax = plt.subplots(figsize=(6, 4.5), dpi = 600)
bars = ax.bar(x_positions,cv_mae_mean, yerr=cv_mae_sd, capsize=5, width=0.75, color="#3B7EA5", edgecolor="black",
    linewidth=0.8, error_kw={"elinewidth": 1.2, "capthick": 1.2})

ax.set_xticks(x_positions)
ax.set_xticklabels(model_labels, fontsize=11)
ax.set_xlabel("Regression Model", fontsize=14, weight = 'bold')
ax.set_ylabel("Cross-validation MAE", fontsize=14, weight = 'bold')
ax.tick_params(axis="both", labelsize=14)
ax.set_ylim(bottom=0, top=max(cv_mae_mean + cv_mae_sd) * 1.20)

# Add mean MAE above each error bar
for bar, mean_value, sd_value in zip(bars,cv_mae_mean,cv_mae_sd):
    ax.text(bar.get_x() + bar.get_width() / 2, mean_value + sd_value +  max(cv_mae_mean + cv_mae_sd) * 0.025,
        f"{mean_value:.3f}",ha="center", va="bottom", fontsize=13)
ax.text(0.97, 0.97, "(e)", transform=ax.transAxes, fontsize=15, fontweight="bold", ha="right", va="top")
plt.tight_layout()
plt.show()


#Saving the optimized models
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

model_file_names = {
    "Linear Regression (LR)": "linear_regression_energy.joblib",
    "Support Vector Regression (SVR)": "svr_energy.joblib",
    "Random Forest Regression (RFR)": "random_forest_energy.joblib",
    "Extreme Gradient Boosting Regression (XGBoostR)": "xgboost_energy.joblib"
}

for model_name, search in searches.items():
    model_path = MODELS_DIR / model_file_names[model_name]
    joblib.dump(search.best_estimator_, model_path)
    print(f"Saved {model_name}: {model_path}")

best_model_row = results_df.loc[results_df["CV_RMSE_Mean"].idxmin()]
best_model_name = best_model_row["Model"]
final_model = searches[best_model_name].best_estimator_

final_model_path = MODELS_DIR / "final_dissipated_energy_model.joblib"
joblib.dump(final_model, final_model_path)

print("\nFinal selected model:", best_model_name)
print("Selection criterion: Lowest repeated-CV RMSE")
print("CV RMSE:", round(best_model_row["CV_RMSE_Mean"], 6))
print("Final model saved at:", final_model_path)

loaded_model = joblib.load(final_model_path)

prediction_before_saving = np.asarray(final_model.predict(X_test)).ravel()
prediction_after_loading = np.asarray(loaded_model.predict(X_test)).ravel()

if np.allclose(prediction_before_saving, prediction_after_loading):
    print("Saved model verified successfully.")
else:
    raise ValueError("Loaded-model predictions do not match the original predictions.")
