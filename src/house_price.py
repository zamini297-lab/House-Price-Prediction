
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score


# 1. Load dataset
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# 2. Explore target variable
print(train_df["SalePrice"].describe())

plt.figure(figsize=(8, 5))
sns.histplot(train_df["SalePrice"], kde=True)
plt.title("Sale Price Distribution")
plt.xlabel("Sale Price")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("images/saleprice_distribution.png")
plt.show()

# 3. Separate features and target
X = train_df.drop("SalePrice", axis=1)
y = train_df["SalePrice"]

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 5. Identify numerical and categorical columns
num_cols = X_train.select_dtypes(include="number").columns
cat_cols = X_train.select_dtypes(exclude="number").columns

# 6. Preprocessing pipelines
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols),
])

# 7. Build model pipeline
model = Pipeline([
    ("prep", preprocess),
    ("reg", LinearRegression()),
])

# 8. Train model
model.fit(X_train, np.log1p(y_train))

# 9. Predict test split
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)

# 10. Evaluate model
rmse_log = root_mean_squared_error(
    np.log1p(y_test),
    y_pred_log
)

rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("RMSE (log):", rmse_log)
print("RMSE:", rmse)
print("R2:", r2)

# 11. Actual vs Predicted plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Actual vs Predicted House Prices")

min_price = min(y_test.min(), y_pred.min())
max_price = max(y_test.max(), y_pred.max())
plt.plot(
    [min_price, max_price],
    [min_price, max_price],
    linestyle="--"
)

plt.tight_layout()
plt.savefig("images/actual_vs_predicted.png")
plt.show()

# 12. Predict Kaggle test dataset
home_price = np.expm1(model.predict(test_df))

predictions = pd.DataFrame({
    "Id": test_df["Id"],
    "SalePrice": home_price
})

predictions.to_csv("predictions.csv", index=False)

print("\nPredictions saved to predictions.csv")
