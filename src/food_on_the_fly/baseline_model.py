import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Step 1: Make fake delivery data
np.random.seed(42)
data = pd.DataFrame(
    {
        "distance_miles": np.random.uniform(1, 20, 500),
        "traffic_level": np.random.randint(1, 5, 500),
        "time_of_day": np.random.randint(0, 24, 500),
    }
)
# Fake delivery time = roughly based on distance and traffic
data["delivery_minutes"] = (
    data["distance_miles"] * 3 + data["traffic_level"] * 5 + np.random.normal(0, 5, 500)  # noqa: E501
)

# Step 2: Split into train and test
X = data[["distance_miles", "traffic_level", "time_of_day"]]
y = data["delivery_minutes"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Train a simple model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Get predictions and calculate metrics
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

# Step 5: Log everything to MLflow
mlflow.set_experiment("delivery-time-baseline")

with mlflow.start_run():
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MAPE", mape)
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print("Logged to MLflow!")
