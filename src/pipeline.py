import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Load data
data = pd.read_csv('data/stock_data.csv')
# Feature engineering
data['Day'] = range(len(data))
data['Lag1'] = data['Close'].shift(1)
data = data.dropna()

X = data[['Day', 'Lag1']]
y = data['Close']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = model.score(X_test, y_test)

print("MSE:", mse)
print("R2:", r2)

# Save model
joblib.dump(model, 'model.pkl')
print("Model saved")