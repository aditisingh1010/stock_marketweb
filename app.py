from flask import Flask
import joblib
import pandas as pd

app = Flask(__name__)

# load trained model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    data = pd.read_csv('data/stock_data.csv')  #
    last_day = len(data)
    last_close = data['Close'].iloc[-1]

    prediction = model.predict([[last_day, last_close]])[0]

    return f"""
    <h1>Stock Prediction System</h1>
    <p><b>Next Day Prediction:</b> {round(prediction,2)}</p>
    <a href='/metrics'>View Model Metrics</a>
    """

@app.route('/metrics')
def metrics():
    data = pd.read_csv('data/stock_data.csv')  # same yaha bhi
    data['Day'] = range(len(data))
    data['Lag1'] = data['Close'].shift(1)
    data = data.dropna()

    X = data[['Day', 'Lag1']]
    y = data['Close']

    score = model.score(X, y)

    return f"<h2>Model R² Score: {round(score,4)}</h2>"

if __name__ == "__main__":
    app.run(debug=True)