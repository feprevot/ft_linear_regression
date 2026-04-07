import csv
import json
from math import sqrt


def load_data(filename):
    """Load evaluation data from a CSV file and return two lists: mileage (km) and price."""
    km = []
    price = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            km.append(float(row['km']))
            price.append(float(row['price']))
    return km, price


def normalize(values):
    """Min-max normalize a list of values to the [0, 1] range.
    Returns (normalized_values, min_value, max_value).
    """
    min_val = min(values)
    max_val = max(values)
    normalized = [(v - min_val) / (max_val - min_val) for v in values]
    return normalized, min_val, max_val


def load_model(filename):
    """Load trained parameters and scaling ranges from the JSON model file."""
    with open(filename, 'r') as f:
        data = json.load(f)
        return data['theta0'], data['theta1'], data['min_km'], data['max_km'], data['min_price'], data['max_price']


def evaluate_model(km_scaled, price_scaled, theta0, theta1):
    """Compute MSE and RMSE on the normalized dataset using the provided thetas."""
    m = len(km_scaled)
    total_error = 0.0

    for i in range(m):
        prediction = theta0 + theta1 * km_scaled[i]
        error = prediction - price_scaled[i]
        total_error += error ** 2

    mse = total_error / m
    rmse = sqrt(mse)
    return mse, rmse


if __name__ == "__main__":
    try:
        theta0, theta1, min_km, max_km, min_price, max_price = load_model("model/thetas.json")
    except Exception:
        print("Launch train.py first.")
        exit(1)

    km, price = load_data("data/data.csv")

    km_scaled = [(v - min_km) / (max_km - min_km) for v in km]
    price_scaled = [(v - min_price) / (max_price - min_price) for v in price]

    mse, rmse = evaluate_model(km_scaled, price_scaled, theta0, theta1)
    print(f"Model evaluation: MSE = {mse:.4f}, RMSE = {rmse:.4f}")
