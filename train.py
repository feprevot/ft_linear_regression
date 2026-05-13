import csv
import json
from utils.plot import plot_data_with_regression

LEARNING_RATE = 0.1
EPOCHS = 1000

def load_data(filename):
    """Load training data from a CSV file and return two lists: mileage (km) and price."""
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

def predict(km, theta0, theta1):
    """Compute the model prediction (in scaled space): y = theta0 + theta1 * x."""
    return theta0 + (theta1 * km)

def train(km, price):
    """Train theta0 and theta1 using gradient descent on the normalized dataset.
    Returns the learned (theta0, theta1) in scaled space.
    """
    m = len(km)
    theta0 = 0.0
    theta1 = 0.0

    for _ in range(EPOCHS):
        sum_error0 = 0.0
        sum_error1 = 0.0

        for i in range(m):
            prediction = predict(km[i], theta0, theta1)
            error = prediction - price[i]
            sum_error0 += error
            sum_error1 += error * km[i]

        grad_theta0 = LEARNING_RATE * (sum_error0 / m)
        grad_theta1 = LEARNING_RATE * (sum_error1 / m)

        theta0 -= grad_theta0
        theta1 -= grad_theta1
    return theta0, theta1

def save_model(theta0, theta1, min_km, max_km, min_price, max_price, filename):
    """Persist the trained parameters and the normalization ranges to a JSON file.
    This allows predict/evaluate scripts to apply the same scaling.
    """
    with open(filename, 'w') as f:
        json.dump({
            'theta0': theta0,
            'theta1': theta1,
            'min_km': min_km,
            'max_km': max_km,
            'min_price': min_price,
            'max_price': max_price
        }, f)

if __name__ == "__main__":
    km, price = load_data("data/data.csv")
    km_scaled, min_km, max_km = normalize(km)
    price_scaled, min_price, max_price = normalize(price)

    theta0, theta1 = train(km_scaled, price_scaled)

    save_model(theta0, theta1, min_km, max_km, min_price, max_price, "model/thetas.json")
    print(f"Training complete. theta0 = {theta0}, theta1 = {theta1}")
    plot_data_with_regression(km, price, theta0, theta1, min_km, max_km, min_price, max_price)
