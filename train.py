import csv
import json
from utils.plot import plot_data_with_regression
from math import sqrt

LEARNING_RATE = 0.1
EPOCHS = 1000

def load_data(filename):
    km = []
    price = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            km.append(float(row['km']))
            price.append(float(row['price']))
    return km, price


def evaluate_model(km_scaled, price_scaled, theta0, theta1):
    m = len(km_scaled)
    total_error = 0.0

    for i in range(m):
        prediction = theta0 + theta1 * km_scaled[i]
        error = prediction - price_scaled[i]
        total_error += error ** 2

    mse = total_error / m
    rmse = sqrt(mse)
    return mse, rmse



def normalize(values):
    min_val = min(values)
    max_val = max(values)
    normalized = [(v - min_val) / (max_val - min_val) for v in values]
    return normalized, min_val, max_val

def predict(km, theta0, theta1):
    return theta0 + theta1 * km

def train(km, price):
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

        tmp_theta0 = LEARNING_RATE * (sum_error0 / m)
        tmp_theta1 = LEARNING_RATE * (sum_error1 / m)

        theta0 -= tmp_theta0
        theta1 -= tmp_theta1

    return theta0, theta1

def save_model(theta0, theta1, min_km, max_km, min_price, max_price, filename):
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
    mse, rmse = evaluate_model(km_scaled, price_scaled, theta0, theta1)
    print(f"Model evaluation: MSE = {mse:.4f}, RMSE = {rmse:.4f}")
    plot_data_with_regression(km, price, theta0, theta1, min_km, max_km, min_price, max_price)
