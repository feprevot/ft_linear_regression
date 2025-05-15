import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def plot_data_with_regression(km, price, theta0, theta1, min_km, max_km, min_price, max_price):
    km_line = list(range(int(min_km), int(max_km) + 10000, 1000))
    normalized_km_line = [(x - min_km) / (max_km - min_km) for x in km_line]

    predicted_scaled = [theta0 + theta1 * x for x in normalized_km_line]
    predicted_prices = [min_price + p * (max_price - min_price) for p in predicted_scaled]

    plt.scatter(km, price, color='blue', label='Data points')
    plt.plot(km_line, predicted_prices, color='red', label='Linear regression')

    plt.xlabel('Mileage (km)')
    plt.ylabel('Price (€)')
    plt.title('Linear regression - Price vs Mileage')
    plt.legend()
    plt.grid(True)
    plt.show()
