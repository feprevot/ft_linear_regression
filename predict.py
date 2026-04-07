import json

def predict(km, theta0, theta1):
    """Compute the scaled prediction: y = theta0 + theta1 * x."""
    return theta0 + theta1 * km

def normalize(value, min_val, max_val):
    """Min-max normalize a single value to the [0, 1] range."""
    return (value - min_val) / (max_val - min_val)

def unnormalize(value, min_val, max_val):
    """Convert a [0, 1] scaled value back to the original range."""
    return min_val + value * (max_val - min_val)

def load_model(filename):
    """Load the trained model parameters and scaling ranges from a JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
        return data['theta0'], data['theta1'], data['min_km'], data['max_km'], data['min_price'], data['max_price']

if __name__ == "__main__":
    try:
        theta0, theta1, min_km, max_km, min_price, max_price = load_model("model/thetas.json")
    except:
        print("Launch train.py first.")
        exit()

    try:
        user_input = input("car mileage : ")
        km = float(user_input)
    except ValueError:
        print("bad entry.")
        exit()

    normalized_km = normalize(km, min_km, max_km)
    scaled_price = predict(normalized_km, theta0, theta1)
    estimated_price = unnormalize(scaled_price, min_price, max_price)

    print(f"estimated price : {estimated_price:.2f} €")
