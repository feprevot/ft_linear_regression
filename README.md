# Linear Regression – Car Price Estimator

This project implements a **simple linear regression** to estimate the **price of a car** based on its **mileage** (`km`). It is structured around two main programs:

1. `train.py` – Trains the model using historical data.
2. `predict.py` – Predicts the price of a car for a given mileage.
3. `evaluate.py` – Evaluates the trained model (MSE / RMSE).

---

## Project Structure

linear_regression_project/
├── data/
│ └── data.csv # Training data (km, price)
├── model/
│ └── thetas.json # File containing learned parameters
├── train.py # Training script
├── predict.py # Prediction script
├── evaluate.py # Evaluation script
├── utils/
│ └── plot.py # Graph plotting (data + regression line)
└── README.md # This file


---

## How It Works

###  Linear Regression

The model tries to find a simple linear relationship between mileage and price:

\[
\text{price} = \theta_0 + \theta_1 \times \text{mileage}
\]

To improve stability and training efficiency, both **mileage** and **price** are **normalized** to a 0–1 scale before training.

###  Training (`train.py`)

- Reads `data/data.csv`
- Applies **gradient descent** to learn parameters `theta0` and `theta1`
- Saves the result in `model/thetas.json`
- Plots:
  - Data points (blue)
  - Learned regression line (red)

###  Evaluation (`evaluate.py`)

- Loads `model/thetas.json`
- Re-loads `data/data.csv`
- Computes evaluation metrics:
  - **MSE**: Mean Squared Error
  - **RMSE**: Root Mean Squared Error

###  Prediction (`predict.py`)

- Loads the learned model and normalization values
- Asks the user to input a mileage value
- Normalizes the input
- Applies the model to estimate the price
- Unnormalizes the output and prints the estimated price

---

## Run with a virtual environment (venv)
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -r requirements.txt`
---

## Glossary

| Term | Definition |
|------|-----------|
| **θ0 (theta0)** | Y-intercept of the regression line — the base price when mileage is 0. |
| **θ1 (theta1)** | Slope of the regression line — how much the price changes per unit of mileage. Negative here: more km → lower price. |
| **Linear Regression** | A method that models the relationship between two variables as a straight line: `price = θ0 + θ1 × mileage`. |
| **Gradient Descent** | An iterative algorithm that adjusts θ0 and θ1 step by step to minimize prediction error. |
| **Learning Rate** | Controls the size of each step during gradient descent. Too high → diverges. Too low → converges slowly. |
| **Epochs** | The number of full passes over the training dataset. More epochs → more refined parameters. |
| **Normalization** | Rescaling values to a [0, 1] range so that mileage and price are on the same scale during training. |
| **Error** | Difference between the predicted price and the actual price for a given data point: `prediction − price`. |

