# Linear Regression – Car Price Estimator

This project implements a **simple linear regression** to estimate the **price of a car** based on its **mileage** (`km`). It is structured around two main programs:

1. `train.py` – Trains the model using historical data.
2. `predict.py` – Predicts the price of a car for a given mileage.

---

## Project Structure

linear_regression_project/
├── data/
│ └── data.csv # Training data (km, price)
├── model/
│ └── thetas.json # File containing learned parameters
├── train.py # Training script
├── predict.py # Prediction script
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
- Computes evaluation metrics (MSE, RMSE)

###  Prediction (`predict.py`)

- Loads the learned model and normalization values
- Asks the user to input a mileage value
- Normalizes the input
- Applies the model to estimate the price
- Unnormalizes the output and prints the estimated price

---

##  Implemented Bonus Features

-  Plotting the data to visualize the distribution
-  Plotting the regression line over the data
-  A program that calculates the **accuracy** of the algorithm:
  - **MSE**: Mean Squared Error
  - **RMSE**: Root Mean Squared Error

---

##  Key Concepts

###  MSE – Mean Squared Error

Measures how far off the predictions are from the actual prices, on average:

\[
\text{MSE} = \frac{1}{m} \sum_{i=1}^{m} (\text{prediction}_i - \text{actual}_i)^2
\]

Lower is better. A value of 0 means perfect prediction.

### 🔹 RMSE – Root Mean Squared Error

The square root of MSE. It has the **same unit as the predicted variable** (price), so it's easier to interpret.

\[
\text{RMSE} = \sqrt{\text{MSE}}
\]

---

##  Hyperparameters

###  Learning Rate (`LEARNING_RATE`)
- Controls how fast the model updates during training
- Too high → unstable / diverges
- Too low → slow learning

###  Epochs (`EPOCHS`)
- The number of times the entire dataset is used during training
- More epochs = better convergence (but slower)

---

