# ft_linear_regression — Car Price Estimator

A **42** project. It implements a **simple linear regression** *from scratch* to
estimate the **price of a car** from its **mileage** (`km`). Gradient descent is
written by hand; the only third-party dependency is `matplotlib`, used to plot
the result.

The model learns the line:

```
price = theta0 + theta1 × mileage
```

`train.py` finds `theta0` and `theta1` from historical data, then `predict.py`
uses them to estimate the price of any given mileage.

---

## Table of contents

- [Project structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
  - [Train](#train)
  - [Predict](#predict)
- [How it works](#how-it-works)
- [The dataset](#the-dataset)
- [Hyperparameters](#hyperparameters)
- [Glossary](#glossary)

---

## Project structure

```
ft_linear_regression/
├── train.py             # Trains the model with gradient descent → model/thetas.json
├── predict.py           # Asks for a mileage and prints the estimated price
├── utils/
│   └── plot.py          # Scatter plot of the data + the learned regression line
├── data/
│   └── data.csv         # Training data (columns: km, price)
├── requirements.txt
└── README.md
```

The trained parameters are written to `model/thetas.json` (see
[Notes](#notes) — this folder must exist before training).

---

## Setup

Python 3, with a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Only dependency: `matplotlib`.

---

## Usage

### Train

```bash
mkdir -p model          # required: train.py writes to model/thetas.json
python3 train.py
```

`train.py` reads `data/data.csv`, trains the parameters, prints the learned
`theta0` / `theta1`, saves them (with the normalization ranges) to
`model/thetas.json`, and opens a plot showing the data points (blue) and the
fitted regression line (red).

### Predict

```bash
python3 predict.py
# car mileage : 50000
# estimated price : 6500.00 €
```

`predict.py` loads the model, prompts for a mileage, and prints the estimated
price.

> If `model/thetas.json` does not exist yet (model never trained), `predict.py`
> falls back to `theta0 = theta1 = 0` and therefore returns `0.00 €`. Train
> first to get a meaningful estimate.

---

## How it works

**1. Load the data** — `data/data.csv` is read into two lists, `km` and `price`.

**2. Normalize (min-max)** — both mileage and price are rescaled to the `[0, 1]`
range:

```
x' = (x − min) / (max − min)
```

Mileage values (tens of thousands) and prices (thousands) live on very different
scales; normalizing keeps gradient descent stable and lets a single learning
rate work for both parameters. The `min`/`max` of each column are saved so the
exact same scaling can be re-applied at prediction time.

**3. Gradient descent** (`train.py`) — starting from `theta0 = theta1 = 0`, for
each epoch the average error over the dataset is used to update both parameters:

```
error      = (theta0 + theta1 × km) − price
theta0    -= learning_rate × mean(error)
theta1    -= learning_rate × mean(error × km)
```

**4. Save** — `theta0`, `theta1` and the four normalization bounds
(`min_km`, `max_km`, `min_price`, `max_price`) go into `model/thetas.json`.

**5. Predict** (`predict.py`) — the input mileage is normalized with the stored
bounds, the line is applied in scaled space, then the result is **un-normalized**
back to euros:

```
scaled_price    = theta0 + theta1 × normalized_km
estimated_price = min_price + scaled_price × (max_price − min_price)
```

---

## The dataset

`data/data.csv` contains 24 `(km, price)` pairs with a header row:

```
km,price
240000,3650
139800,3800
...
```

As expected for used cars, price decreases as mileage increases, so the learned
`theta1` (slope) is negative.

---

## Hyperparameters

Defined at the top of `train.py`:

| Constant | Value | Role |
|---|---|---|
| `LEARNING_RATE` | `0.1` | step size of each gradient-descent update |
| `EPOCHS` | `1000` | number of full passes over the dataset |

---

## Glossary

| Term | Definition |
|------|-----------|
| **θ0 (theta0)** | Intercept of the regression line — the base price (in scaled space) when mileage is 0. |
| **θ1 (theta1)** | Slope of the line — how much the price changes per unit of mileage. Negative here: more km → lower price. |
| **Linear regression** | Models the relationship between two variables as a straight line: `price = θ0 + θ1 × mileage`. |
| **Gradient descent** | Iterative algorithm that nudges θ0 and θ1 step by step to minimize the prediction error. |
| **Learning rate** | Size of each gradient-descent step. Too high → diverges; too low → converges slowly. |
| **Epochs** | Number of full passes over the training dataset. |
| **Normalization** | Rescaling values to `[0, 1]` so mileage and price share the same scale during training. |
| **Error** | Difference between the predicted and the actual price for a data point: `prediction − price`. |
