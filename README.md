# Flood Forecasting as a Least Squares Problem

When a flood wave hits an upstream gauge on the Tar River in North Carolina, it takes 1–4 days to reach the downstream target gauge at Wilson, NC. This project asks: if we build a matrix from upstream observations and solve a least squares problem, can we predict floods days in advance — and how wrong will we be?

---

## The Core Idea

We have 31 years of daily streamflow and precipitation records across 8 river gauges. The goal is to predict flow at the target gauge $k$ days into the future using what we observe today across the network.

This maps directly onto a linear system. Stack one row per day into a design matrix $A$, put tomorrow's (or day $k$'s) flow into a vector $b$, and solve:

$$\min_x \|Ax - b\|^2$$

The solution $\hat{x} = (A^TA)^{-1}A^Tb$ gives us the weights. Multiplied by tomorrow's features, it gives us a forecast.

---

## Data

**Source:** USGS streamflow + Daymet precipitation, Tar River NC, 1993–2024.

8 gauges total — 7 upstream, 1 target. Flood = flow above 8,641 cfs (95th percentile of training data). About 5% of days are flood days.

Routing lags from NHDPlus (how long water takes to travel between gauges):

```
02081500  →  4 days upstream
02081747  →  3 days
02082770  →  3 days
02082950  →  3 days
02083000  →  3 days
02082585  →  1 day
02083500  →  1 day upstream
■ 02084000  ←  target
```

The max lag of 4 days means we build 4 separate forecasting problems: predict 1, 2, 3, and 4 days ahead.

---

## Building the Design Matrix

Each row of $A$ is one day. Each column is a feature. We use 25 features:

- **4 local:** log-flow at target, daily precip, 7-day rolling precip, 14-day rolling precip
- **7 upstream:** log-flow at each upstream gauge, observed today (no lag shift — the matrix learns the routing from temporal patterns)
- **14 basin precip:** rainfall between each consecutive gauge pair, daily and 3-day rolling, shifted 1 day for runoff delay

All features are log-transformed and z-score normalized (fit on training data only). Target vector $b$ is log-flow $k$ days ahead.

Training matrix is roughly 7,000 × 26 (25 features + bias). We solve four systems — one per horizon.

---

## Two Ways to Solve It

We compare three approaches to solving $Ax \approx b$, which is the computational linear algebra part of this project:

**1. Normal equations**
Form $A^TA$ (26×26), then solve $(A^TA)\hat{x} = A^Tb$. Fast, but squaring the matrix doubles the condition number. Numerically fragile if upstream gauges are correlated.

**2. QR factorization**
Factor $A = QR$, solve $R\hat{x} = Q^Tb$. More stable — avoids forming $A^TA$. This is what `numpy.linalg.lstsq` uses internally.

---

## Uncertainty: Residual Analysis

Once we have $\hat{x}$, we compute residuals on the validation set (2017–2020):

$$r_i = \log(\text{observed}_i) - \log(\text{predicted}_i)$$

Positive residual = model underpredicted. Negative = overpredicted. We look at:

- Distribution of $r$ — is the model systematically biased? (It is, upward during floods)
- How $\|r\|$ grows with forecast horizon — uncertainty increasing from $t+1$ to $t+4$
- Quantiles of $r$ as simple prediction bounds: take the 5th and 95th percentile of validation residuals, add them to any new prediction, get an interval

No distributional assumptions. Just: here is the empirical spread of past errors. Assume future errors look similar.

---

## What We Compare

| Model | Features | Method |
|---|---|---|
| Persistence baseline | none | $\hat{Q}(t+k) = Q(t)$ |
| Local linear | 4 local | Least squares |
| Upstream linear | 25 (local + upstream + precip) | Least squares |

Three forecasting horizons of interest: $t+1$ (easy), $t+2$, $t+4$ (hard). Does the upstream model beat persistence at 4-day lead? That's the question.

---


# 18.0851 Final Project — Predicting River Floods with Least Squares

## What I'm doing

I'm predicting how high a river will get a few days from now, using sensor readings from today.

The river I picked (Tar River, North Carolina) has 8 water-level sensors along it. Water flows past the upstream sensors first and reaches the last sensor (my target) a few days later. So if today's readings at the upstream sensors are unusually high, the target sensor will likely be high in a few days.

I want a formula that turns "today's 25 sensor readings" into "predicted water level 4 days from now."

## How this becomes a math problem

I have 31 years of daily history — about 7,000 days. For each day, I know:
- The 25 sensor readings on that day → these become a row of a matrix $A$
- What the target sensor read 4 days later → this becomes an entry of a vector $b$

Stack everything together:
- $A$ is a $7{,}000 \times 26$ matrix (25 features + a constant column for the intercept)
- $b$ is a vector of length $7{,}000$

I want to find weights $x \in \mathbb{R}^{26}$ such that $Ax \approx b$. There's no exact solution (way more equations than unknowns), so I solve the least squares problem:

$$\min_{x} \; \|Ax - b\|^2$$

## The math I actually use

1. **Normal equations:** $A^TA \, \hat{x} = A^T b$, derived by setting the gradient of $\|Ax-b\|^2$ to zero
2. **QR factorization:** $A = QR$, then solve $R\hat{x} = Q^Tb$ — more numerically stable than normal equations
3. **Weighted least squares:** I care more about flood days than normal days, so I solve $A^TWA\,\hat{x} = A^TWb$ with a diagonal weight matrix $W$
4. **Condition number analysis:** $\kappa(A) = \sigma_{\max}/\sigma_{\min}$ from the singular values — tells me how much measurement noise gets amplified
5. **Residuals:** $r_i = b_i - a_i^T\hat{x}$ on validation data — I use the spread of these to build prediction intervals around each forecast

## Why this is interesting mathematically

- The sensors are correlated (nearby sensors tend to move together) → the columns of $A$ are nearly linearly dependent → high condition number → motivates using QR over the normal equations
- The data is unbalanced (flood days are 5% of the data, but they're the ones I care about) → standard least squares ignores them → motivates weighted least squares
- I compare three different solvers on the same problem and check they agree to machine precision

## Files

```
data/                              # 31 years of preprocessed sensor data
notebooks/linear_least_squares.ipynb   # all the math, in code
```

## Reproducing

Open the notebook and run all cells. It loads the parquet files in `data/`, builds the design matrix, solves the system three different ways, and saves the resulting predictions plus interval coverage statistics.
