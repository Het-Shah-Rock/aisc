# ANFIS 5-LAYER IMPLEMENTATION (Rectified & Stable)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. SYNTHETIC DATASET FOR MUMBAI SUMMER
np.random.seed(42)
N = 600

Temperature = np.random.uniform(28, 38, N)
Humidity = np.random.uniform(55, 90, N)
Occupancy = np.random.randint(1, 6, N)
Appliances = np.random.randint(3, 20, N)
Hour = np.random.randint(0, 24, N)

Energy = (
    0.14*Temperature +
    0.08*Humidity +
    0.25*Occupancy +
    0.35*Appliances +
    0.6*np.where((Hour>=18)|(Hour<=7),1,0) +
    np.random.normal(0, 1.3, N)
)

df = pd.DataFrame({"Temperature": Temperature,
                   "Humidity": Humidity,
                   "Occupancy": Occupancy,
                   "Appliances": Appliances,
                   "Hour": Hour,
                   "Energy": Energy})

X = df[['Temperature','Humidity','Occupancy','Appliances','Hour']].values
y = df['Energy'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# 2. MEMBERSHIP FUNCTIONS (LAYER 1)
x_temp = np.arange(20, 45, 1)
temp_low  = fuzz.trimf(x_temp,[20,20,30])
temp_med  = fuzz.trimf(x_temp,[28,33,37])
temp_high = fuzz.trimf(x_temp,[34,40,45])

x_hum = np.arange(40,100,1)
hum_low  = fuzz.trimf(x_hum,[40,50,65])
hum_med  = fuzz.trimf(x_hum,[60,72,80])
hum_high = fuzz.trimf(x_hum,[75,90,100])


# FUZZIFICATION + LAYER 2 RULE FIRING
def fuzzify(T, H):
    mu_TL = fuzz.interp_membership(x_temp, temp_low, T)
    mu_TM = fuzz.interp_membership(x_temp, temp_med, T)
    mu_TH = fuzz.interp_membership(x_temp, temp_high, T)

    mu_HL = fuzz.interp_membership(x_hum, hum_low, H)
    mu_HM = fuzz.interp_membership(x_hum, hum_med, H)
    mu_HH = fuzz.interp_membership(x_hum, hum_high, H)

    w1 = mu_TL * mu_HL
    w2 = mu_TM * mu_HM
    w3 = mu_TH * mu_HH
    return np.array([w1, w2, w3])


# BUILD TRAIN MATRIX FOR LSE (Layer 3, 4)
rule_outputs = []

for i in range(len(X_train)):
    w = fuzzify(X_train[i][0], X_train[i][1])
    s = np.sum(w)

    # Avoid division by zero
    if s == 0:
        w = np.array([1/3, 1/3, 1/3])
    else:
        w = w / s

    # Layer 4 linear contribution
    row = np.hstack([(w[j] * np.append(X_train[i], 1)) for j in range(3)])
    rule_outputs.append(row)

rule_outputs = np.nan_to_num(np.array(rule_outputs))   # Fix NaN/inf


# HYBRID LEARNING (LSE replaced by Ridge Regression)
reg = Ridge(alpha=0.01)
reg.fit(rule_outputs, y_train)
coeffs = reg.coef_


# LAYER 5 PREDICTION
def anfis_predict(x):
    w = fuzzify(x[0], x[1])
    s = np.sum(w)
    if s == 0:
        w = np.array([1/3,1/3,1/3])
    else:
        w = w / s

    fv = np.hstack([(w[j] * np.append(x,1)) for j in range(3)])
    return np.dot(fv, coeffs)

y_pred = np.array([anfis_predict(x) for x in X_test])


# MODEL EVALUATION
print("\nEvaluation Metrics:")
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAE :", mean_absolute_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Energy")
plt.ylabel("Predicted Energy")
plt.title("ANFIS 5-Layer Output: Actual vs Predicted")
plt.grid(True)
plt.show()
