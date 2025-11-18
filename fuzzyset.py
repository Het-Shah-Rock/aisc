import numpy as np
import matplotlib.pyplot as plt


# Step 0: Print Problem Overview and Fuzzy Set Definitions

print("FUZZY MODEL: AIR QUALITY INDEX (AQI) vs HEALTH EFFECTS \n")
print("Fuzzy Sets for Air Quality (Set A):")
print("  1. Good (G)      → AQI 0–80      → Mostly Healthy (H)")
print("  2. Moderate (M)  → AQI 80–150    → May cause Mild Symptoms (MS)")
print("  3. Poor (P)      → AQI 150–220   → Likely Mild to Serious Symptoms (MS–SS)")
print("  4. Severe (S)    → AQI 220–300+  → Causes Serious Symptoms (SS)\n")

print("Fuzzy Sets for Health Effects (Set B):")
print("  Healthy (H)          → Minimal or no symptoms")
print("  Mild Symptoms (MS)   → Slight discomfort, coughing, fatigue")
print("  Serious Symptoms (SS)→ Breathlessness, respiratory distress\n")

print("Goal: Model how AQI affects human health using fuzzy sets, operations,")
print("and fuzzy compositions (Max-Min and Max-Product).\n")
print("-----------------------------------------------------------\n")

# Step 1: Define Universe and Fuzzy Sets for AQI
aqi = np.linspace(0, 300, 300)

# Fuzzy sets for AQI Levels
mu_good = np.exp(-((aqi - 50)**2) / (2 * 25**2))
mu_moderate = np.exp(-((aqi - 100)**2) / (2 * 25**2))
mu_poor = np.exp(-((aqi - 180)**2) / (2 * 30**2))
mu_severe = np.exp(-((aqi - 250)**2) / (2 * 20**2))

plt.figure(figsize=(8, 5))
plt.plot(aqi, mu_good, label="Good")
plt.plot(aqi, mu_moderate, label="Moderate")
plt.plot(aqi, mu_poor, label="Poor")
plt.plot(aqi, mu_severe, label="Severe")
plt.title("Fuzzy Sets for Air Quality Index (AQI)")
plt.xlabel("AQI Value")
plt.ylabel("Membership Value")
plt.legend()
plt.grid(True)
plt.show()

# Step 2: Define Fuzzy Sets for Health Effects
health = np.linspace(0, 1, 300)
mu_healthy = np.exp(-((health - 0.1)**2) / (2 * 0.1**2))
mu_mild = np.exp(-((health - 0.5)**2) / (2 * 0.1**2))
mu_serious = np.exp(-((health - 0.9)**2) / (2 * 0.1**2))

plt.figure(figsize=(8, 5))
plt.plot(health, mu_healthy, label="Healthy")
plt.plot(health, mu_mild, label="Mild Symptoms")
plt.plot(health, mu_serious, label="Serious Symptoms")
plt.title("Fuzzy Sets for Health Effects")
plt.xlabel("Health Condition (0-1 scale)")
plt.ylabel("Membership Value")
plt.legend()
plt.grid(True)
plt.show()

# Step 3: Fuzzy Operations between AQI sets
algebraic_sum = mu_good + mu_moderate - (mu_good * mu_moderate)
algebraic_product = mu_good * mu_moderate
bounded_sum = np.minimum(mu_good + mu_moderate, 1)
bounded_diff = np.maximum(mu_good - mu_moderate, 0)

plt.figure(figsize=(10, 6))
plt.plot(aqi, mu_good, 'b--', label="Good")
plt.plot(aqi, mu_moderate, 'r--', label="Moderate")
plt.plot(aqi, algebraic_sum, 'g', label="Algebraic Sum (Union)")
plt.plot(aqi, algebraic_product, 'm', label="Algebraic Product (Intersection)")
plt.plot(aqi, bounded_sum, 'c', label="Bounded Sum")
plt.plot(aqi, bounded_diff, 'y', label="Bounded Difference")
plt.title("Fuzzy Operations between AQI Sets")
plt.xlabel("AQI Value")
plt.ylabel("Membership Value")
plt.legend()
plt.grid(True)
plt.show()

# Step 4: Define Fuzzy Relation (AQI → Health Effects)
# Rows: AQI categories (G, M, P, S)
# Columns: Health Effects (H, MS, SS)
R = np.array([
    [0.9, 0.1, 0.0],  # Good Air → Mostly Healthy
    [0.5, 0.6, 0.2],  # Moderate → Mild Symptoms
    [0.2, 0.7, 0.6],  # Poor → Mild to Serious
    [0.0, 0.3, 0.9]   # Severe → Serious Symptoms
])

# Step 5: Define Desired Health Condition (Set C)
C = np.array([0.8, 0.5, 0.1])  # Prefer Healthy > Mild > Serious

# Step 6: Max-Min Composition (C o R)
max_min_comp = np.zeros(R.shape[0])
for i in range(R.shape[0]):
    max_min_comp[i] = np.max(np.minimum(C, R[i, :]))

# Step 7: Max-Product Composition
max_prod_comp = np.zeros(R.shape[0])
for i in range(R.shape[0]):
    max_prod_comp[i] = np.max(C * R[i, :])

# Step 8: Display composition results
labels = ["Good", "Moderate", "Poor", "Severe"]

plt.figure(figsize=(8, 5))
plt.bar(labels, max_min_comp, color='skyblue', label='Max-Min Composition')
plt.bar(labels, max_prod_comp, color='orange', alpha=0.7, label='Max-Product Composition')
plt.title("Fuzzy Composition Results (AQI → Health Condition)")
plt.ylabel("Membership Value")
plt.legend()
plt.show()

# Step 9: Print Interpretation
print("INTERPRETATION OF RESULTS:")
for i, label in enumerate(labels):
    print(f"{label} Air → Health Impact:")
    print(f"   Max-Min Composition: {max_min_comp[i]:.2f}")
    print(f"   Max-Product Composition: {max_prod_comp[i]:.2f}")
    if label == "Good":
        print("   → Mostly Healthy, minimal symptoms.\n")
    elif label == "Moderate":
        print("   → Some mild symptoms may occur.\n")
    elif label == "Poor":
        print("   → Health likely affected, mild to serious symptoms.\n")
    else:
        print("   → High risk of serious health issues.\n")
