# Fuzzy Logic Controller for Traffic Signal

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# 1. Define Input and Output Variables

# Input 1: Traffic Density (vehicles per minute)
traffic_density = ctrl.Antecedent(np.arange(0, 51, 1), 'traffic_density')

# Input 2: Waiting Time (seconds)
waiting_time = ctrl.Antecedent(np.arange(0, 121, 1), 'waiting_time')

# Output: Green Light Duration (seconds)
green_duration = ctrl.Consequent(np.arange(10, 91, 1), 'green_duration')


# 2. Define Membership Functions

# Traffic Density
traffic_density['low'] = fuzz.trapmf(traffic_density.universe, [0, 0, 10, 15])
traffic_density['medium'] = fuzz.trimf(traffic_density.universe, [10, 20, 30])
traffic_density['high'] = fuzz.trapmf(traffic_density.universe, [25, 35, 50, 50])

# Waiting Time
waiting_time['short'] = fuzz.trapmf(waiting_time.universe, [0, 0, 15, 30])
waiting_time['moderate'] = fuzz.trimf(waiting_time.universe, [20, 50, 80])
waiting_time['long'] = fuzz.trapmf(waiting_time.universe, [70, 90, 120, 120])

# Green Light Duration
green_duration['short'] = fuzz.trimf(green_duration.universe, [10, 20, 30])
green_duration['medium'] = fuzz.trimf(green_duration.universe, [30, 45, 60])
green_duration['long'] = fuzz.trimf(green_duration.universe, [60, 75, 90])


# 3. Define Fuzzy Rules

rule1 = ctrl.Rule(traffic_density['low'] & waiting_time['short'], green_duration['short'])
rule2 = ctrl.Rule(traffic_density['low'] & waiting_time['moderate'], green_duration['medium'])
rule3 = ctrl.Rule(traffic_density['low'] & waiting_time['long'], green_duration['medium'])

rule4 = ctrl.Rule(traffic_density['medium'] & waiting_time['short'], green_duration['medium'])
rule5 = ctrl.Rule(traffic_density['medium'] & waiting_time['moderate'], green_duration['medium'])
rule6 = ctrl.Rule(traffic_density['medium'] & waiting_time['long'], green_duration['long'])

rule7 = ctrl.Rule(traffic_density['high'] & waiting_time['short'], green_duration['medium'])
rule8 = ctrl.Rule(traffic_density['high'] & waiting_time['moderate'], green_duration['long'])
rule9 = ctrl.Rule(traffic_density['high'] & waiting_time['long'], green_duration['long'])


# 4. Create Control System
traffic_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
traffic_sim = ctrl.ControlSystemSimulation(traffic_ctrl)


# 5. Test the System for Different Inputs

# Example 1: Low density + Short waiting
traffic_sim.input['traffic_density'] = 5
traffic_sim.input['waiting_time'] = 10
traffic_sim.compute()
print(f"\nCase 1: Low Density + Short Waiting → Green Duration = {traffic_sim.output['green_duration']:.2f} sec")

# Example 2: Medium density + Moderate waiting
traffic_sim.input['traffic_density'] = 20
traffic_sim.input['waiting_time'] = 50
traffic_sim.compute()
print(f"Case 2: Medium Density + Moderate Waiting → Green Duration = {traffic_sim.output['green_duration']:.2f} sec")

# Example 3: High density + Long waiting
traffic_sim.input['traffic_density'] = 40
traffic_sim.input['waiting_time'] = 100
traffic_sim.compute()
print(f"Case 3: High Density + Long Waiting → Green Duration = {traffic_sim.output['green_duration']:.2f} sec")


# 6. Visualization
# Membership functions and results visualization

traffic_density.view()
waiting_time.view()
green_duration.view(sim=traffic_sim)
