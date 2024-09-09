import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

###### Constants defining metabolite pools ######
# Volume fractions and water space fractions
V_c = 1.0  # buffer volume fraction        # L buffer (L cuvette)**(-1)
V_m = 0.0005  # mitochondrial volume fraction # L mito (L cuvette)**(-1)
V_m2c = V_m / V_c  # mito to cyto volume ratio     # L mito (L cuvette)**(-1)
W_c = 1.0  # buffer water space            # L buffer water (L buffer)**(-1)
W_m = 0.7238  # mitochondrial water space     # L mito water (L mito)**(-1)
W_x = 0.9 * W_m  # matrix water space            # L matrix water (L mito)**(-1)
W_i = 0.1 * W_m  # intermembrane water space     # L IM water (L mito)**(-1)

# Total pool concentrations
NAD_tot = 2.97e-3  # NAD+ and NADH conc            # mol (L matrix water)**(-1)
Q_tot = 1.35e-3  # Q and QH2 conc                # mol (L matrix water)**(-1)
c_tot = 2.7e-3  # cytochrome c ox and red conc  # mol (L IM water)**(-1)

# Membrane capacitance
Cm = 3.1e-3

###### Set fixed pH, cation concentrations, and O2 partial pressure ######
# pH
pH_x = 7.40
pH_c = 7.20

# K+ concentrations
K_x = 100e-3  # mol (L matrix water)**(-1)
K_c = 140e-3  # mol (L cyto water)**(-1)

# Mg2+ concentrations
Mg_x = 1.0e-3  # mol (L matrix water)**(-1)
Mg_c = 1.0e-3  # mol (L cyto water)**(-1)

# Oxygen partial pressure
PO2 = 25  # mmHg

conc = np.array([pH_x, pH_c, K_x, K_c, Mg_x, Mg_c, PO2])

###### Parameter vector ######
X_DH = 0.1732
X_C1 = 1.0e4
X_C3 = 1.0e6
X_C4 = 0.0125
X_F = 1.0e3
E_ANT = 0.325
E_PiC = 5.0e6
X_H = 1.0e3
X_AtC = 0


def dXdt(t, X, activity_array, solve_ode, phosphate_control):
    # Unpack variables
    DPsi, sumATP_x, sumADP_x, sumPi_x, NADH_x, QH2_x, cred_i, sumATP_c, sumADP_c = X

    if phosphate_control:  # phosphate control
        X_DH, X_C1, X_C3, X_C4, X_F, E_ANT, E_PiC, X_H, X_AtC, sumPi_c, k_Pi3, k_Pi4 = activity_array
        r_coeffiecient = 4.5807 / 4.2530
        k_Pi1_coefficient = 0.13413 / 0.13890
        k_Pi2_coefficient = 0.67668 / 0.62396
    else:  # without phosphate control
        X_DH, X_C1, X_C3, X_C4, X_F, E_ANT, E_PiC, X_H, X_AtC, sumPi_c = activity_array
        r_coeffiecient = 1
        k_Pi1_coefficient = 1
        k_Pi2_coefficient = 1
        k_Pi3 = 1
        k_Pi4 = 1

    # Hydrogen ion concentration
    H_x = 10 ** (-pH_x)  # mol (L matrix water)**(-1)
    H_c = 10 ** (-pH_c)  # mol (L cuvette water)**(-1)

    # Oxygen concentration
    a_3 = 1.74e-6  # oxygen solubility in cuvette   # mol (L matrix water * mmHg)**(-1)
    O2_x = a_3 * PO2  # mol (L matrix water)**(-1)

    # Thermochemical constants
    R = 8.314  # J (mol K)**(-1)
    T = 37 + 273.15  # K
    F = 96485  # C mol**(-1)

    # Proton motive force parameters (dimensionless)
    n_F = 8 / 3
    n_C1 = 4
    n_C3 = 2
    n_C4 = 4

    # Dissociation constants
    K_MgATP = 10 ** (-3.88)
    K_HATP = 10 ** (-6.33)
    K_KATP = 10 ** (-1.02)
    K_MgADP = 10 ** (-3.00)
    K_HADP = 10 ** (-6.26)
    K_KADP = 10 ** (-0.89)
    K_MgPi = 10 ** (-1.66)
    K_HPi = 10 ** (-6.62)
    K_KPi = 10 ** (-0.42)

    ## Other concentrations computed from the state variables:
    NAD_x = NAD_tot - NADH_x  # mol (L matrix water)**(-1)
    Q_x = Q_tot - QH2_x  # mol (L matrix water)**(-1)
    cox_i = c_tot - cred_i  # mol (L matrix water)**(-1)

    ## Binding polynomials
    # Matrix species # mol (L mito water)**(-1)
    PATP_x = 1 + H_x / K_HATP + Mg_x / K_MgATP + K_x / K_KATP
    PADP_x = 1 + H_x / K_HADP + Mg_x / K_MgADP + K_x / K_KADP
    PPi_x = 1 + H_x / K_HPi + Mg_x / K_MgPi + K_x / K_KPi

    # Cytosol species # mol (L cuvette water)**(-1)
    PATP_c = 1 + H_c / K_HATP + Mg_c / K_MgATP + K_c / K_KATP
    PADP_c = 1 + H_c / K_HADP + Mg_c / K_MgADP + K_c / K_KADP
    PPi_c = 1 + H_c / K_HPi + Mg_c / K_MgPi + K_c / K_KPi

    ## Unbound species
    # Matrix species
    ATP_x = sumATP_x / PATP_x  # [ATP4-]_x
    ADP_x = sumADP_x / PADP_x  # [ADP3-]_x
    Pi_x = sumPi_x / PPi_x  # [HPO42-]_x

    # Cytosol species
    ATP_c = sumATP_c / PATP_c  # [ATP4-]_c
    ADP_c = sumADP_c / PADP_c  # [ADP3-]_c
    Pi_c = sumPi_c / PPi_c  # [HPO42-]_c

    ###### NADH Dehydrogenase ######
    # Constants
    r = 6.8385 * r_coeffiecient

    k_Pi1 = 4.659e-4 * k_Pi1_coefficient  # mol (L matrix water)**(-1)
    k_Pi2 = 6.578e-4 * k_Pi2_coefficient  # mol (L matrix water)**(-1)

    # Flux
    J_DH = X_DH * (r * NAD_x - NADH_x) * ((1 + sumPi_x / k_Pi1) / (1 + sumPi_x / k_Pi2))

    ###### Complex I ######
    # NADH_x + Q_x + 5H+_x <-> NAD+_x + QH2_x + 4H+_i + 4DPsi

    # Gibbs energy (J mol**(-1))
    DrGo_C1 = -109680
    DrGapp_C1 = DrGo_C1 - R * T * np.log(H_x)

    # Apparent equilibrium constant
    Kapp_C1 = np.exp(-(DrGapp_C1 + n_C1 * F * DPsi) / (R * T)) * ((H_x / H_c) ** n_C1)

    # Flux (mol (s * L mito)**(-1))
    J_C1 = X_C1 * (Kapp_C1 * NADH_x * Q_x - NAD_x * QH2_x)

    ###### Complex III ######
    # QH2_x + 2cuvetteC(ox)3+_i + 2H+_x <-> Q_x + 2cuvetteC(red)2+_i + 4H+_i + 2DPsi

    # Gibbs energy (J mol**(-1))
    DrGo_C3 = 46690
    DrGapp_C3 = DrGo_C3 + 2 * R * T * np.log(H_c)

    # Apparent equilibrium constant
    Kapp_C3 = np.exp(-(DrGapp_C3 + n_C3 * F * DPsi) / (R * T)) * (H_x / H_c) ** n_C3

    # Flux (mol (s * L mito)**(-1))
    # J_C3 = X_C3 * (Kapp_C3 * cox_i ** 2 * QH2_x - cred_i ** 2 * Q_x) # original implementation
    J_C3 = X_C3 * ((1 + PPi_x / k_Pi3) / (1 + PPi_x / k_Pi4)) * (
            Kapp_C3 * cox_i ** 2 * QH2_x - cred_i ** 2 * Q_x)  # new implementation with phosphate control

    ###### Complex IV ######
    # 2 cytoC(red)2+_i + 0.5O2_x + 4H+_x <-> cytoC(ox)3+_x + H2O_x + 2H+_i + 2DPsi

    # Constants
    k_O2 = 1.2e-4  # mol (L matrix water)**(-1)

    # Gibbs energy (J mol**(-1))
    DrGo_C4 = -202160  # J mol**(-1)
    DrGapp_C4 = DrGo_C4 - 2 * R * T * np.log(H_c)

    # Apparent equilibrium constant
    Kapp_C4 = np.exp(-(DrGapp_C4 + n_C4 * F * DPsi) / (R * T)) * (H_x / H_c) ** n_C4

    # Flux (mol (s * L mito)**(-1))
    J_C4 = X_C4 * (Kapp_C4 ** 0.5 * cred_i * O2_x ** 0.25 - cox_i) * (1 / (1 + k_O2 / O2_x))

    ###### F0F1-ATPase ######
    # ADP3-_x + HPO42-_x + H+_x + n_A*H+_i <-> ATP4- + H2O + n_A*H+_x

    # Gibbs energy (J mol**(-1))
    DrGo_F = 4990
    DrGapp_F = DrGo_F + R * T * np.log(H_x * PATP_x / (PADP_x * PPi_x))

    # Apparent equilibrium constant
    Kapp_F = np.exp((DrGapp_F + n_F * F * DPsi) / (R * T)) * (H_c / H_x) ** n_F

    # Flux (mol (s * L mito)**(-1))
    J_F = X_F * (Kapp_F * sumADP_x * sumPi_x - sumATP_x)

    ###### ANT ######
    # ATP4-_x + ADP3-_i <-> ATP4-_i + ADP3-_x

    # Constants
    del_D = 0.0167
    del_T = 0.0699
    k2o_ANT = 9.54 / 60  # s**(-1)
    k3o_ANT = 30.05 / 60  # s**(-1)
    K0o_D = 38.89e-6  # mol (L cuvette water)**(-1)
    K0o_T = 56.05e-6  # mol (L cuvette water)**(-1)
    A = +0.2829
    B = -0.2086
    C = +0.2372

    phi = F * DPsi / (R * T)

    # Reaction rates
    k2_ANT = k2o_ANT * np.exp((A * (-3) + B * (-4) + C) * phi)
    k3_ANT = k3o_ANT * np.exp((A * (-4) + B * (-3) + C) * phi)

    # Dissociation constants
    K0_D = K0o_D * np.exp(3 * del_D * phi)
    K0_T = K0o_T * np.exp(4 * del_T * phi)

    q = k3_ANT * K0_D * np.exp(phi) / (k2_ANT * K0_T)
    term1 = k2_ANT * ATP_x * ADP_c * q / K0_D
    term2 = k3_ANT * ADP_x * ATP_c / K0_T
    num = term1 - term2
    den = (1 + ATP_c / K0_T + ADP_c / K0_D) * (ADP_x + ATP_x * q)

    # Flux (mol (s * L mito)**(-1))
    J_ANT = E_ANT * num / den

    ###### H+-PI2 cotransporter ######
    # H2PO42-_x + H+_x = H2PO42-_c + H+_c

    # Constant
    k_PiC = 1.61e-3  # mol (L cuvette)**(-1)

    # H2P04- species
    HPi_c = Pi_c * (H_c / K_HPi)
    HPi_x = Pi_x * (H_x / K_HPi)

    # Flux (mol (s * L mito)**(-1))
    J_PiC = E_PiC * (H_c * HPi_c - H_x * HPi_x) / (k_PiC + HPi_c)

    ###### H+ leak ######

    # Flux (mol (s * L mito)**(-1))
    J_H = X_H * (H_c * np.exp(phi / 2) - H_x * np.exp(-phi / 2))

    ###### ATPase ######
    # ATP4- + H2O = ADP3- + PI2- + H+

    # Flux (mol (s * L cyto)**(-1))
    J_AtC = X_AtC / V_c

    ###### Differential equations (equation 23) ######
    # Membrane potential
    dDPsi = (n_C1 * J_C1 + n_C3 * J_C3 + n_C4 * J_C4 - n_F * J_F - J_ANT - J_H) / Cm

    # Matrix species
    dATP_x = (J_F - J_ANT) / W_x
    dADP_x = (-J_F + J_ANT) / W_x
    dPi_x = (-J_F + J_PiC) / W_x
    dNADH_x = (J_DH - J_C1) / W_x
    dQH2_x = (J_C1 - J_C3) / W_x

    # IMS species
    dcred_i = 2 * (J_C3 - J_C4) / W_i

    # Buffer species
    dATP_c = (V_m2c * J_ANT - J_AtC) / W_c
    dADP_c = (-V_m2c * J_ANT + J_AtC) / W_c

    dX = [dDPsi, dATP_x, dADP_x, dPi_x, dNADH_x, dQH2_x, dcred_i, dATP_c, dADP_c]

    # Calculate state-dependent quantities after model is solved.
    if solve_ode == 1:
        return dX
    else:
        J = np.array([PATP_x, PADP_x, PPi_x, PATP_c, PADP_c, PPi_c, J_DH, J_C1, J_C3, J_C4, J_F, J_ANT])
        return dX, J


## TODO  - Changing Pi experiment

## START
# Membrane potential
DPsi_0 = 175 * 1e-3

# Matrix species
ATP_x_0 = 0.5e-3
ADP_x_0 = 9.5e-3
Pi_x_0 = 0.3e-3
NADH_x_0 = 0.1 * NAD_tot
QH2_x_0 = 0.1 * Q_tot

# IM species
cred_i_0 = 0.1 * c_tot

# Cytosol species
ATP_c_0 = 5.0e-3
# ADP_c_0 = 1.3e-3  # adp according to beard's simulation
ADP_c_0 = 2.5e-3

Pi_test = np.arange(0, 10e-3 + 5e-6, 5e-4)
steady_state = np.zeros((len(Pi_test), 9))
J_C4 = np.zeros(len(Pi_test))


def find_closest_indexes(array, values):
    array = np.array(array)  # Ensure input is a NumPy array
    values = np.array(values)  # Ensure input values are a NumPy array
    indexes = []

    for value in values:
        index = (np.abs(array - value)).argmin()
        indexes.append(index)

    return indexes


def calculate_absolute_error(search_array, index_array, reference_array):
    # Fetch the elements from search_array based on index_array
    selected_search_elements = [search_array[i] for i in index_array]

    # Calculate absolute errors
    absolute_errors = [abs(selected - ref) for selected, ref in zip(selected_search_elements, reference_array)]

    # Sum the absolute errors
    total_error = sum(absolute_errors)

    return total_error


def objective_function(params, experiment_pi_o2, experiment_o2_normalized, experiment_pi_mp, experiment_mp_normalized,
                       is_phosphate_control, is_mp_considered, is_reference_dataset, o2_normalization_factor):
    if is_phosphate_control:
        X_DH, X_C1, X_C3, X_C4, X_F, E_ANT, E_PiC, X_H, k_Pi3, k_Pi4 = params  # phosphate control
        activity_array = np.array([X_DH, X_C1, X_C3, X_C4, X_F, E_ANT, E_PiC, X_H, 0.1, 1.3, k_Pi3,
                                   k_Pi4])  # phosphate control , initial Pi=1.3
    else:
        X_DH, X_C1, X_C3, X_C4, X_F, E_ANT, E_PiC, X_H = params  # without phosphate control
        activity_array = np.array(
            [X_DH, X_C1, X_C3, X_C4, X_F, E_ANT, E_PiC, X_H, 0.1, 1.3])  # without phosphate control , initial Pi=1.3

    for i in range(len(Pi_test)):
        X_0 = np.array([DPsi_0, ATP_x_0, ADP_x_0, Pi_x_0, NADH_x_0, QH2_x_0, cred_i_0, ATP_c_0, ADP_c_0])
        activity_array[9] = Pi_test[i]

        # run for long time to acheive steady-state
        steady_state_temp_results = solve_ivp(dXdt, [0, 3000], X_0, method='Radau',
                                              args=(activity_array, 1, is_phosphate_control)).y[:, -1]
        steady_state[i] = steady_state_temp_results
        f, J = dXdt(3000, steady_state_temp_results, activity_array, 0, is_phosphate_control)
        J_C4[i] = J[9]  # oxygen flux in mol O / sec / (L mito)

    closest_pi_indexes_o2 = find_closest_indexes(Pi_test, experiment_pi_o2)
    closest_pi_indexes_mp = find_closest_indexes(Pi_test, experiment_pi_mp)

    # convert to units of nmol / min / UCS
    # using the conversion factor 0.0012232 mL of mito per UCS
    JO2 = J_C4 / 2 * 60 * 1e9 * 0.0000012232
    if is_reference_dataset:
        JO2 = JO2 / max(
            abs(JO2))  # normalized to non db parameter maximum value 172.22488639746368=  value obtained from the reference dataset
    else:
        JO2 = JO2 / o2_normalization_factor

    DPsi, ATP_x, ADP_x, Pi_x, NADH_x, QH2_x, cred_i, ATP_c, ADP_c = steady_state.T

    # DPsi normalized between 0 and 1
    min_DPsi = min(DPsi)
    max_DPsi = max(DPsi)
    DPsi_normalized = [(i - min_DPsi) / (max_DPsi - min_DPsi) for i in DPsi]

    # Normalization of experimental data of membrane potential between 0 and 1
    min_mp_experiment = min(experiment_mp_normalized)
    max_mp_experiment = max(experiment_mp_normalized)
    experiment_mp_normalized_new = [(i - min_mp_experiment) / (max_mp_experiment - min_mp_experiment) for i in
                                    experiment_mp_normalized]

    # error calculation
    error_o2 = calculate_absolute_error(JO2, closest_pi_indexes_o2, experiment_o2_normalized)
    error_mp = calculate_absolute_error(DPsi_normalized, closest_pi_indexes_mp, experiment_mp_normalized_new)

    if is_mp_considered:
        absolute_error = error_o2 + error_mp
    else:
        absolute_error = error_o2

    print('absolute error is ' + str(absolute_error))
    if min(JO2) < 0:
        return 1000000
    else:
        return absolute_error


# experiment  data points
pi_titration_concentrations_for_o2 = [[0.5, 1, 2, 3, 4, 5, 7],
                                      [0.5, 1, 2, 3, 5, 7],
                                      [0.5, 1, 1.5, 2, 3, 5, 9],
                                      [0.5, 1, 2, 4, 8],
                                      [0.5, 1, 3, 5, 9]]

o2_flux_values_normalized = [[0.595969871, 0.661480177, 0.728574675, 0.786505926, 0.811438874, 0.80031344, 0.793685748],
                             [0.60498659, 0.705104873, 0.77001633, 0.806611977, 0.775745748, 0.785701089],
                             [0.778967218, 0.869278811, 0.942183422, 0.962066498, 1, 1.030390548, 0.940624715],
                             [0.512502575, 0.564077952, 0.586076113, 0.654017919, 0.631786164],
                             [0.618061469, 0.637596278, 0.732428972, 0.717020278, 0.711700711]]

pi_titration_concentrations_for_mp = [[0.5, 1, 2, 3, 4, 5],
                                      [0.5, 1, 2, 3, 5],
                                      [0.5, 1, 2, 3, 5, 9],
                                      [0.5, 1, 2, 4, 8],
                                      [0.5, 1, 3, 5]]

mp_values = [[201.7241734, 201.3120786, 201.4525371, 201.5212692, 201.7047127, 201.8942404],
             [193.4604233, 193.7241005, 194.0106914, 194.7185199, 196.0151709],
             [185.5473316, 185.7961981, 185.7815174, 186.1682006, 186.564346, 188.3755885],
             [200.3182832, 200.2205146, 201.0965589, 201.9522624, 202.6569549],
             [190.8869506, 190.774823, 190.7757864, 191.0537214]]

# Ensure results directory exists
results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Prepare to collect data for Excel
excel_data = []

for i in range(1, 2):  # initial condition changing loop
    # lower_bounds = np.array([0, 1.0e2, 1.0e4, 0.0025, 1.0e1, 0.2, 5.0e4, 1.0e1])
    # upper_bounds = np.array([0.25, 1.0e6, 1.0e8, 0.0625, 1.0e5, 0.45, 5.0e8, 1.0e5])

    lower_bounds = np.array([0.08, 4.0e3, 5.0e5, 0.0090, 1.0e4, 0.321, 4.0e7, 6.0e4])
    upper_bounds = np.array([0.13, 5.0e3, 7.0e5, 0.0100, 1.2e4, 0.322, 5.0e7, 7.0e4])

    if i == 0:  # Randall's original initial guess
        # initial_guess = np.array([0.1732, 1.0e4, 1.0e6, 0.0125, 1.0e3, 0.325, 5.0e6, 1.0e3])
        initial_guess = np.array([0.06, 4.5e3, 6.0e5, 0.0095, 1.1e4, 0.3215, 6.0e7, 6.5e3])

    else:  # Randomized initial guess
        lower_bounds = lower_bounds
        upper_bounds = upper_bounds
        initial_guess = [random.uniform(lower, upper) for lower, upper in zip(lower_bounds, upper_bounds)]

    for j in range(1, 2):  # phosphate control changing loop
        if j == 0:
            is_phosphate_control = False
        else:
            is_phosphate_control = True
            # Add 2 more values to the end
            lower_bounds = np.append(lower_bounds, [8.0e1, 0.0284])
            upper_bounds = np.append(upper_bounds, [9.0e1, 0.0285])
            initial_guess = np.append(initial_guess, [random.uniform(8.0e1, 9.0e1), random.uniform(0.0284, 0.0285)])

        for k in range(1, 2):  # considering membrane potential error
            if k == 0:
                is_mp_considered = False
            else:
                is_mp_considered = True

            for l in range(1):  # Iterate for n=3 rounds to find an average or best optimized parameters
                o2_normalization_value = 100

                # Reference data set parameter optimization
                pi_titration_concentrations_for_o2_temp = [0.5e-3, 1e-3, 1.5e-3, 2e-3, 3e-3, 5e-3, 9e-3]
                o2_flux_values_normalized_temp = [0.778967218, 0.869278811, 0.942183422, 0.962066498, 1, 1.030390548,
                                                  0.940624715]

                pi_titration_concentrations_for_mp_temp = [0.5e-3, 1e-3, 2e-3, 3e-3, 5e-3, 9e-3]
                mp_values_normalized_temp = [185.5473316, 185.7961981, 185.7815174, 186.1682006, 186.564346,
                                             188.3755885]

                is_reference_dataset = True

                # Optimization using least squares
                result = least_squares(
                    objective_function, initial_guess,
                    args=(
                        pi_titration_concentrations_for_o2_temp, o2_flux_values_normalized_temp,
                        pi_titration_concentrations_for_mp_temp, mp_values_normalized_temp,
                        is_phosphate_control, is_mp_considered, is_reference_dataset, 100
                    ),
                    bounds=(lower_bounds, upper_bounds)
                )

                # Optimized parameters
                optimized_params = result.x

                # Extract parameters
                X_DH = optimized_params[0]
                X_C1 = optimized_params[1]
                X_C3 = optimized_params[2]
                X_C4 = optimized_params[3]
                X_F = optimized_params[4]
                E_ANT = optimized_params[5]
                E_PiC = optimized_params[6]
                X_H = optimized_params[7]

                # Set activity array based on phosphate control
                activity_array = np.array(
                    [X_DH, X_C1, X_C3, X_C4, X_F, E_ANT, E_PiC, X_H, 0.1, 1.3])  # without phosphate control
                if j == 1:
                    k_pi3 = optimized_params[8]
                    k_pi4 = optimized_params[9]
                    activity_array = np.array([X_DH, X_C1, X_C3, X_C4, X_F, E_ANT, E_PiC, X_H, 0.1, 1.3, k_pi3, k_pi4])

                for m in range(len(Pi_test)):
                    X_0 = np.array([DPsi_0, ATP_x_0, ADP_x_0, Pi_x_0, NADH_x_0, QH2_x_0, cred_i_0, ATP_c_0, ADP_c_0])
                    activity_array[9] = Pi_test[m]

                    # Run for long time to achieve steady-state
                    steady_state_temp_results = solve_ivp(
                        dXdt, [0, 3000], X_0, method='Radau',
                        args=(activity_array, 1, is_phosphate_control)
                    ).y[:, -1]
                    steady_state[m] = steady_state_temp_results
                    _, J = dXdt(3000, steady_state_temp_results, activity_array, 0, is_phosphate_control)
                    J_C4[m] = J[9]  # oxygen flux in mol O / sec / (L mito)

                # Normalize oxygen flux
                JO2 = J_C4 / 2 * 60 * 1e9 * 0.0000012232
                o2_normalization_value = max(abs(JO2))
                JO2 = JO2 / max(abs(JO2))
                DPsi, ATP_x, ADP_x, Pi_x, NADH_x, QH2_x, cred_i, ATP_c, ADP_c = steady_state.T

                # DPsi normalized between 0 and 1
                min_DPsi = min(DPsi)
                max_DPsi = max(DPsi)
                DPsi_normalized = [(val - min_DPsi) / (max_DPsi - min_DPsi) for val in DPsi]

                is_reference_dataset = False

                # Perform optimization for each data point
                # for n in range(len(pi_titration_concentrations_for_o2)):
                for n in range(3, 4):

                    # Define the unique identifier
                    unique_id = f"i{i}_j{j}_k{k}_l{l}_n{n}"

                    result = least_squares(
                        objective_function, initial_guess,
                        args=(pi_titration_concentrations_for_o2[n], o2_flux_values_normalized[n],
                              pi_titration_concentrations_for_mp[n], mp_values[n], is_phosphate_control,
                              is_mp_considered, is_reference_dataset,
                              o2_normalization_value), bounds=(lower_bounds, upper_bounds)
                    )

                    # Optimized parameters
                    optimized_params = result.x

                    # Print the results
                    print(f"Data point {i}: Optimized Parameters")
                    print("X_DH =", optimized_params[0])
                    print("X_C1 =", optimized_params[1])
                    print("X_C3 =", optimized_params[2])
                    print("X_C4 =", optimized_params[3])
                    print("X_F =", optimized_params[4])
                    print("E_ANT =", optimized_params[5])
                    print("E_PiC =", optimized_params[6])
                    print("X_H =", optimized_params[7])

                    X_DH = optimized_params[0]
                    X_C1 = optimized_params[1]
                    X_C3 = optimized_params[2]
                    X_C4 = optimized_params[3]
                    X_F = optimized_params[4]
                    E_ANT = optimized_params[5]
                    E_PiC = optimized_params[6]
                    X_H = optimized_params[7]

                    activity_array = np.array(
                        [X_DH, X_C1, X_C3, X_C4, X_F, E_ANT, E_PiC, X_H, 0.1, 1.3])  # without phosphate control
                    if j == 1:
                        k_pi3 = optimized_params[8]
                        k_pi4 = optimized_params[9]
                        activity_array = np.array(
                            [X_DH, X_C1, X_C3, X_C4, X_F, E_ANT, E_PiC, X_H, 0.1, 1.3, k_pi3, k_pi4])

                    # draw for new data point
                    for m in range(len(Pi_test)):
                        X_0 = np.array(
                            [DPsi_0, ATP_x_0, ADP_x_0, Pi_x_0, NADH_x_0, QH2_x_0, cred_i_0, ATP_c_0, ADP_c_0])
                        activity_array[9] = Pi_test[m]

                        # Run for long time to achieve steady-state
                        steady_state_temp_results = solve_ivp(
                            dXdt, [0, 3000], X_0, method='Radau',
                            args=(activity_array, 1, is_phosphate_control)
                        ).y[:, -1]
                        steady_state[m] = steady_state_temp_results
                        _, J = dXdt(3000, steady_state_temp_results, activity_array, 0, is_phosphate_control)
                        J_C4[m] = J[9]  # oxygen flux in mol O / sec / (L mito)

                    # Normalize oxygen flux
                    JO2 = J_C4 / 2 * 60 * 1e9 * 0.0000012232
                    JO2 = JO2 / o2_normalization_value
                    DPsi, ATP_x, ADP_x, Pi_x, NADH_x, QH2_x, cred_i, ATP_c, ADP_c = steady_state.T

                    # DPsi normalized between 0 and 1
                    min_DPsi = min(DPsi)
                    max_DPsi = max(DPsi)
                    DPsi_normalized = [(val - min_DPsi) / (max_DPsi - min_DPsi) for val in DPsi]

                    min_mp_experiment = min(mp_values[n])
                    max_mp_experiment = max(mp_values[n])
                    experiment_mp_normalized_new = [(p - min_mp_experiment) / (max_mp_experiment - min_mp_experiment)
                                                    for p in mp_values[n]]

                    # Create a figure with 1 row and 2 columns
                    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

                    patient_id = n + 1

                    # Pi Vs O2 flux plot
                    ax[0].plot(Pi_test * 1000, JO2, 'red', label='2.5mM ADP')  # Membrane Potential
                    ax[0].scatter(pi_titration_concentrations_for_o2[n], o2_flux_values_normalized[n],
                                  color=(74 / 255, 4 / 255, 4 / 255),
                                  label='Experimental Data')  # Plot the experimental data points
                    ax[0].set_xlabel('Buffer Pi(mM)')  # Set the X-axis label
                    ax[0].set_ylabel('MVO2 Normalized')  # Set the Y-axis label
                    ax[0].set_title('PiC vs O2 Flux Patient {}'.format(patient_id))
                    # ax[0, 0].set_ylim(0.2,0.9)
                    ax[0].legend()

                    # # Pi Vs Membrane potential plot
                    ax[1].plot(Pi_test * 1000, DPsi_normalized, 'red', label='2.5mM ADP')  # Membrane Potential
                    ax[1].scatter(pi_titration_concentrations_for_mp[n], experiment_mp_normalized_new,
                                  color=(74 / 255, 4 / 255, 4 / 255),
                                  label='Experimental Data')  # Plot the experimental data points
                    ax[1].set_xlabel('Buffer Pi(mM)')  # Set the X-axis label
                    ax[1].set_ylabel('Membrane potential')  # Set the Y-axis label
                    ax[1].set_title(
                        'PiC vs Membrane potential Patient {}'.format(patient_id))  # Set the title of the plot
                    # ax[0, 1].set_ylim(184, 188)  # Set the y-axis range
                    ax[1].legend()  # Display the legend

                    # Adjust layout to prevent overlap
                    plt.tight_layout()

                    # Save plots to PNG files
                    plot_filename = f"{results_dir}/plot_i{i}_j{j}_k{k}_l{l}_n{patient_id}.png"
                    plt.savefig(plot_filename)
                    plt.close()

                    if len(activity_array) == 10:  # if no pi control
                        activity_array = np.append(activity_array, [1, 1])

                    # Collect optimized parameters for Excel
                    excel_data.append([
                        f"i{i}_j{j}_k{k}_l{l}_n{patient_id}",
                        *activity_array
                    ])

# Save optimized parameters to an Excel file
excel_filename = f"{results_dir}/optimized_parameters.xlsx"
df = pd.DataFrame(excel_data,
                  columns=["Unique ID", "X_DH", "X_C1", "X_C3", "X_C4", "X_F", "E_ANT", "E_PiC", "X_H", "X_Atc", "Pi",
                           "k_pi3", "k_pi4"])
df.to_excel(excel_filename, index=False)

print("Results and plots saved successfully.")

# # plot figures

# # # Show the figure

# best params  so far
# X_DH = 0.21309847148678188
# X_C1 = 535251.3624726401
# X_C3 = 96787829.25630796
# X_C4 = 0.021514538514574227
# X_F = 97859.73324575866
# E_ANT = 0.2418535121913848
# E_PiC = 361815718.6790009
# X_H = 92127.0045538844
# kpi_3 = 0.00010355321502021973
# kpi_4 = 0.015255448575410927


# another best with huge varying range , unsure parameter values validity
# X_DH = 19.162510335060247
# X_C1 = 1091173.3354059136
# X_C3 = 8073418241.092943
# X_C4 = 5.217851436783569
# X_F = 2518243.0637788647
# E_ANT = 5.950318573174307
# E_PiC = 22277861596.914978
# X_H = 9972244.09581357
# kpi_3 = 0.01643130228792644
# kpi_4 = 2.858182671478121

# params from minimize model - overfitting
# X_DH = 0.12500001865990146
# X_C1 = 653682.0742195459
# X_C3 = 86686659.88403115
# X_C4 = 0.03249998147164959
# X_F = 59068.37832843617
# E_ANT = 0.32499921349434957
# E_PiC = 428920248.95559597
# X_H = 86253.70617599746
# kpi_3 = 0.00019999999613863877
# kpi_4 = 0.027500003730726513


## reference
# pi_titration_concentrations_for_o2_temp =[0.5, 1, 1.5, 2, 3, 5, 9]
# o2_flux_values_normalized_temp  = [0.778967218,0.869278811,0.942183422,0.962066498,1,1.030390548,0.940624715]

# pi_titration_concentrations_for_mp_temp=[0.5, 1, 2, 3, 5, 9]
# mp_values_normalized_temp = [185.5473316,185.7961981,185.7815174,186.1682006,186.564346,188.3755885]

# min_mp_experiment = min(mp_values_normalized_temp)
# max_mp_experiment = max(mp_values_normalized_temp)
# experiment_mp_normalized_new = [(i - min_mp_experiment) / (max_mp_experiment - min_mp_experiment) for i in mp_values_normalized_temp]

# ## other data
# # pi_titration_concentrations_for_o2_temp =[0.5, 1, 3, 5, 9]
# # o2_flux_values_normalized_temp  =  [0.657075515,0.677843425,0.778662266,0.762280926,0.75662557]

# # pi_titration_concentrations_for_mp_temp=[0.5, 1, 3, 5, 9]
# # mp_values_normalized_temp = [190.8869506,190.774823,190.7757864,191.0537214,190.1153395]

# # min_mp_experiment = min(mp_values_normalized_temp)
# # max_mp_experiment = max(mp_values_normalized_temp)
# # experiment_mp_normalized_new = [(i - min_mp_experiment) / (max_mp_experiment - min_mp_experiment) for i in mp_values_normalized_temp]


# for i in range(len(Pi_test)):
#     X_0 = np.array([DPsi_0, ATP_x_0, ADP_x_0, Pi_x_0, NADH_x_0, QH2_x_0, cred_i_0, ATP_c_0, ADP_c_0])
#      # activity_array = np.array([X_DH, X_C1, X_C3, X_C4, X_F, E_ANT, E_PiC, X_H, 0.1,Pi_test[i]]) # without phosphate control
#     activity_array = np.array([X_DH, X_C1, X_C3, X_C4, X_F, E_ANT, E_PiC, X_H, 0.1,Pi_test[i],k_pi3,k_pi4])

#     # run for long time to acheive steady-state
#     steady_state_temp_results = solve_ivp(dXdt, [0, 3000], X_0, method='Radau', args=(activity_array, 1,True)).y[:, -1]
#     steady_state[i] = steady_state_temp_results
#     f, J = dXdt(3000, steady_state_temp_results, activity_array, 0,True)
#     J_C4[i] = J[9]  # oxygen flux in mol O / sec / (L mito)

# JO2 = J_C4 / 2 * 60 * 1e9 * 0.0000012232
# print(max(abs(JO2)))
# JO2 = JO2/ max(abs(JO2))
# DPsi, ATP_x, ADP_x, Pi_x, NADH_x, QH2_x, cred_i, ATP_c, ADP_c = steady_state.T
# # DPsi normalized between 0 and 1
# min_DPsi = min(DPsi)
# max_DPsi = max(DPsi)
# DPsi_normalized = [(i - min_DPsi) / (max_DPsi - min_DPsi) for i in DPsi]
# #

# fig, ax = plt.subplots(2, 2, figsize=(12, 10))


# # Pi Vs O2 flux plot
# ax[0, 0].plot(Pi_test * 1000,JO2, 'red', label='2.5mM ADP')  # Membrane Potential
# ax[0, 0].scatter(pi_titration_concentrations_for_o2_temp , o2_flux_values_normalized_temp, color=(74/255, 4/255, 4/255), label='Experimental Data')  # Plot the experimental data points
# ax[0, 0].set_xlabel('Buffer Pi(mM)')  # Set the X-axis label
# ax[0, 0].set_ylabel('MVO2 Normalized')  # Set the Y-axis label
# ax[0, 0].set_title('PiC vs O2 Flux Patient 5 (DB)')  # Set the title of the plot
# # ax[0, 0].set_ylim(0.2,0.9)
# ax[0, 0].legend()


# # # Pi Vs Membrane potential plot
# ax[0, 1].plot(Pi_test * 1000 , DPsi_normalized, 'red', label='2.5mM ADP')  # Membrane Potential
# ax[0, 1].scatter(pi_titration_concentrations_for_mp_temp , experiment_mp_normalized_new, color=(74/255, 4/255, 4/255), label='Experimental Data')  # Plot the experimental data points
# ax[0, 1].set_xlabel('Buffer Pi(mM)')  # Set the X-axis label
# ax[0, 1].set_ylabel('Membrane potential')  # Set the Y-axis label
# ax[0, 1].set_title('PiC vs Membrane potential Patient 5 (DB)')  # Set the title of the plot
# # ax[0, 1].set_ylim(184, 188)  # Set the y-axis range
# ax[0, 1].legend()  # Display the legend

# # Adjust layout to prevent overlap
# plt.tight_layout()
