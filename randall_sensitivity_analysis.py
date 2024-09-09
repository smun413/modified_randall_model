import os

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

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

Pi_test = np.arange(0, 10e-3 + 5e-6, 2e-4)
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


def objective_function(params, experiment_pi_o2, experiment_o2_normalized, is_phosphate_control, is_reference_dataset,
                       o2_normalization_factor):
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

    # convert to units of nmol / min / UCS
    # using the conversion factor 0.0012232 mL of mito per UCS
    JO2 = J_C4 / 2 * 60 * 1e9 * 0.0000012232
    if is_reference_dataset:
        JO2 = JO2 / max(
            abs(JO2))  # normalized
    else:
        print('not reference ds')
        JO2 = JO2 / o2_normalization_factor

    # error calculation
    error_o2 = calculate_absolute_error(JO2, closest_pi_indexes_o2, experiment_o2_normalized)

    absolute_error = error_o2

    return absolute_error


# Prepare to collect data for Excel
excel_data = []


# Sensitivity Analysis Function
def sensitivity_analysis(params, pi_points, o2_flux, lower_bounds, upper_bounds):
    deltas = (upper_bounds - lower_bounds) / 10

    o2_normalization_factor = 170
    for j in range(len(params)):
        sensitivity_array = []
        patient_id = j + 1
        for i in range(len(params[j])):
            # Create a copy of the parameters
            perturbed_params_up = params[j].copy()
            perturbed_params_down = params[j].copy()

            # Perturb the i-th parameter up and down by delta
            perturbed_params_up[i] = perturbed_params_up[i] + deltas[i]
            perturbed_params_down[i] = perturbed_params_down[i] - deltas[i]

            # Calculate residuals for perturbed parameters
            residuals_up = objective_function(perturbed_params_up, pi_points[j], o2_flux[j], False, False,
                                              o2_normalization_factor)
            residuals_down = objective_function(perturbed_params_down, pi_points[j], o2_flux[j], False, False,
                                                o2_normalization_factor)

            # Calculate the sensitivity as the average change in residuals due to the perturbation
            sensitivity = (np.linalg.norm(residuals_up) - np.linalg.norm(residuals_down)) / (2 * deltas[i])

            # Collect sensitivity array for excel
            sensitivity_array.append(sensitivity)

        excel_data.append([
            f"patient_id{patient_id}",
            *sensitivity_array
        ])
    return None


# experiment  data points
pi_titration_concentrations_for_o2 = [[0, 0.5, 1, 2, 3, 4, 5, 7],
                                      [0, 0.5, 1, 2, 3, 5, 7],
                                      [0, 0.5, 1, 1.5, 2, 3, 5, 9],
                                      [0, 0.5, 1, 2, 4, 8],
                                      [0, 0.5, 1, 3, 5, 9]]

o2_flux_values_normalized = [
    [0.510097236, 0.633589423, 0.703234953, 0.774564673, 0.836152733, 0.86265953, 0.850831823, 0.843785769],
    [0.485529741, 0.643175307, 0.749613381, 0.818622261, 0.857527944, 0.824713338, 0.835297092],
    [0.648810344, 0.828138158, 0.924150511, 1.001657098, 1.022795258, 1.063123246, 1.095432144, 1],
    [0.473379193, 0.544853401, 0.599684384, 0.62307114, 0.695301653, 0.671666558],
    [0.525713241, 0.657075515, 0.677843425, 0.778662266, 0.762280926, 0.75662557]]

# pre optimized parameters
optimized_parameter_values = [[0.141035, 14112.04, 1126250, 0.011207, 18858.8, 0.302088, 10626683, 18337.35],
                              [0.129068, 188010.1, 7526702, 0.020369, 19741.38, 0.323577, 18135874, 19694.33],
                              [0.168016, 61862.25, 2988649, 0.015271, 6482.243, 0.317963, 7939636, 6482.553],
                              [0.102397, 120280.2, 7593095, 0.021775, 9344.718, 0.270855, 6090298, 9264.916],
                              [0.126528, 86592.28, 5793089, 0.014857, 9849.969, 0.291316, 7419082, 9436.941]]

lower_bounds = np.array([0.05, 1.0e2, 1.0e4, 0.0025, 1.0e1, 0.2, 5.0e4, 1.0e1])
upper_bounds = np.array([0.18, 1.0e6, 1.0e8, 0.0625, 1.0e5, 0.45, 5.0e8, 1.0e5])

sensitivity_analysis(optimized_parameter_values, pi_titration_concentrations_for_o2, o2_flux_values_normalized,
                     lower_bounds, upper_bounds)

# Ensure results directory exists
results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Save optimized parameters to an Excel file
excel_filename = f"{results_dir}/parameter_sensitivity.xlsx"
df = pd.DataFrame(excel_data,
                  columns=["Patient_Param_No", "X_DH", "X_C1", "X_C3", "X_C4", "X_F", " E_ANT", " E_PIC", " X_H"])
df.to_excel(excel_filename, index=False)

print("Results and plots saved successfully.")
