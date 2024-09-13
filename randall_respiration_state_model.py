import numpy as np
import matplotlib.pyplot as plt
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

activity_array = np.array([X_DH, X_C1, X_C3, X_C4, X_F, E_ANT, E_PiC, X_H, X_AtC])

###### Initial Conditions ######
# Membrane Potential
DPsi_0 = 175 / 1000  # V

# Matrix species
sumATP_x_0 = 0.5e-3  # mol (L matrix water)**(-1)
sumADP_x_0 = 9.5e-3  # mol (L matrix water)**(-1)
sumPi_x_0 = 0.3e-3  # mol (L matrix water)**(-1)
NADH_x_0 = 0  # mol (L matrix water)**(-1)
QH2_x_0 = 0.1 * Q_tot  # mol (L matrix water)**(-1)

# IMS species
cred_i_0 = 0.1 * c_tot  # mol (L IMS water)**(-1)

# Cytosolic species
sumATP_c_0 = 0  # mol (L cyto water)^(-1)
sumADP_c_0 = 0  # mol (L cyto water)^(-1)
sumPi_c_0 = 5.0e-3  # mol (L cyto water)^(-1)

X_0 = np.array(
    [DPsi_0, sumATP_x_0, sumADP_x_0, sumPi_x_0, NADH_x_0, QH2_x_0, cred_i_0, sumATP_c_0, sumADP_c_0, sumPi_c_0])


def dXdt(t, X, activity_array, solve_ode):
    # Unpack variables
    DPsi, sumATP_x, sumADP_x, sumPi_x, NADH_x, QH2_x, cred_i, sumATP_c, sumADP_c, sumPi_c = X
    X_DH, X_C1, X_C3, X_C4, X_F, E_ANT, E_PiC, X_H, X_AtC = activity_array

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

    # Other concentrations computed from the state variables
    NAD_x = NAD_tot - NADH_x  ## mol (L matrix water)**(-1)
    Q_x = Q_tot - QH2_x  ## mol (L matrix water)**(-1)
    cox_i = c_tot - cred_i  ## mol (L matrix water)**(-1)

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

    # Cytosolic species
    ATP_c = sumATP_c / PATP_c  # [ATP4-]_c
    ADP_c = sumADP_c / PADP_c  # [ADP3-]_c
    Pi_c = sumPi_c / PPi_c  # [HPO42-]_c

    ###### NADH Dehydrogenase ######
    # Constants
    r = 6.8385
    k_Pi1 = 4.659e-4  # mol (L matrix water)**(-1)
    k_Pi2 = 6.578e-4  # mol (L matrix water)**(-1)

    # Flux (mol (s * L mito)**(-1))
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
    J_C3 = X_C3 * (Kapp_C3 * cox_i ** 2 * QH2_x - cred_i ** 2 * Q_x)

    ###### Complex IV ######
    # 2 cytoC(red)2+_i + 0.5O2_x + 4H+_x <-> cytoC(ox)3+_x + H2O_x + 2H+_i +2DPsi

    # Constant
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
    dPi_c = (-V_m2c * J_PiC + J_AtC) / W_c

    dX = [dDPsi, dATP_x, dADP_x, dPi_x, dNADH_x, dQH2_x, dcred_i, dATP_c, dADP_c, dPi_c]

    # Calculate state-dependent quantities after model is solved.
    if solve_ode == 1:
        return dX
    else:
        J = np.array([PATP_x, PADP_x, PPi_x, PATP_c, PADP_c, PPi_c, J_DH, J_C1, J_C3, J_C4, J_F, J_ANT, J_PiC])
        return dX, J


### Four State Model ###
# State 1 - no substrates
t_1 = np.linspace(0, 25, 25 * 2)
X_AtC = 0.
X_DH = 0.001  # Kept non-zero for solver stability
activity_array = np.array([X_DH, X_C1, X_C3, X_C4, X_F, E_ANT, E_PiC, X_H, X_AtC])
state_1_results = solve_ivp(dXdt, [0, 25], X_0, method='Radau', t_eval=t_1, args=(activity_array, 1,))

# State 2 - Add substrate (i.e. turn on X_DH)
t_2 = np.linspace(25, 75, (75 - 25) * 2)
X_DH = 0.0866 * 2
activity_array = np.array([X_DH, X_C1, X_C3, X_C4, X_F, E_ANT, E_PiC, X_H, X_AtC])
state_2_results = solve_ivp(dXdt, [25, 75], state_1_results.y[:, -1], method='Radau', t_eval=t_2,
                            args=(activity_array, 1,))

# State 3 and 4 - Add ADP
t_3 = np.linspace(75, 200, (200 - 75) * 2)
state_2_results.y[8, -1] = 0.375e-3  # Molar
state_3_results = solve_ivp(dXdt, [75, 200], state_2_results.y[:, -1], method='Radau', t_eval=t_3,
                            args=(activity_array, 1,))

# Concatenate Results
# Note: remove redundant time points
all_results = np.hstack((state_1_results.y[:, 0:-1], state_2_results.y[:, 0:-1], state_3_results.y))
t = np.concatenate((state_1_results.t[0:-1], state_2_results.t[0:-1], state_3_results.t))

DPsi, sumATP_x, sumADP_x, sumPi_x, NADH_x, QH2_x, cred_i, sumATP_c, sumADP_c, sumPi_c = all_results

# Calculate complex IV Flux
J_C4 = np.zeros(len(t))
for i in range(len(t)):
    dX, J = dXdt(t[i], all_results[:, i], activity_array, 0)
    J_C4[i] = J[9]

# Convert complex IV flux to oxygen flux in nmol O2 / U citrate synthase
JO2 = J_C4 / 2 * 60 * 1e9 * 0.0000012232

fig, ax = plt.subplots(1,3, figsize = (30,5))

ax[0].plot((-5,205),(0,0),'k')
ax[0].plot((0,0),(-5,205),'k:', linewidth = 1.75)
ax[0].plot((25,25),(-5,205),'k:', linewidth = 1.75)
ax[0].plot((75,75),(-5,205),'k:', linewidth = 1.75)
ax[0].plot((150,150),(-5,205),'k:', linewidth = 1.75)
ax[0].plot(t, JO2)
ax[0].set_ylim([-5,205])
ax[0].set_xlim([-5,205])
ax[0].text(12.5, 190, 'State 1', horizontalalignment='center', verticalalignment='center', fontsize = 15)
ax[0].text(50, 190, 'State 2', horizontalalignment='center', verticalalignment='center', fontsize = 15)
ax[0].text(112.5, 190, 'State 3', horizontalalignment='center', verticalalignment='center', fontsize = 15)
ax[0].text(175, 190, 'State 4', horizontalalignment='center', verticalalignment='center', fontsize = 15)
ax[0].set_xlabel('Times (s)', fontsize = 15)
ax[0].set_ylabel('OCR (nmol O$_2$ min$^{-1}$ U CS$^{-1}$)', fontsize = 15)

ax[1].plot((-5,205),(0,0),'k')
ax[1].plot((0,0),(-0.05,1),'k:', linewidth = 1.75)
ax[1].plot((25,25),(-0.05,1),'k:', linewidth = 1.75)
ax[1].plot((75,75),(-0.05,1),'k:', linewidth = 1.75)
ax[1].plot((150,150),(-0.05,1),'k:', linewidth = 1.75)
ax[1].plot(t, NADH_x/NAD_tot,'r')
ax[1].set_ylim([-0.05,1])
ax[1].set_xlim([-5,205])
ax[1].text(12.5, 0.9, 'State 1', horizontalalignment='center', verticalalignment='center', fontsize = 15)
ax[1].text(50, .9, 'State 2', horizontalalignment='center', verticalalignment='center', fontsize = 15)
ax[1].text(112.5, .9, 'State 3', horizontalalignment='center', verticalalignment='center', fontsize = 15)
ax[1].text(175, .9, 'State 4', horizontalalignment='center', verticalalignment='center', fontsize = 15)
ax[1].set_xlabel('Times (s)', fontsize = 15)
ax[1].set_ylabel('[NADH]$_x$ (mM)', fontsize = 15)

ax[2].plot((0,0),(75,210),'k:', linewidth = 1.75)
ax[2].plot((25,25),(75,210),'k:', linewidth = 1.75)
ax[2].plot((75,75),(75,210),'k:', linewidth = 1.75)
ax[2].plot((150,150),(75,210),'k:', linewidth = 1.75)
ax[2].plot(t, DPsi*1000,'g')
ax[2].set_ylim([75,210])
ax[2].set_xlim([-5,205])
ax[2].text(12.5, 200, 'State 1', horizontalalignment='center', verticalalignment='center', fontsize = 15)
ax[2].text(50, 200, 'State 2', horizontalalignment='center', verticalalignment='center', fontsize = 15)
ax[2].text(112.5, 200, 'State 3', horizontalalignment='center', verticalalignment='center', fontsize = 15)
ax[2].text(175, 200, 'State 4', horizontalalignment='center', verticalalignment='center', fontsize = 15)
ax[2].set_xlabel('Times (s)', fontsize = 15)
ax[2].set_ylabel('$\Delta \Psi$ (mV)', fontsize = 15)

plt.show()