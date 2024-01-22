import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./Resultats/Torque.txt", sep="\t", header=0, usecols=[0, 1, 2])
param = pd.read_csv("./Resultats/Variables.txt", sep="\t", header=0, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8])

#   1.Initialisation

print('\n---------> Parametres\n')

C_ech = 1 # Rapport d'echelle
R = param.loc[0]['R_turb']/C_ech
Rho = param.loc[0]['Rho1']
Eta = param.loc[0]['Eta1']
Vair = param.loc[0]['VitesseAir']
Vrot = param.loc[0]['VitesseRotation']
print("Rayon turbine =", R, 'm')
print("Rho =", Rho, 'kg/m^3')
print("Eta =", Eta, 'Pa.s')
print("Vitesse air =", Vair, 'm/s')
print("Vitesse rotation = ", Vrot, 'rad/s')

Re = 2*Rho*Vair*R/Eta
print("Reynolds = ", Re)

Lambda = R*Vrot/Vair
print("Tip speed ratio = ", Lambda)

Tau = 5 # Temps regime transitoire
iNonValide = df[df['Temps']<Tau].index.tolist()
iDepart = iNonValide[-1]


#   2.Analyse couple/temps

print('\n---------> Courbes\n')

plt.figure("Courbe T = f(t)")
plt.xlabel("Temps (s)")
plt.ylabel("Couple (N.m)")
plt.grid(True)
plt.plot(df['Temps'][iDepart:], df['Torque'][iDepart:])
plt.show()

#   2.Analyse Cp/temps

Cp = df['Torque'][iDepart:]*Vrot/(0.5*Rho*Vair**3)
Cp_5 = np.percentile(Cp, 5)
Cp_95 = np.percentile(Cp, 95)
plt.figure("Courbe Cp = f(t)")
plt.xlabel("Temps (s)")
plt.ylabel("Coeff de puissance")
plt.grid(True)
plt.plot(df['Temps'][iDepart:], Cp)
plt.axhline(Cp_5, c='r', ls='--')
plt.axhline(Cp_95, c='r', ls='--')
plt.show()

#   3.Compte rendu

print('\n---------> Analyse\n')

Cp_max = np.amax(Cp)
print("Cp_max =", Cp_max)
Cp_moy = np.average(Cp)
print("Cp_moy =", Cp_moy)
print("Cp_95 =", Cp_95)