# loading libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import scipy.stats as sts
import math
import scipy.optimize as opt
import sys
from scipy.stats.distributions import chi2

# inicializacia pola, do ktoreho ulozime data
pts = pd.DataFrame([])
pd.set_option('display.max_rows', pts.shape[0]+1)

# nastavenie poctu binov a hranic oblasti, v ktorej budeme hladat Higgsa
nbins = 50
x_min = 105
x_max = 155

# exponencialna funckia na popis pozadi
def exponential(x, A, B):
	return A * np.exp( - B * x)

# normalizacia exponencialnej funkcie
def norm_exp(A, B, xmin, xmax):
	return A*(np.exp(-xmin*B)-np.exp(-xmax*B))/B

# nacitanie suboru s datami a ulozenie dat do pola
path = "data/data_real.txt"  # tady se nastavuje cesta k souboru (relativni ke slozce ze ktere skript zpoustite) s nasimulovanymi hodnotami data_real.txt
file = pd.read_csv(path, sep=" ", header=None, names=["mass"])
pts = pts.append(file, ignore_index=True)
pts = np.array(pts["mass"])
pts = np.sort(pts)

# inicializacia histogramu
count, bins, ignored = plt.hist(pts, nbins, density=True, edgecolor="c", range=(x_min,x_max))
data_entries, bins = np.histogram(pts, bins=np.linspace(x_min, x_max, nbins+1), density=True)
# format osi
plt.title("Higgs boson")
plt.xlabel("mass [GeV/$c^2$]")
plt.ylabel("Counts")

# inicializacia pola s x-ovymi hodnotami
dist_pts = np.linspace(x_min, x_max, nbins)
binscenters = np.array([0.5 * (dist_pts[i] + dist_pts[i+1]) for i in range(len(dist_pts)-1)])
n = len(pts)

# nacitanie hmotnosti z prikazoveho riadku
m0 = float(sys.argv[1])

# sirka gaussovho rozdelenie - pevna hodnota
sigma = 1.46

# rychle nafitovanie dat pre zistenie parametrov exponencialneho pozadia
#############################################################################################################
#  	zvolte parametry pocatecni hodnoty parametru fitu A a B doplnenim p0=[ , ] na nasledujicim radku        #
#############################################################################################################
popt, pcov = opt.curve_fit(exponential, xdata=dist_pts, ydata=data_entries, p0=[ , ])
A, B = popt

# funkcia, ktora vracia hustotu pravdepodobnosti
def probability_density(xvals, nu_s, nu_b, m0, sigma):
	s_1 = (nu_s)/(nu_s+nu_b)
	s_2 = ((nu_b)/(nu_s+nu_b))*exponential(xvals, A, B)/norm_exp(A, B, np.min(xvals), np.max(xvals))
	pdf = s_1*(1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (xvals - m0)**2 / (2 * sigma**2))) + s_2
	return pdf

# vypocet -log(L)
def likelihood(params, *args):
	nu_s, nu_b = params
	xvals = args[0]
	pdf_vals = probability_density(xvals, nu_s, nu_b, m0, sigma)
	ln_pdf_vals = np.log(pdf_vals)
#############################################################################################################
#   likelihood obsahuje v menovateli n!. Pri vypocte tak dostavame log(n!), co je vsak pri velkom mnozstve  #
#   dat velmi neprakticke na vypocet. Nahradte preto log(n!) pomocou Stirlingovej aproximacie               #
	log_factorial =																																		                
#############################################################################################################
	log_lik_val = ln_pdf_vals.sum()  + n * np.log(nu_s+nu_b) - (nu_s+nu_b) - log_factorial
	return -log_lik_val

# inicializacia minimalizacie - uvodne hodnoty parametrov nu_s a nu_b
nu_s_init = 0.01*n
nu_b_init = 0.99*n
params_init = np.array([nu_s_init, nu_b_init])
# limity hodnot parametrov nu_s a nu_b pri minimalizacii
bnds = ((0.005*n, 0.015*n), (0.95*n, n))

# minimalizacia likelihoodu
results_uncstr = opt.minimize(likelihood, params_init, args=pts, method='TNC', bounds=bnds, options={"maxiter":2000})

# vysledne hodnoty minimalizacie vlozime do novych premennych
nu_s_fin, nu_b_fin = results_uncstr.x

# vypocet q0
#############################################################################################################
#  tu doplnte vypocet q0 pomocou funkcie likelihood([nu_s, nu_b], pts). Dajte si pozor na to, ze nami       #
#  zadefinovana funkcia likelihood vracia zaporny logaritmus likelihoodu                                    #
q0 =                                                                                                     
#############################################################################################################
print( f'q0: {q0}' )

# vypocet signifikancie
#############################################################################################################
#  tu doplnte vypocet signifikancie z                                                                       #
z =                                                                                                       
#############################################################################################################
print( f'z: {z}')

# vypocet p-value
p_value = chi2.sf(q0, df=1)
print( f'p-value: {p_value}')

# vykreslenie so sginalom a bez signalu
plt.plot(dist_pts, probability_density(dist_pts, nu_s_fin, nu_b_fin, m0, sigma), color="b")
plt.plot(dist_pts, probability_density(dist_pts, 0, nu_b_fin, m0, sigma), color="g")
plt.show()
