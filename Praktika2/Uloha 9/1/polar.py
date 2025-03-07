import numpy as np
import math

data = np.loadtxt("data.txt")
n=data.shape[0]

z = np.zeros((n, 1))

j = 0
for i in data:
	E1 = i[0]
	E2 = i[1]
	phi1 = i[2]
	phi2 = i[3]
	phi = phi1-phi2

	stEx2 = 0.5*E1**2
	stEy2 = 0.5*E2**2
	stExEy = 0.5*E1*E2*math.cos(math.radians(phi))
	silenost = 0.5*E1*E2*math.sin(math.radians(phi))

	P1 = (stEx2- stEy2)/(stEx2 + stEy2)
	P2 = (2*stExEy)/(stEx2 + stEy2)
	P2 = (2*silenost)/(stEx2 + stEy2)

	velikost = math.sqrt(P1**2 + P2**2 + P3**2)
	print(velikost)

