import numpy as np
import matplotlib.pyplot as plt

x,y,dy = np.loadtxt('sand.txt',unpack=True)

# given fit params
m = 16.1
dm = 1.0
q = -2.61
dq = 0.34

# covariance matrix
cov = np.array([[1.068, -0.302],
                [-0.302, 0.118]])

# (a) plot
xx = np.linspace(min(x),max(x),100)
yy = m*xx+q

plt.errorbar(x,y,yerr=dy,fmt='o',label='data',capsize=3)
plt.plot(xx,yy,'r-',label='fit')
plt.xlabel('diameter [mm]')
plt.ylabel('slope')
plt.legend()
plt.title('Beach slope vs sand diameter')
#plt.show()

# (b) ignoring correlation
x0 = 1.5
y0 = m*x0 + q
dy0_nocorr = np.sqrt( (x0*dm)**2 + dq**2 )
print("b) ignoring corr: y(1.5mm)=%.3f ± %.3f"%(y0,dy0_nocorr))

# (c) including correlation
var_m = cov[0,0]
var_q = cov[1,1]
cov_mq = cov[0,1]

dy0_corr = np.sqrt( (x0**2)*var_m + var_q + 2*x0*cov_mq )
print("c) with corr: y(1.5mm)=%.3f ± %.3f"%(y0,dy0_corr))

if dy0_corr<dy0_nocorr:
    print("uncertainty smaller when correlation is included (negative covariance reduces total error)")
else:
    print("uncertainty larger when correlation is included")
