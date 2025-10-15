# imports
import matplotlib.pyplot as plt
from scipy.stats import binom, norm, poisson
import numpy as np

# Exercise 1: Particle detection efficiency

"""
(b) Write and execute a function to plot the probability distribution for the number of
signals registered if the particle travels through four detectors. 
"""

p = 0.85
n = 4

def signal_distribution(p, n):
    x = range(n + 1)
    y = [binom.pmf(k, n, p) for k in x]
    plt.bar(x, y)
    plt.xlabel('Number of signals')
    plt.ylabel('Probability')
    plt.title('Signal detection probability distribution')
    #plt.show()
signal_distribution(p, n)

"""
(d) You run your experiment with four detectors and produce 1000 particles which are
sent through the four detector planes. Plot the probability distribution of the number
of detected particles (number of particles with 3 or more signals). 
"""


n_particles = 1000
n_detectors = 4

signals_model = np.random.binomial(n_detectors, p, n_particles)
detected_particles = np.sum(signals_model >= 3)

plt.hist(signals_model, bins=np.arange(0, n_detectors+2)-0.5, color='lightblue', edgecolor='black')
plt.xlabel("Amount of signals")
plt.ylabel("Amount of particles")
plt.title("Distribution of signals for 1000 particles")
#plt.show()

"""
Does the width of
the distribution agree with what one would expect from the Poisson distribution? 

Answer: no it does not, because the variance in Poisson distribution is equal to the mean (and therefore greater),
the variance of binomial distiribution is np(1-p) which is smaller. 
"""



# Exercise 3 

"""
The Z-boson decays with a probability of 82% into charged particles and with about 18%
probability into neutrinos, which cannot be detected in regular particle detectors. In some
experiment, 500 Z-bosons were produced during a running time of 125 hours. Hint: this is a
new experiment which detects charged particles with a 100% efficiency.
"""

"""
(a) Write a Python script that uses a binomial distribution to calculate the probability for
390 or more Z-bosons to be detected using charged particles. 
"""

n = 500
p = 0.82

# for 390 or more
P_binom = binom.sf(389, n, p) #  because CDF gives P(X <= k)
print(f"Binomial P(X >= 390) = {P_binom:.6f}")

k = np.arange(350, 451) # expected mean 500 * 0,82 = 410 with std 8.6 ~
plt.figure(figsize=(8,4))
plt.bar(k, binom.pmf(k, n, p), alpha=0.6, label="Binomial PMF")
plt.title("Binomial distribution for Z boson detections")
plt.xlabel("Number of detected Z bosons")
plt.ylabel("Probability")
plt.legend()
#plt.show()


"""
(b) Knowing the expected value and its standard deviation, use a gaussian approximation
of the binomial distribution. Write a Python script that determines the same as in a).
Plot both the original distribution and the approximation together. How good is this
approximation?
"""
mu = n * p # parameters 
sigma = np.sqrt(n * p * (1 - p))

# normal approximation with continuity correction
z = (389.5 - mu) / sigma
P_gauss = 1 - norm.cdf(z)
print(f"gaussian approx P(X >= 390) = {P_gauss:.6f}")

x = np.arange(370, 451)
plt.figure(figsize=(8,4))
plt.bar(x, binom.pmf(x, n, p), alpha=0.6, label="Binomial PMF")
plt.plot(x, norm.pdf(x, mu, sigma)*(x[1]-x[0]), 'r--', label="Gaussian approximation")
plt.title("Binomial vs Gaussian approximation")
plt.xlabel("Number of detected Z bosons")
plt.ylabel("Probability density")
plt.legend()
#plt.show()

"""
(c) Now make a Poisson approximation. Write a Python script to determine the same as
in a). Plot both the original distribution and its approximation together. How good is
this approximation? 
"""
lam = n * p
P_poisson = poisson.sf(389, lam)
print(f"Poisson approx P(X >= 390) = {P_poisson:.6f}")

plt.figure(figsize=(8,4))
plt.bar(x, binom.pmf(x, n, p), alpha=0.6, label="Binomial PMF")
plt.plot(x, poisson.pmf(x, lam), 'g--', label="Poisson approx")
plt.title("Binomial vs Poisson approximation")
plt.xlabel("Number of detected Z bosons")
plt.ylabel("Probability")
plt.legend()
#plt.show()


"""
(d) Write a Python script to determine the probability that at least one Z-boson was
created, but could not be observed because it decayed to neutrinos, during the first
hour of running this experiment. You may assume that the rate of Z-bosons being
produced is constant.
Use both the binomial distribution and its Poisson approximation to determine this
and plot both distributions together. How good is the approximation this time? Why
is it different to c)? 
"""

rate_per_hour = 500 / 125  # bosons per hour
p_neu = 0.18

# for 4 bosons
n_hour = int(rate_per_hour)
P_neu_binom = 1 - (1 - p_neu)**n_hour
print(f"Binomial (n=4, p=0.18) P(>=1 neutrino decay) = {P_neu_binom:.6f}")

# Poisson approimation
lam_neu = n_hour * p_neu
P_neu_poisson = 1 - np.exp(-lam_neu)
print(f"Poisson (Lambda=0.72) P(>=1 neutrino decay) = {P_neu_poisson:.6f}")

k2 = np.arange(0, 7)
plt.figure(figsize=(8,4))
plt.bar(k2 - 0.15, binom.pmf(k2, n_hour, p_neu), width=0.3, label="Binomial PMF (n=4, p=0.18)")
plt.bar(k2 + 0.15, poisson.pmf(k2, lam_neu), width=0.3, label="Poisson PMF (Lambda=0.72)")
plt.title("Number of neutrino decays in 1h")
plt.xlabel("k = number of neutrino decays")
plt.ylabel("Probability")
plt.legend()
#plt.show()

