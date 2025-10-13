# imports
import matplotlib.pyplot as plt
from scipy.stats import binom
import numpy as np

# Exercise 1: Particle detection efficiency

"""
(b) Write and execute a function to plot the probability distribution for the number of
signals registered if the particle travels through four detectors. (2 points)
"""

p = 0.85
n = 4

def signal_distribution(p, n):
    x = range(n + 1)
    y = [binom.pmf(k, n, p) for k in x]
    plt.bar(x, y)
    plt.xlabel('Number of Signals')
    plt.ylabel('Probability')
    plt.title('Signal Detection Probability Distribution')
    plt.show()
signal_distribution(p, n)

"""
(d) You run your experiment with four detectors and produce 1000 particles which are
sent through the four detector planes. Plot the probability distribution of the number
of detected particles (number of particles with 3 or more signals). Does the width of
the distribution agree with what one would expect from the Poisson distribution? (2
points)
"""

n_particles = 1000
n_detectors = 4


signals_model = np.random.binomial(n_detectors, p, n_particles)
detected_particles = np.sum(signals_model >= 3)

plt.hist(signals_model, bins=np.arange(0, n_detectors+2)-0.5, color='lightgreen', edgecolor='black')
plt.xlabel("Amount of signals")
plt.ylabel("Amount of particles")
plt.title("Distribution of signals for 1000 particles")
plt.show()
