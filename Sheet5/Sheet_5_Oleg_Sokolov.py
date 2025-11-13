# imports
import numpy as np
from scipy.stats import norm, poisson, binom, t

# Exercise 1: P-value examples
"""
a) The standard particle physics theory predicts an electron-to-muon ratio RK to be exactly 1.0.
You measure RK = 0.83 ± 0.06. Calculate the p-value of your result with
respect to the standard theory.
"""
# we check the deviation in both directions => two-tailed
RK = 0.83
RK_theory = 1.0
sigma_RK = 0.06
z_a = (RK - RK_theory) / sigma_RK
p_a = 2 * norm.sf(abs(z_a))
print("(a) two-tailed p =", round(p_a, 5))


"""
(b) I measure earths gravitational acceleration to be g = 9.70±0.10 m/s
2 and you measure
it to be g = 9.90 ± 0.09 m/s
2 km. Calculate the compatibility p-value of the two
measurements.
"""
# comparing two measurements without expected direction => two-tailed
g1, g2 = 9.70, 9.90
s1, s2 = 0.10, 0.09
z_b = (g1 - g2) / np.sqrt(s1**2 + s2**2)
p_b = 2 * norm.sf(abs(z_b))
print("(b) two-tailed p =", round(p_b, 3))

"""
(c) You run an experiment designed to look for a new physics signal. An average of 1.5
background events is predicted during the experiment run period. You observe 6 events,
four times the background prediction. Calculate the p-value of the background-only
hypothesis.
"""

# checking for increase in counts => one-tailed (increase)
lam = 1.5
k_obs = 6
p_c = 1 - poisson.cdf(k_obs - 1, lam)
print("(c) one-tailed p =", round(p_c, 5))

"""
(d) Gun crime has gone up by 20% in the UK this year! claims a tabloid newspaper. These
figures are based on 50 gun incidents in 2019 and 60 gun incidents in 2020. Calculate
the p-value that the gun crime level has actually stayed constant.
"""

# checking for increase in counts => one-tailed (increase)
n_total = 50 + 60
k_2020 = 60
p_d = binom.sf(k_2020 - 1, n_total, 0.5)  
print("(d) one-tailed p =", round(p_d, 3))

"""
(e) A vaccine trial is being performed. During the trial period, without a vaccine, the
average infection rate is 3000 people per 1 million citizens. A group of 8924 people are
given the vaccine and 3 are infected during the trial period. Calculate the p-value with
respect the hypothesis that the vaccine is not effective.
"""

# checking for decrease in counts => one-tailed (decrease)
n = 8924
p0 = 3000 / 1000000
infected = 3
p_e = binom.cdf(infected, n, p0)
print("(e) one-tailed p =", "{:.2e}".format(p_e))

"""
(f) The height of a hockey team and a soccer team are compared in the table below. If
the standard deviation (σ) is unknown, what is the p-value that these samples are
compatible with each other? What is the p-value if the standard deviation of both
parent height distributions are known to be σ = 5 cm?
"""
vb = np.array([187, 185, 183, 176, 190])
fb = np.array([170, 174, 186, 178, 185, 176, 182, 184, 179, 189, 177])

mean_vb = vb.mean()
mean_fb = fb.mean()
n1, n2 = len(vb), len(fb)

# unknown std deviation 
s1 = vb.var(ddof=1)
s2 = fb.var(ddof=1)
s_p = np.sqrt(((n1 - 1)*s1 + (n2 - 1)*s2) / (n1 + n2 - 2))
SE_t = s_p * np.sqrt(1/n1 + 1/n2)
t_val = (mean_vb - mean_fb) / SE_t
df = n1 + n2 - 2
p_f_t = 2 * t.sf(abs(t_val), df)
print("(f) two-tailed p (sigma unknown) =", round(p_f_t, 3))

# known std deviation
sigma = 5
SE_z = np.sqrt(sigma**2/n1 + sigma**2/n2)
z_val = (mean_vb - mean_fb) / SE_z
p_f_z = 2 * norm.sf(abs(z_val))
print("(f) two-tailed p (sigma 5 known) =", round(p_f_z, 3))

# Exercise 2: Radiation safety

"""
You measure the radiation levels in a laboratory using a Geiger counter. You measure 240
counts in a 5 minute interval with your counter, which corresponds to 0.1 microSievert (µSv)
per hour.
(a) Determine a normal 68% interval for your measurement in units of µSv per hour.
"""
# given
counts = 240         # in 5 min
time_min = 5
rate_hour = 0.1      # µSv per hour
limit_year = 1000    # µSv per year (public limit)

counts_hr = counts * (60 / time_min)

# poisson error on counts
# sqrt(N) because of Poisson stats
sigma_counts = np.sqrt(counts)
sigma_counts_hr = sigma_counts * (60 / time_min)

# convert to µSv/h uncertainty (scale with ratio)
sigma_rate = rate_hour * (sigma_counts_hr / counts_hr)

# normal distribution -> 1 sigma interval 68%
lower = rate_hour - sigma_rate
upper = rate_hour + sigma_rate
print("(a) 68% interval =", round(lower,4), "-", round(upper,4), "µSv/h")

"""
(b) Determine a 90% upper confidence limit for your measurement in units of µSv per hour
"""
# for 90% upper, use z=1.28
z_90 = norm.ppf(0.9)
upper90 = rate_hour + z_90 * sigma_rate
print("(b) 90% upper CL =", round(upper90,4), "µSv/h")

"""
(c) Is your 90% confidence limit below the radiation requirement for the general public of
1000µSv per year?
"""

# 1 year = 8760 hours
upper_year = upper90 * 8760
print("(c) 90% upper CL per year =", round(upper_year,2), "µSv/year")

if upper_year < limit_year:
    print("=> Below the public safety limit")
else:
    print("=> Above the limit")


# Exercise 3: Planet identification 

m = 90          # 10^24 kg
dm = 5
d = 52          # 10^6 m
dd = 0.2
rho = -0.6      # correlation between m and d

# measurement vector
x = np.array([m, d])

#covariance matrix (taking into account correlation)
cov = np.array([[dm**2, rho*dm*dd],
                [rho*dm*dd, dd**2]])

inv_cov = np.linalg.inv(cov)

uranus = np.array([86.8, 51.1])
neptune = np.array([102.0, 49.5])

def maha_dist(x, mu, inv_cov):
    diff = x - mu
    return np.sqrt(diff.T @ inv_cov @ diff)

#calculate distances
dist_uranus = maha_dist(x, uranus, inv_cov)
dist_neptune = maha_dist(x, neptune, inv_cov)

print("Mahalanobis distances:")
print(" Uranus  =", round(dist_uranus, 2), "sigmas")
print(" Neptune =", round(dist_neptune, 2), "sigmas")

if dist_uranus < dist_neptune:
    best = "Uranus"
    diff_sigma = dist_neptune - dist_uranus
else:
    best = "Neptune"
    diff_sigma = dist_uranus - dist_neptune

print("\nBest fit planet:", best)
print("Difference to the other one =", round(diff_sigma, 2), "sigmas")

