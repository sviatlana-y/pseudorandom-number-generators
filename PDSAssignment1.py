# YSC2244: Programming for Data Science, 2019-2020, Sem1 - Assignment 1

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats  # to test the distributions


# ** Exercise 1 ** : lcg(m, a, c, x) Linear Congruential Generator (LCG)

def lcg(m, a, c, s):  # the function performs one iteration of the LCG and returns the new seed
    while True:
        s = (a * s + c) % m
        return s


print("The value is:", lcg((2 ** 31 - 1), 48271, 0, 123))

rand_seed = 123


def lcg_rand():  # the function performs one iteration of the LCG following MINSTD
    m = 2 ** 31 - 1
    a = 48271
    c = 0
    global rand_seed  # we declare rand_seed as a global variable in order to use newly generated seeds
    # in later iterations
    rand_seed = (a * rand_seed + c) % m
    return rand_seed


data = [lcg_rand() for x in range(10)]
print("Random numbers following MINSTD:", data)


def lcg_rand_zero_to_one():  # I use this generator in later exercises to get random numbers in the [0;1) interval
    m = 2 ** 31 - 1
    return lcg_rand() / m  # division by m scales the values to produce uniformly distributed random numbers
    # between 0 and 1


data = [lcg_rand_zero_to_one() for x in range(10)]
print("Random numbers from 0 to 1:", data)

# ** Exercise 2 ** : lcg_randint (lb , ub , size) returns a list of 'size' random integers within ['lb', 'ub') range

rand_seed = 42


def lcg_randint(lb, ub, size):  # in the first version I use the modulo operator to bring the results within the
    # desired upper bound
    if lcg_rand() % ub >= lb:  # I make sure the results satisfy the lower bound condition before storing them
        # in the list
        res = [lcg_rand() % ub for x in range(size)]
        return res


def lcg_randint2(lb, ub, size):  # in the second version I use lcg_rand_zero_to_one() to scale the values to fall within
    # the [0;1) interval and multiply them by the upper bound to get them to satisfy the desired upper bound condition
    if lcg_rand_zero_to_one() * ub >= lb:  # I make sure the results satisfy the lower bound
        res = [int(lcg_rand_zero_to_one() * ub) for x in range(size)]
        return res


data = lcg_randint(0, 100, 30)
print("A list of {} random numbers from {} to {}:".format(30, 0, 100), data)
data2 = lcg_randint2(0, 100, 30)
print("A list of {} random numbers from {} to {} (version 2):".format(30, 0, 100), data2)

values_randint1 = lcg_randint(0, 100, size=1000000)
hist_res_randint1 = plt.hist(values_randint1, 100)
plt.title('LCG distribution within [0; 100) range', fontsize=15)
plt.show()

values_randint2 = lcg_randint2(0, 100, size=1000000)
hist_res_randint2 = plt.hist(values_randint2, 100)
plt.title('LCG distribution within [0; 100) range (version 2)', fontsize=15)
plt.show()

# ** Exercise 3 **: uniform_float() returns float in [0,1[ interval following uniform distribution

# Since lcg_rand_zero_to_one() already does its job in the [0,1) interval,
# I generalise the function here to be able to take any interval:

rand_seed = 5


def rand_uniform(a: float, b: float) -> float:  # the function generates random floating numbers in the range [a, b)
    return a + (b - a) * lcg_rand_zero_to_one()


data = [rand_uniform(0, 1) for x in range(10)]
print("A list of 10 random numbers between 0 and 1 from a uniform distribution:", data)

values_uniform = [rand_uniform(0, 1) for x in range(1000000)]
hist_res_uniform = plt.hist(values_uniform, 1000)
plt.title('Uniform distribution within [0; 1) range', fontsize=15)
plt.show()

# Test whether our values differ from a uniform distribution (Kolmogorov-Smirnov test)
ks, p = stats.kstest(np.asarray(values_uniform), 'uniform')
alpha = 1e-3
print("null hypothesis: x is from a uniform distribution")
print("p = {:g}".format(p))
if p < alpha:
    print("The null hypothesis can be rejected for rand_uniform")
else:
    print("The null hypothesis cannot be rejected for rand_uniform")


# It is unlikely that our values did not come from a uniform distribution

# ** Exercise 4 ** : normal_float() returns float in [0,1[ interval following normal (gaussian) distribution

def normal_float() -> float:  # using Kinderman and Monahan ratio-of-uniforms method
    magic_constant = 4 * np.exp(-0.5) / np.sqrt(2.0)  # the magic number used for normal distribution
    mu = 0.0  # the mean of our distribution
    sigma = 1.0  # the standard deviation of our distribution
    while 1.0:
        u1 = lcg_rand_zero_to_one()
        u2 = 1.0 - lcg_rand_zero_to_one()
        z = magic_constant * (u1 - 0.5) / u2
        zz = z * z / 4.0
        if zz <= - np.log(u2):
            break
    return mu + z * sigma


def normal_float2() -> float:  # version without the magic constant
    mu = 0.0  # the mean of our distribution
    sigma = 1.0  # the standard deviation of our distribution
    x1 = 1 - lcg_rand_zero_to_one()
    x2 = 1 - lcg_rand_zero_to_one()
    r = np.sqrt(-2.0 * np.log(x1)) * np.cos(2.0 * np.pi * x2)
    return r * sigma + mu


data = [normal_float() for x in range(10)]
print("A list of 10 random numbers from a standard normal distribution:", data)

values_normal = [normal_float() for x in range(100000)]
hist_res_normal = plt.hist(values_normal, 100)
plt.title('Standard normal distribution', fontsize=15)
plt.show()

# Test whether our values differ from a normal distribution (Kolmogorov-Smirnov test)
k2, p = stats.normaltest(np.asarray(values_normal))
alpha = 1e-3
print("null hypothesis: x is from a normal distribution")
print("p = {:g}".format(p))
if p < alpha:
    print("The null hypothesis can be rejected for normal_float")  # null hypothesis: x comes from a normal distribution
else:
    print("The null hypothesis cannot be rejected for normal_float")

# It is unlikely that our values did not come from a normal distribution

data2 = [normal_float2() for x in range(10)]
print("A list of 10 random numbers from a standard normal distribution (version 2):", data2)

values_normal2 = [normal_float2() for x in range(100000)]
hist_res_normal2 = plt.hist(values_normal2, 100)
plt.title('Standard normal distribution (version 2)', fontsize=15)
plt.show()

# Test whether our values differ from a normal distribution (second implementation)
k2, p = stats.normaltest(np.asarray(values_normal2))
alpha = 1e-3
print("null hypothesis: x is from a normal distribution")
print("p = {:g}".format(p))
if p < alpha:
    print("The null hypothesis can be rejected for normal_float2")  # null hypothesis: x is from a normal distribution
else:
    print("The null hypothesis cannot be rejected for normal_float2")


# It is unlikely that our values did not come from a normal distribution

# ** Exercise 5 **: poisson_int(L) returns integer in [0, L[ interval following poisson distribution

def rand_poisson(lam):  # lam is the expected number of occurrences
    lam_exp = np.exp(-lam)
    p = 1.0  # the probability of k occurrences given lam (initially set at 1,
    # multiplied by a random float in the [0,1) interval each loop)
    k = 0  # the number of occurrences (initially set at 0, incremented by 1 with each loop)
    while p > lam_exp:
        k += 1
        p = p * lcg_rand_zero_to_one()
    else:
        return k - 1


data = [rand_poisson(lam=10) for x in range(10)]
print("A list of 10 random integers with lam of 10 from a poisson distribution:", data)

values_poisson = [rand_poisson(lam=10) for x in range(100000)]
hist_res_poisson = plt.hist(values_poisson, 100)
plt.title('Poisson distribution with lam of 10', fontsize=15)
plt.show()

# test whether my values fit poisson distribution (Kolmogorov-Smirnov test)

print("The mean of my poisson values is:", sum(values_poisson) / len(values_poisson))  # is approximately equal to 10

chisq, p = stats.chisquare(np.asarray(values_poisson))
alpha = 1e-3
print("null hypothesis: the observed difference did not arise by chance")
print("p = {:g}".format(p))
if p < alpha:
    print("The null hypothesis can be rejected for rand_poisson")
else:
    print("The null hypothesis cannot be rejected for rand_poisson")


# It is unlikely that our values did not come from a poisson distribution

# ** Exercise 6 **: and another well-known distribution of your choice.

def rand_pareto(alpha):  # alpha is the shape parameter (tail index)
    u = 1.0 - lcg_rand_zero_to_one()
    return 1.0 / u ** (1.0 / alpha)  # can also be: return beta / u ** (1.0 / alpha)
    # if we want to have the option to set the scale parameter


data = [rand_pareto(alpha=5) for x in range(10)]
print("A list of 10 random numbers from a pareto distribution with the shape parameter of 5:", data)

values_pareto = [rand_pareto(alpha=5) for x in range(100000)]
hist_res = plt.hist(values_pareto, 100)
plt.title('Pareto distribution with shape parameter 5', fontsize=15)
plt.show()

# Test whether our values differ from a pareto distribution (Kolmogorov-Smirnov test)

ks, p = stats.kstest(np.asarray(values_pareto), 'pareto', args=(5,))
alpha = 1e-3
print("null hypothesis: x is from a pareto distribution")
print("p = {:g}".format(p))
if p < alpha:
    print("The null hypothesis can be rejected for rand_pareto")  # null hypothesis: x is from a pareto distribution
else:
    print("The null hypothesis cannot be rejected for rand_pareto")
# It is unlikely that our values did not come from a pareto distribution
