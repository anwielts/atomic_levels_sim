import math
import random
from numba import jit, int32, float64, void
# setting boltzman constant
boltzman_constant = 1.38064852E-23

# class for all distributions needed in the simulation
# class Distributions:

# function calculates probability for a given velocity, v^2 version
#@profile
@jit(cache=True)
def maxwell_boltzmann_distribution_v_2(velocity, mass, temperature):

    P_v = 4*math.pi*math.pow((mass/(2*math.pi*boltzman_constant*temperature)), 1.5)*math.pow(velocity, 2)*math.exp((-mass*math.pow(velocity, 2))/(2*boltzman_constant*temperature))
    #P_v = 4*math.pi*math.pow(velocity, 2)*math.exp((-mass_lithium*math.pow(velocity, 2))/(2*boltzman_constant*temperature))

    return P_v

# function calculates probability for a given velocity, v^3 version
@jit(cache=True)
def maxwell_boltzmann_distribution_v_3(velocity, mass, temperature):

    P_v = 0.5*math.pow((mass/(boltzman_constant*temperature)), 2)*math.pow(velocity, 3)*math.exp((-mass*math.pow(velocity, 2))/(2*boltzman_constant*temperature))
    #P_v = 4*math.pi*math.pow(velocity, 2)*math.exp((-mass_lithium*math.pow(velocity, 2))/(2*boltzman_constant*temperature))

    return P_v

# function calculates a uniformly distributed value between a two values
# @profile
@jit(nopython=True)
def uniform_distribution(start_value, end_value):

    uniform_value = random.uniform(start_value, end_value)
	
    return uniform_value

# function calculates a value from a given gaussian distribution
#@profile	
def gauss_distribution(mu, sigma):

    gauss_value = random.gauss(mu, sigma)
	
    return gauss_value

# function samples numbers from a distribution, in this case from the Maxwell Boltzmann distribution
#@profile
@jit(nopython=True)
def number_sampling(distribution, max_value, start_value, end_value, mass, temperature):

    # first a random value is uniformly drawn between a lower and an upper limit (e.g. minimal and maximal velocity)
    random_val = uniform_distribution(start_value, end_value)
    # a sample is drawn uniformly between 0 and the maximum probability (e.g. maximal probability of a Maxwell Boltzmann distribution for a given atom, velocity range and temperature)
    sample = uniform_distribution(0, max_value)
    # if the sample value is smaller than the value from the Maxwell Boltzmann disribution it is returned, if not number_sampling is recursively called
    if sample < maxwell_boltzmann_distribution_v_3(random_val, mass, temperature):
        return random_val
    else:
        return number_sampling(distribution, max_value, start_value, end_value, mass, temperature)