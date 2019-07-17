import math
import random
from distributions import maxwell_boltzmann_distribution_v_2, maxwell_boltzmann_distribution_v_3

from numba import jit, int32, float64, void

# This class is a collection of all helper functions which cannot be added thematically to any of the other modules
# class Help_Functions:

# @profile
# @jit -> makes it slower, check why??
@jit(nopython=True)
def random_velocity_assignment(complete_velocity):
    '''
    :param complete_velocity: Float, velocity value.
    :return: Floats, the three velocity components in x-, y- and z-direction.

    Divides the complete velocity (magnitude of the velocity vector) into three components and
    assigns these to the three velocity components (x, y, z). Assigning follows a spherical
    distribution.
    '''
    vel = complete_velocity
    # FALSE: draw random numbers between -0.5*math.pi and 0.5*math.pi and 0 and 2
    # times math.pi (spherical coordinate system) for dividing velocity into the three coordinates components
    # random_number_theta = random.uniform(-0.5*math.pi-threshold, 0.5*math.pi+threshold)
    # random_number_phi = random.uniform(0, 2*math.pi+threshold)

    # RIGHT: take two random numbers u and v between 0 and 1
    u = random.random()
    v = random.random()
    # for equally distributed points on a sphere use this two formulas.
    # Source: http://mathworld.wolfram.com/SpherePointPicking.html
    random_number_theta = 2 * math.pi * u
    random_number_phi = math.acos(v)

    # Assigning velocity components as in the spherical coordinate system
    x_velocity_component = math.cos(random_number_theta) * math.sin(random_number_phi) * vel
    y_velocity_component = math.sin(random_number_theta) * math.sin(random_number_phi) * vel
    z_velocity_component = math.cos(random_number_phi) * vel

    # return x_vel, y_vel, z_vel
    return x_velocity_component, y_velocity_component, z_velocity_component


# simple function that calculates maximum probability of a Maxwell Boltzmann distribution for a
# veloctity range, an atom mass and a temperature
@jit(cache=True)
def calculate_p_max(n, v_min, v_max, atom_mass, temperature):
    '''
    :param n: Integer, number of atoms.
    :param v_min: Integer, the minimum velocity.
    :param v_max: Integer, the maximum velocity.
    :param atom_mass: Float, the mass of the atoms in the simulation.
    :param temperature: Float, the evaporation temperature of the atoms.
    :return: Float, the maximum probability of the Maxwell Boltzmann distribution for the
    provided velocities, atom mass and temperature.

    Calculates the maximum probability which then can be used for the acceptance-rejection method
    for drawing random numbers.
    '''
    p_max = 0
    
    # for loop for 0 to number of atoms
    for i in range(0, n):
        # draw a random veloctity uniformly distributed between minimal and maximal velocity
        random_vel = random.uniform(v_min, v_max)

        # if probability of the drawn velocity inserted into the Maxwell Boltzmann distribution is
        # higher than the highest probability this probability is set to be the highest one
        if maxwell_boltzmann_distribution_v_3(random_vel, atom_mass, temperature) > p_max:
            p_max = maxwell_boltzmann_distribution_v_3(random_vel, atom_mass, temperature)

    return p_max