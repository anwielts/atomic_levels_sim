# scipy.constants provides physical constants needed, e.g. vacuum velocity c, atomic mass unit u,...
import scipy.constants as scc
# math module provides mathematical operations like power, squareroot and so on
import math
# numpy needed for numpy.polyfit function
import numpy as np
# import spline interpolator from scipy
from scipy.interpolate import InterpolatedUnivariateSpline
# math functions
import math

from numba import jit, jitclass


# value of bohr magnetron
bohr_magnetron = scc.physical_constants['Bohr magneton'][0]
# reduced planck quantum
h_bar = scc.hbar
# wavelength of laser
# laser_wavelength = Light_Atom_Interaction.wavelength

# class for initializng and configuring magnetic fields
# class Magnetic_Field:

polynom = None

max_dist = 0.2

def maximum_distance_setter(maximum_distance):
    max_dist = maximum_distance
    print(max_dist)
    return max_dist



precision = 4

# fit a polynom of some degree to magnetic field data provided. Fit finishes if least-quare residuals are smaller than 0.01
#@jit(nopython=True)
def magnetic_field_polyfit(position_array, magnetic_strength_array):
    x_data = position_array
    y_data = magnetic_strength_array
    degree = 1
	
    polynom = np.polyfit(x_data, y_data, degree, full=True)
    while polynom[1] > 0.0015:
        degree += 1
        polynom = np.polyfit(x_data, y_data, degree, full=True)
	
    return polynom


# @jit(nopython=True)
def magnetic_field_spline_fit(position_array, magnetic_strength_array):
    spline_inter = InterpolatedUnivariateSpline(position_array, magnetic_strength_array, k=5)
    steps = (position_array[-1] - position_array[0])*math.pow(10, precision) + 1
    x_steps = np.linspace(position_array[0], position_array[-1], steps)

    discrete_interpolation = np.zeros(int(steps))

    for i in range(len(x_steps)):
        discrete_interpolation[i] = (spline_inter(x_steps[i]))

    return discrete_interpolation


@jit(nopython=True)
def spline_fit_field_function(spline, position):
    position_rounded = round(position, precision)
    a = len(spline) - (max_dist - position_rounded) * math.pow(10, precision)
    return spline[int(a)]

@jit(nopython=True)
def max_step_length_fit_field_function(max_step_array, position):
    position_rounded = round(position, precision)
    a = len(max_step_array) - (max_dist - position_rounded) * math.pow(10, precision)
    return max_step_array[int(a)]


# function which takes the loaded or calculated fit polynom and a distance value to calculate the magnetic field strength at this position
@jit(nopython=True)
def magnetic_field_function(fitted_polynom, position):

    function_value = 0
    fit_polynom = fitted_polynom
	
    for i in range(0, len(fit_polynom[0])):
        function_value += fit_polynom[0][i] * (position**(fit_polynom[2]-i-1))

    return float(function_value)
