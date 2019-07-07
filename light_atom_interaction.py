import math
from numba import jit


# function for calculating the Doppler shift
@jit(nopython=True)
def doppler_shift(velocity_x, velocity_y, velocity_z, init_freq, wavevector_x, wavevector_y, wavevector_z, wavelength):

    reduced_freq = init_freq - (wavevector_x * abs(velocity_x) + wavevector_y * abs(velocity_y) + wavevector_z * abs(velocity_z)) * 2 * (math.pi/wavelength)

    return reduced_freq


# lorentzian probability distribution for calculating the probability of excitment
@jit(nopython=True)
def lorentzian_probability(atom_freq, laser_frequency, laser_detuning, natural_line_width, laser_intensity, laser_sat_intensity):

    laser_freq_modus = laser_frequency - 2 * math.pi * laser_detuning

    excitement_probability = 0.5 * (laser_sat_intensity / laser_intensity) * (natural_line_width ** 2) / (4 * (laser_freq_modus - atom_freq) ** 2 + (natural_line_width ** 2) * (1 + laser_sat_intensity / laser_intensity))

    return excitement_probability


@jit(nopython=True)
def laser_intensity_gauss(radius_squared, distance, start_intensity):
    width = distance * (0.0098/0.59) + 0.002
    intensity = start_intensity * math.exp(((radius_squared)/width**2))

    return intensity


# according to formula 9.3 in "Atomic physics" by C. J. Foot
def scatter_rate(natural_line_width, atom_freq, laser_frequency, rabi_freq, velocity_z, wavelength):

    return (natural_line_width/2) * (rabi_freq**2 / 2)/((laser_frequency - atom_freq + ((2 * math.pi * velocity_z)/wavelength))**2 + (rabi_freq**2 / 2) + (natural_line_width**2 / 4))