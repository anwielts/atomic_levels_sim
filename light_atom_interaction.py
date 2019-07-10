import math
from numba import jit


# function for calculating the Doppler shift
@jit(nopython=True)
def doppler_shift(velocity_x, velocity_y, velocity_z, init_freq, wavevector_x, wavevector_y, wavevector_z, wavelength):
    '''
    :param velocity_x: Float, x-component of the atom's velocity vector.
    :param velocity_y: Float, y-component of the atom's velocity vector.
    :param velocity_z: Float, z-component of the atom's velocity vector.
    :param init_freq: Float, the initial frequency of the atom. Depends on the current ground state of the atom.
    :param wavevector_x: Float, x-component of the wavevector of the laser light.
    :param wavevector_y: Float, y-component of the wavevector of the laser light.
    :param wavevector_z: Float, z-component of the wavevector of the laser light.
    :param wavelength: Float, the wavelength of the laser.
    :return: Float, the Doppler-shifted frequency.

    Calculates the Doppler shift for a moving atom.
    '''
    reduced_freq = init_freq - (wavevector_x * abs(velocity_x) + wavevector_y * abs(velocity_y) + wavevector_z * abs(velocity_z)) * 2 * (math.pi/wavelength)

    return reduced_freq


# lorentzian probability distribution for calculating the probability of excitment
@jit(nopython=True)
def lorentzian_probability(atom_freq, laser_frequency, laser_detuning, natural_line_width, laser_intensity, laser_sat_intensity):
    '''
    :param atom_freq: Float, the Doppler shifted freuqency of the atom.
    :param laser_frequency: Float, the frequency of the laser.
    :param laser_detuning: Float, the detuning of the laser.
    :param natural_line_width: Float, the natural line width of the excited state.
    :param laser_intensity: Float, the intensity of the laser at the current position of the atom.
    :param laser_sat_intensity: Float, the saturation intensity of the atom.
    :return: Float, excitation probability of the atom.

    Calculates the scatter rate using the lorentzian probability.
    '''

    laser_freq_modus = laser_frequency - 2 * math.pi * laser_detuning

    excitation_probability = 0.5 * (laser_sat_intensity / laser_intensity) * (natural_line_width ** 2) / (4 * (laser_freq_modus - atom_freq) ** 2 + (natural_line_width ** 2) * (1 + laser_sat_intensity / laser_intensity))

    return excitation_probability


@jit(nopython=True)
def laser_intensity_gauss(radius_squared, distance, start_intensity):
    '''
    :param radius_squared: Float, the radial position of the atom in the beam.
    :param distance: Float, the distance of the atom to the laser origin.
    :param start_intensity: Float, the initial intensity of the laser.
    :return: Float, the intensity the atom receives at the current position.

    Calculates the intensity of the laser assuming a Gaussian beam profile.
    '''

    width = distance * (0.0098/0.59) + 0.002
    intensity = start_intensity * math.exp(((radius_squared)/width**2))

    return intensity


def scatter_rate(natural_line_width, atom_freq, laser_frequency, rabi_freq, velocity_z, wavelength):
    '''
    :param natural_line_width: Float, the natural line width of the excited state.
    :param atom_freq: Float, the Doppler shifted freuqency of the atom.
    :param laser_frequency: Float, the frequency of the laser.
    :param rabi_freq: Float, the current Rabi frequency of the atom.
    :param velocity_z: Float, z-component of the atom's velocity vector.
    :param wavelength: Float, the wavelength of the laser.
    :return: Flaot, scatter rate.

    Calculates the scatter rate using formula 9.3 in "Atomic physics" by C. J. Foot
    '''
    return (natural_line_width/2) * (rabi_freq**2 / 2)/((laser_frequency - atom_freq + ((2 * math.pi * velocity_z)/wavelength))**2 + (rabi_freq**2 / 2) + (natural_line_width**2 / 4))