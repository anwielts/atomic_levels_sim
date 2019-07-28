# module for calculating runtime of simulation
from datetime import datetime
# module for sleeping
import time
# import tqdm for nice progress bars
from tqdm import tqdm
# module for parsing JSON files
import json
# import pickle module for saving python objects
import pickle
# module for creating pathes, directories and folders
import pathlib
# numpy needed for arrays with routines written in C
import numpy as np
# scipy.stats 
import scipy.stats as scs
# scipy.constants provides physical constants needed, e.g. vacuum velocity c, atomic mass unit u,...
import scipy.constants as scc
# math module provides mathematical operations like power, squareroot and so on
import math
# module for calculating mean of lists
from statistics import mean
# random needed for drawing random numbers which are e.g. uniformly or gaussian distributed
import random
# matplotlib for plotting
import matplotlib.pyplot as plt
# for plotting several plots into one pdf
from matplotlib.backends.backend_pdf import PdfPages
# mplot3d needed for 3D plotting
from mpl_toolkits import mplot3d
# this module is needed for plotting circles (opening of aperture,...)
from matplotlib.patches import Circle
# for 3D plotting
import mpl_toolkits.mplot3d.art3d as art3d
# module for using python data object 'dictonary' holding lists
from collections import defaultdict
# Module for evaluating statistics, needed for calculating number of bins in histogram
import statistics
# argparse is used for parsing command line arguments
import argparse

# import of self written modules, first one handles the needed calculations around the Maxwell Boltzmann disribution
# from distributions import Distributions
from distributions import number_sampling
# module for handling the interaction of the laser light with the atoms
# from light_atom_interaction import Light_Atom_Interaction
from light_atom_interaction import doppler_shift, lorentzian_probability, laser_intensity_gauss
# atomic material module holds informations (energy gap between ground and excited state, mass, einstein coefficents,...) about the used atom sorts
from atomic_material import Lithium_6
from atomic_material import Atom
# util contains all functions which are needed in some special case but do not belong to any superordinate topic
# from util import Help_Functions
from util import calculate_p_max, random_velocity_assignment
# this module contains all informations and all calculations regaring the magnetic field 
# from magnetic_field import Magnetic_Field
from magnetic_field import magnetic_field_polyfit, magnetic_field_function, magnetic_field_spline_fit, spline_fit_field_function, max_step_length_fit_field_function, maximum_distance_setter
# import specific plotting methods used in this masterthesis
from plotting import line_plotting, hist_plotting, slice_plotting, eval_plotting, state_occupation_development_plot
# script for logging simulation information to a log file
from log_file import create_log_file
# module for loading all settings
from load_settings import load_files

# import numba for JIT compiling
from numba import jit, int32, float64, void, cuda


# timestep function does the simulation steps for discrete time steps
# @profile
# @jit(nogil=True)
@jit(nopython=True)
def timestep(atom_count, p_max, v_min, v_max, x_min, x_max, y_min, y_max, exciting_freq, time_step, wavelength,
             laser_beam_radius, zeeman_radius, zeeman_distance, target_center_z, bohr_magnetron_scc, spline_fit,
             target_radius, intensity, laser_saturation_intensity, max_step_fit_function, probe_laser_angle,
             slicing_position_array, cutoff_magnetic_field, capture_velocity, lande_factor_excited_state,
             lande_factor_ground_state, freq_shift_splitting, ground_state_quantum_numbers_array,
             quantum_numbers_excited_state):

    cutoff_number = 10000
    magnetic_field_cutoff = cutoff_magnetic_field
    h_bar = scc.hbar
    r_target = target_radius
    loop_counter = 0
    bohr_magnetron = bohr_magnetron_scc
    # initialize mot_counter for counting atoms entering mot, the indices of the atoms in
    # the MOT and three lists for saving velocity components of these atoms + one list for saving velocity
    atoms_in_mot = 0
    # indices_of_atoms_in_mot = []
    vel_x_atoms_in_mot = []
    vel_y_atoms_in_mot = []
    vel_z_atoms_in_mot = []
    vel_z_atoms_in_mot_upper_groundstate = []
    vel_z_atoms_in_mot_lower_groundstate = []
    dead_atoms_upper_groundstate = []
    dead_atoms_lower_groundstate = []
    # vel_atoms_in_mot = []

    # plane_slice_pos = [0.0, 0.1, 0.2, 0.3, 0.38, 0.4, 0.49, 0.495, 0.5]
    plane_slice_pos = slicing_position_array
    plane_slice_flags = []
    plane_slice_upper_groundstate = [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
    plane_slice_lower_groundstate = [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]

    for i in range(len(plane_slice_pos)):
        plane_slice_flags.append(0)

    # lists for saving atoms which have been getting excited at least once
    # excitment_list_x = []
    # excitment_list_y = []
    # excitment_list_z = []
    # counter for counting how many times the atoms were in total excited
    excitation_counter = 0
    # lists for saving dead atoms position
    # dead_atom_list_x = []
    # dead_atom_list_y = []
    # dead_atom_list_z = []
    # list for appending indices of dead atoms
    # dead_atom_list = []

    # capture velocity of MOT according to Stefans document. TODO: Get this via GUI
    # capture_velocity = 120
    # number of atoms which have a total velocity/z component
    # of the velocity below the capture velocity of the MOT when entering the MOT
    capture_count_total_vel = 0
    capture_count_z_vel = 0
    count_vel_z_below_0 = 0
    count_vel_z_0_50 = 0
    count_vel_z_51_100 = 0
    count_vel_z_101_150 = 0
    count_vel_z_151_200 = 0
    count_vel_z_201_250 = 0
    count_vel_z_251_300 = 0
    wavevector_x = 0
    wavevector_y = 0
    wavevector_z = -1

    observing_specific_atoms = []
    start_x_vel_atoms_in_mot = []
    start_x_vel = []
    start_y_vel_atoms_in_mot = []
    start_y_vel = []
    start_z_vel_atoms_in_mot = []
    start_z_vel = []
    start_z_vel_upper_state_atoms_in_mot = []
    start_z_vel_lower_state_atoms_in_mot = []

    # initializing lists with 1 for observing several variables. Needed for compiling through Numba
    observing_z_vel = [0.0]
    observing_z_pos = [0.0]
    observing_magnetic_field = [0.0]
    excitation_freq_development = [0.0]
    excitation_probability_development = [0.0]
    zeeman_shift = [0.0]
    ground_state_quantum_numbers = ground_state_quantum_numbers_array
    # quantum_numbers_excited_state = [-1.5, -0.5, 0.5, 1.5]

    excited_state_quantum_numbers = [0.0]

    light_beam_radius_squared = laser_beam_radius**2
    zeeman_radius_squared = zeeman_radius**2
    threshold = 1E-35
    mean_free_path_bigger = 0
    mean_free_path_smaller = 0
    print(atom_count)
    # for every atom do the simulation steps
    for i in range(0, atom_count):
        if i % 1000 == 0:
            print("Atom number", i)
        # print("\nAtom number--------------------------------", i, "\n")
        mot_flag = 0
        for f in range(len(plane_slice_flags)):
            plane_slice_flags[f] = 0
        x_pos = random.uniform(x_min, x_max+threshold)
        y_pos = random.uniform(y_min, y_max+threshold)
        z_pos = 0.0

		# only z velocity component
        x_velocity = 0.0
        y_velocity = 0.0
        z_velocity = number_sampling(1, p_max, v_min, v_max, mass_lithium_6, temperature)

        # for testing set the z-component of the velocity to a fixed value
        # z_velocity = 100 + i*75
        # z_velocity = 100 + i*100
        # z_velocity = 500
        # z_velocity = random.uniform(v_min, v_max)

        # 2 pi emission
        # total_velocity = number_sampling(1, p_max, v_min, v_max, mass_lithium_6, temperature)
        # x_velocity, y_velocity, z_velocity = random_velocity_assignment(total_velocity)
        # print(x_velocity, y_velocity, z_velocity)

        # conus velocity distribution
        # total_velocity = number_sampling(1, p_max, v_min, v_max, mass_lithium_6, temperature)
        # x_velocity = 0.002 * total_velocity
        # y_velocity = 0.002 * total_velocity
        # z_velocity = 0.998 * total_velocity

        start_x_velocity = x_velocity
        start_y_velocity = y_velocity
        start_z_velocity = z_velocity

        start_x_vel.append(start_x_velocity)
        start_y_vel.append(start_y_velocity)
        start_z_vel.append(start_z_velocity)

        excitation_frequency = exciting_freq
        atom_dead = 1

        # TODO: Divide time step from JSON by life time of excited state for normalizing time steps with a factor
        # Choose time step in a way that minimum free path length is at least 0.5 mm/0.0005 m
        # minimum_path_length = 1E-6
        minimum_path_length = max_step_length_fit_field_function(max_step_fit_function, z_pos - target_center_z)
        initial_atom_time_step = minimum_path_length/(z_velocity + threshold)
        # initial_atom_time_step = 2.7E-8
        atom_time_step = initial_atom_time_step

        # sublevel is the magnetic quantum number m_J. In the ground state the value varies between 0.5 and -0.5
        # if random.random() < 1.0:
        if random.random() < 0.667:
        # if i == 0:
            current_groundstate = 1
            initial_state = 1
        else:
        # if i == 1:
            current_groundstate = 0
            initial_state = 0

        # lande_factor_ground_state = 2.0023
        # lande_factor_excited_state = 1.335

        if i == 3:
            observing_specific_atoms.append(z_velocity)

        total_flight_distance = 0

        # if the atom is not dead do the steps
        while atom_dead != 0:
            # [0.01, 0.02, 0.97] means 1% sigma minus, 2% pi and 97% sigma plus light
            laser_light_polarisation = [0.0, 0.0, 1.0]
            # Choose time step in a way that minimum free path length is at least 0.5 mm/0.0005 m
            # minimum_path_length = max_step_length_fit_field_function(max_step_fit_function, z_pos - target_center_z)
            # print("Min path length: ", minimum_path_length, "Z position: ", z_pos)
            # initial_atom_time_step = minimum_path_length/(z_velocity + threshold)
            # atom_time_step = initial_atom_time_step
            total_flight_distance += z_pos
            # freq_shift_splitting = [-228E6, 0.0]
            frequency_shift_list = [0.0, 0.0, 0.0, 0.0]
            current_excitation_freq = [0.0, 0.0, 0.0, 0.0]
            excitation_prob = [0.0, 0.0, 0.0, 0.0]
            loop_counter += 1
            if atom_count < cutoff_number:
                observing_z_pos.append(z_pos)
                observing_z_vel.append(z_velocity)
            # print(z_velocity)
            # position updating for all atoms before calculating new velocity depending of effects occuring
            x_pos += x_velocity * atom_time_step
            y_pos += y_velocity * atom_time_step
            z_pos += z_velocity * atom_time_step

            # derive the squared x- and y-position for the following comparisons if the position in the xy-plane is beyond some (geometrical) limit
            x_y_pos_component_squared = x_pos**2 + y_pos**2

            # check if atoms move outside the laser beam or hit the wall of the MOT
            if x_y_pos_component_squared > light_beam_radius_squared or z_pos > 0.62 or z_pos < 0.0 or z_velocity < 0.0:
                # set the index to 0 (=dead)
                atom_dead = 0
                # if z_velocity < 0.0 and current_groundstate == 1:
                #     print(z_pos, start_z_velocity, z_velocity, x_y_pos_component_squared)

                if current_groundstate == 0:
                    dead_atoms_lower_groundstate.append(z_velocity)
                if current_groundstate == 1:
                    dead_atoms_upper_groundstate.append(z_velocity + 2000)

                if atom_count < cutoff_number:
                    observing_z_pos.append(z_pos)
                    observing_z_vel.append(z_velocity)
                    observing_z_pos.append(-1)
                    observing_z_vel.append(-1)

                continue

            # is the atom is not hitting the rear wall of the MOT
            if z_pos < target_center_z + target_radius:

                # check if atom is inside the beam of the laser and if the frequency for exciting the atom is equal to the frequency of the laser
                if x_y_pos_component_squared < light_beam_radius_squared:
                    for a in range(len(quantum_numbers_excited_state)):
                        frequency_shift_list[a] = doppler_shift(x_velocity, y_velocity, z_velocity,
                                                                                 excitation_frequency -
                                                                                 freq_shift_splitting[
                                                                                     current_groundstate],
                                                                                 wavevector_x, wavevector_y,
                                                                                 wavevector_z, wavelength)
                        if quantum_numbers_excited_state[a] - ground_state_quantum_numbers[current_groundstate] == 0 or quantum_numbers_excited_state[a] - ground_state_quantum_numbers[current_groundstate] == 1 or quantum_numbers_excited_state[a] - ground_state_quantum_numbers[current_groundstate] == -1 :
                            if z_pos > zeeman_distance:
                                zeeman_shift_freq = (-bohr_magnetron * (quantum_numbers_excited_state[a] * lande_factor_excited_state - ground_state_quantum_numbers[current_groundstate] * lande_factor_ground_state) * spline_fit_field_function(spline_fit, z_pos - target_center_z))/h_bar
                                frequency_shift_list[a] += zeeman_shift_freq
                            excitation_prob[a] = lorentzian_probability(frequency_shift_list[a], laser_frequency,
                                                                laser_detuning, natural_line_width, laser_intensity_gauss(x_y_pos_component_squared, z_pos, intensity),
                                                                laser_saturation_intensity)

                    # excitation_index = -1
                    excitation_prob_sum = 0
                    if spline_fit_field_function(spline_fit, z_pos - target_center_z) < magnetic_field_cutoff:
                        laser_light_polarisation = [1.0, 1.0, 1.0]
                    for b in range(len(laser_light_polarisation)):
                        excitation_prob[current_groundstate + b] *= laser_light_polarisation[b]
                        excitation_prob_sum += excitation_prob[current_groundstate+b]
                    # print("Exc. Prob. Sum:", excitation_prob_sum)
                    random_number_for_lorentz = random.random()
                    # print(spline_fit_field_function(spline_fit, z_pos - target_center_z), "Polarisation:",
                    #       laser_light_polarisation)
                    # print(excitation_prob, excitation_prob_sum, random_number_for_lorentz, current_groundstate)
                    if random_number_for_lorentz < excitation_prob[current_groundstate]/excitation_prob_sum:
                        excitation_index = current_groundstate
                    elif excitation_prob[current_groundstate]/excitation_prob_sum < random_number_for_lorentz < excitation_prob[current_groundstate + 1]/excitation_prob_sum:
                        excitation_index = current_groundstate + 1
                    elif excitation_prob[current_groundstate + 1] / excitation_prob_sum < random_number_for_lorentz < excitation_prob[current_groundstate + 2] / excitation_prob_sum:
                        excitation_index = current_groundstate + 2
                    # print(current_groundstate, random_number_for_lorentz, excitation_prob[0]/excitation_prob_sum, excitation_prob[1]/excitation_prob_sum, excitation_prob[2]/excitation_prob_sum, excitation_prob[3]/excitation_prob_sum)
                    # excitation_probability = excitation_prob[excitation_index]
                    excitation_probability = excitation_prob_sum
                    # print(current_groundstate, excitation_prob)
                    # print("High field regime")
                    # print(excitation_index, excitation_probability)
                    if atom_count < cutoff_number:
                        excitation_probability_development.append(excitation_probability)
                        ground_state_quantum_numbers.append(current_groundstate)
                        excited_state_quantum_numbers.append(quantum_numbers_excited_state[excitation_index] + 4)
                        excitation_freq_development.append(current_excitation_freq[excitation_index])
                        observing_z_pos.append(z_pos)
                        observing_z_vel.append(z_velocity)
                        observing_magnetic_field.append(spline_fit_field_function(spline_fit, z_pos - target_center_z))

                    for i in range(len(plane_slice_pos)):
                        if plane_slice_pos[i] < z_pos and plane_slice_flags[i] != 1:
                            if current_groundstate == 0:
                                plane_slice_lower_groundstate[i].append(z_velocity)
                            if current_groundstate == 1:
                                plane_slice_upper_groundstate[i].append(z_velocity + 2000)

                            plane_slice_flags[i] = 1

                    # set the time step to the natural life time due to the natural line width
                    rnd_number = random.random()
                    excitation_time_step = ((-1/((excitation_probability + threshold) * (natural_line_width))) * math.log(rnd_number + 1E-35))
                    # print("Exc time step", excitation_time_step)
                    # print(excitation_time_step < initial_atom_time_step)
                    # print(excitation_time_step, excitation_probability, rnd_number, natural_line_width)
                    if excitation_time_step < initial_atom_time_step:
                        if excitation_index == 3:
                            current_groundstate = 1
                        elif excitation_index == 0:
                            current_groundstate = 0
                        elif excitation_index == 1 or excitation_index == 2:
                            if random.random() < 0.667:
                                current_groundstate = 1
                            else:
                                current_groundstate = 0

                        # print("Excited number", excitation_index)
                        # print("Ground number", current_groundstate)

                        # de-excitation
                        excitation_counter += 1
                        mean_free_path_smaller += 1
                        # print("Atom time step", atom_time_step)
                        atom_time_step = excitation_time_step
                        # momentum kick against movement direction (z-direction)
                        z_velocity -= h_bar * ((2 * math.pi)/(wavelength * mass_lithium_6))
                        # RIGHT: take two random numbers u and v between 0 and 1
                        u = random.random()
                        v = random.random()
                        # for equally distributed points on a sphere use this two formulas. Source: http://mathworld.wolfram.com/SpherePointPicking.html
                        random_number_theta = 2 * math.pi * u
                        random_number_phi = math.acos(2 * v - 1)
                        # Assigning velocity components as in the spherical coordinate system
                        x_velocity -= math.cos(random_number_theta) * math.sin(random_number_phi) * scc.h/(mass_lithium_6 * wavelength)
                        y_velocity -= math.sin(random_number_theta) * math.sin(random_number_phi) * scc.h/(mass_lithium_6 * wavelength)
                        z_velocity -= math.cos(random_number_phi) * scc.h/(mass_lithium_6 * wavelength)

                        # x_velocity -= math.cos(random_number_theta) * math.sin(random_number_phi) * (2 * math.pi/wavelength) * scc.h/(2 * math.pi)
                        # y_velocity -= math.sin(random_number_theta) * math.sin(random_number_phi) * (2 * math.pi/wavelength) * scc.h/(2 * math.pi)
                        # z_velocity -= math.cos(random_number_phi) * (2 * math.pi/wavelength) * scc.h/(2 * math.pi)

                    else:
                        minimum_path_length = max_step_length_fit_field_function(max_step_fit_function, z_pos - target_center_z)
                        # print("Min path length: ", minimum_path_length, "Z position: ", z_pos)
                        initial_atom_time_step = minimum_path_length / (z_velocity + threshold)
                        # atom_time_step = initial_atom_time_step
                        atom_time_step = 27E-9
                        mean_free_path_bigger += 1

                    if i == 3:
                        observing_specific_atoms.append(z_velocity)
            # atoms entering mot, therefor save number of entering atoms and their velocity components 					
            if target_center_z <= z_pos <= target_center_z + r_target and x_y_pos_component_squared < r_target**2 and mot_flag == 0:
                mot_flag = 1
                atoms_in_mot += 1
                vel_x_atoms_in_mot.append(x_velocity)
                start_x_vel_atoms_in_mot.append(start_x_velocity)
                vel_y_atoms_in_mot.append(y_velocity)
                start_y_vel_atoms_in_mot.append(start_y_velocity)
                vel_z_atoms_in_mot.append(z_velocity)
                # vel_z_atoms_in_mot.append(z_velocity*math.cos(probe_laser_angle))
                # vel_z_atoms_in_mot.append(z_velocity + 150)
                # vel_z_atoms_in_mot.append(z_velocity + 1850)
                start_z_vel_atoms_in_mot.append(start_z_velocity)

                if initial_state == 1:
                    start_z_vel_upper_state_atoms_in_mot.append((start_z_velocity + 2000))
                else:
                    start_z_vel_lower_state_atoms_in_mot.append(start_z_velocity)

                if current_groundstate == 1:
                    # vel_z_atoms_in_mot_upper_groundstate.append(z_velocity)
                    vel_z_atoms_in_mot_upper_groundstate.append(z_velocity + 2000)
                else:
                    vel_z_atoms_in_mot_lower_groundstate.append(z_velocity)

                if atom_count < cutoff_number:
                    observing_z_pos.append(-1)
                    observing_z_vel.append(-1)
                
                if z_velocity < capture_velocity:
                    capture_count_z_vel += 1

                if z_velocity < 0:
                    count_vel_z_below_0 += 1
                if 0 < z_velocity < 51:
                    count_vel_z_0_50 += 1
                if 51 < z_velocity < 101:
                    count_vel_z_51_100 += 1
                if 101 < z_velocity < 151:
                    count_vel_z_101_150 += 1
                if 151 < z_velocity < 201:
                    count_vel_z_151_200 += 1
                if 201 < z_velocity < 251:
                    count_vel_z_201_250 += 1
                if 251 < z_velocity < 301:
                    count_vel_z_251_300 += 1

                total_flight_distance += z_pos
                atom_dead = 0
                continue

    # print runtime to console before plotting
    # runtime = datetime.now() - startTime
    # print(datetime.now() - startTime)
    print("Number of atoms in lower groundstate", len(vel_z_atoms_in_mot_lower_groundstate))
    print("Excitations per distance: ", excitation_counter/z_pos)
    print("Excitations:", excitation_counter)
    print("Capture count z vel:", capture_count_z_vel)
    print("Below 0: ", count_vel_z_below_0, "0-50:  ", count_vel_z_0_50, "  51-100:  ", count_vel_z_51_100, "  101-150:  ", count_vel_z_101_150, "  151-200:  ", count_vel_z_151_200, "  201-250:  ", count_vel_z_201_250, "  251-300:  ", count_vel_z_251_300)
    print("Mean free path smaller than mag:", mean_free_path_smaller, "Bigger:", mean_free_path_bigger)
    a = 0
    b = 0
    for p in range(len(plane_slice_lower_groundstate)):
        a += len(plane_slice_lower_groundstate[p])

    for p in range(len(plane_slice_upper_groundstate)):
        b += len(plane_slice_upper_groundstate[p])

    print(a + b)

    return atoms_in_mot, excitation_counter, capture_velocity, capture_count_z_vel, observing_z_pos, observing_magnetic_field, excitation_freq_development, excitation_probability_development, observing_z_vel, vel_x_atoms_in_mot, vel_y_atoms_in_mot, vel_z_atoms_in_mot, vel_z_atoms_in_mot_upper_groundstate, vel_z_atoms_in_mot_lower_groundstate, start_z_vel_atoms_in_mot, start_z_vel_upper_state_atoms_in_mot, start_z_vel_lower_state_atoms_in_mot, zeeman_shift, loop_counter, start_z_vel, ground_state_quantum_numbers, quantum_numbers_excited_state, plane_slice_upper_groundstate, plane_slice_lower_groundstate, dead_atoms_upper_groundstate, dead_atoms_lower_groundstate


if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser(description='Load necessary information from files')
    parser.add_argument('--sim_params_file',
                        help='Specify the JSON-file containing information about the simulation parameters')
    parser.add_argument('--exp_params_file',
                        help='Specify the JSON-file containing information about the experiment setup')
    parser.add_argument('--atom_params_file',
                        help='Specify the JSON-file containing information about the atom parameters')
    parser.add_argument('--raw_magnetic_field_data',
                        help='Specify the location of the TXT-file containing the raw magnetic field data')
    parser.add_argument('--max_step_size',
                        help='Specify the location of the TXT-file containing the maximum step size for a given magnetic field')
    args = parser.parse_args()

    sim_param_data, exp_param_data, atomic_data, spline_fit, file_name_magnetic_field, max_step_length_file, maximum_distance = load_files(args.sim_params_file,
                                                                                                                                           args.exp_params_file,
                                                                                                                                           args.atom_params_file,
                                                                                                                                           args.raw_magnetic_field_data,
                                                                                                                                           args.max_step_size)
    '''
    sim_param_data, exp_param_data, atomic_data, spline_fit, file_name_magnetic_field, max_step_length_file, maximum_distance = load_files("C:/Users/ACW/Documents/JGU/NawiInf/Kurse/MA/Code/particle_simulation_ma/JSONs/sim_parameter_test_1.json",
                                                                                                            "C:/Users/ACW/Documents/JGU/NawiInf/Kurse/MA/Code/particle_simulation_ma/JSONs/exp_setup_test_1.json",
                                                                                                            "C:/Users/ACW/Documents/JGU/NawiInf/Kurse/MA/Code/particle_simulation_ma/JSONs/atom_parameter_test_1.json",
                                                                                                            "C:/Users/ACW/Documents/JGU/NawiInf/Kurse/MA/Code/particle_simulation_ma/magnetic_field_measurement/MinusFitListtxt.txt",
                                                                                                            "C:/Users/ACW/Documents/JGU/NawiInf/Kurse/MA/Code/particle_simulation_ma/magnetic_field_measurement/maximum_step_length_MinusFitListtxt.txt")

    print(maximum_distance)

    # mass of observed atom
    mass_lithium_6 = Lithium_6.mass_lithium_six
    # necessary excitment wavelength of the atom to excite it from ground to excited state
    exciting_wavelength = Lithium_6.wavelength_ground_excited
    # necessary excitment frequency of the atom to excite it from ground to excited state
    exciting_freq = scc.c/exciting_wavelength + 76E6
    # some constants
    # h_bar = scc.hbar
    bohr_magnetron = scc.physical_constants['Bohr magneton'][0]
    boltzman_constant = 1.38064852E-23
    # temperature at which atom species vaporises
    temperature = sim_param_data['temperature']
    # number of observed atoms
    n = sim_param_data['particle_number']
    # minimal considered velocity
    v_min = sim_param_data['velocity_min']
    # maximal considered velocity
    v_max = sim_param_data['velocity_max']
    # threshold for limits of used distributions
    threshold = 0.000001
    # minimal and maximal starting positions of atoms
    y_min = exp_param_data["center_atomic_source"] - 0.5 * exp_param_data["y_expansion_atomic_source"]
    y_max = exp_param_data["center_atomic_source"] + 0.5 * exp_param_data["y_expansion_atomic_source"]
    x_min = exp_param_data["center_atomic_source"] - 0.5 * exp_param_data["x_expansion_atomic_source"]
    x_max = exp_param_data["center_atomic_source"] + 0.5 * exp_param_data["x_expansion_atomic_source"]
    # length of zeeman slower
    slower_length = exp_param_data["zeeman_slower_length"]
    # duration of whole simulation
    duration_time = sim_param_data['sim_duration_time']
    # step size of one time step
    time_step = sim_param_data['time_step']
    # for i in range(0, duration_time, time_step):
        # timestep_list.append(i)
    # radius of aperture
    aperture_radius = exp_param_data["aperture_diameter"]/2
    # distance of aperture from origin
    aperture_distance = exp_param_data["aperture_distance"]
    # maximum radius in which atoms can move
    max_radius = 0.25
	# radius of nozzle
    nozzle_radius = exp_param_data["nozzle_diameter"]/2
	# noozle length
    nozzle_length = exp_param_data["nozzle_length"]
	# zeeman slower radius
    zeeman_radius = exp_param_data["zeeman_slower_diameter"]/2
	# zeeman slower distance
    zeeman_distance = exp_param_data["zeeman_slower_distance"]
	# zeeman slower length
    zeeman_length = exp_param_data["zeeman_slower_length"]
    # target spherical volume - atoms in this volume add to the atoms cloud velocitiy distribution
    # target_center_x = exp_param_data["mot_distance"]
    # target_center_y = exp_param_data["mot_distance"]
    target_center_x = exp_param_data["center_atomic_source"]
    target_center_y = exp_param_data["center_atomic_source"]
    target_center_z = exp_param_data["mot_distance"]
    print(target_center_z)
    target_radius = exp_param_data["mot_radius"]
	# total length of experimental setup
    total_length = exp_param_data["mot_distance"] + exp_param_data["mot_radius"]
    # number of bins in histogram
    # k = sqrt(n)
    # bin_count = round(math.sqrt(n))
    # Sturges Rule:
    # bin_count = round(1 + 3.3 * math.log(n, 10))
    # educated guessing
    bin_count = 80

    # laser properties
    natural_line_width = 2 * math.pi * 5.87E6
    intensity = sim_param_data['slower_laser_intensity']
    wavelength = scc.c/sim_param_data['slower_laser_frequency']
    laser_frequency = sim_param_data['slower_laser_frequency']
    laser_detuning = sim_param_data['slower_laser_detuning']
    laser_saturation_intensity = sim_param_data['slower_laser_saturation'] * intensity
    saturation_parameter = sim_param_data['slower_laser_saturation']
    print("Saturation", saturation_parameter)
    decay_time = 1/natural_line_width
    photon_per_second = intensity/(scc.h * laser_frequency)
    area_laser = math.pi * (sim_param_data['slower_laser_diameter']/2)**2
    area_atom = math.pi * Lithium_6.radius_atom**2
    photon_per_second_and_atom = photon_per_second/(area_laser/area_atom)

    laser_beam_radius = sim_param_data['slower_laser_diameter']/2
    probe_laser_angle = sim_param_data['probe_laser_angle']
    slicing_positions = sim_param_data['positions_for_slicing']
    magnetic_field_cutoff = sim_param_data['B_field_cutoff']
    capture_vel = sim_param_data['capture_velocity']

    # properties of the atom
    lande_factors_exc_state = atomic_data['lande_factor_excited_state']
    lande_factors_ground_state = atomic_data['lande_factor_ground_state']
    freq_offset = atomic_data['frequency_offset_ground_state']
    ground_state_quantum_numbers = atomic_data['mf_ground_state_quantum_numbers']
    exc_state_quantum_numbers = atomic_data['mf_excited_state_quantum_numbers']

    # test area, new JSON parameters read in
    print(sim_param_data['capture_velocity'])
    print(sim_param_data['positions_for_slicing'])
    print(sim_param_data['laser_polarisation'])
    print(sim_param_data['B_field_cutoff'])
    print("Lande factors", atomic_data['lande_factor_excited_state'], atomic_data['lande_factor_ground_state'])
    print("Frequency offset", atomic_data['frequency_offset_ground_state'])
    print("Quantum numbers ground state", atomic_data['mf_ground_state_quantum_numbers'])
    print("Quantum numbers excited state", atomic_data['mf_excited_state_quantum_numbers'])

    p_max = calculate_p_max(n, v_min, v_max, mass_lithium_6, temperature)
    # function call of timestep
    startTime = datetime.now()
    # line breaks used for better readability
    atoms_in_mot, excitation_counter, capture_velocity, capture_count_z_velocity, observing_z_position,\
    observing_magnetic_field, excitation_freq_development, excitation_probability_development, observing_z_velocity,\
    vel_x_atoms_in_mot, vel_y_atoms_in_mot, vel_z_atoms_in_mot, vel_upper_groundstate, vel_lower_groundstate,\
    start_z_vel_atoms_in_mot, start_vel_upper_state, start_vel_lower_state, zeeman_shift, loop_Counter, start_vel_z,\
    ground_state_m_J, excited_state_m_J, vel_z_plane_slices_upper_gs, vel_z_plane_slices_lower_gs, vel_dead_atoms_upper,\
    vel_dead_atoms_lower = timestep(n, p_max, v_min, v_max, x_min, x_max, y_min, y_max, exciting_freq, time_step,
                                    wavelength, laser_beam_radius, zeeman_radius, zeeman_distance, target_center_z,
                                    bohr_magnetron, spline_fit, target_radius, intensity, laser_saturation_intensity,
                                    max_step_length_file, probe_laser_angle, slicing_positions, magnetic_field_cutoff,
                                    capture_vel, lande_factors_exc_state, lande_factors_ground_state, freq_offset,
                                    ground_state_quantum_numbers, exc_state_quantum_numbers)

    runtime = datetime.now() - startTime
    print(runtime)
    print("Number of total loops:", loop_Counter)
    # create output folder or if folder already exists, create log folder for simulation run
    create_log_file(file_name_magnetic_field, atoms_in_mot, excitation_counter, capture_velocity, start_vel_z,
                   capture_count_z_velocity, exp_param_data, sim_param_data, runtime, startTime,
                   bin_count, v_max)
    print("Writing velocity distribution in trap to file")
    pathlib.Path('simulation_results/' + str(startTime.strftime("%Y_%m_%d %H_%M_%S"))).mkdir(parents=True,
                                                                                             exist_ok=True)
    with open('simulation_results/' + str(startTime.strftime("%Y_%m_%d %H_%M_%S")) + '/mot_vel_distribution.txt',
              'w') as mot_file:
        for i in range(len(vel_x_atoms_in_mot)):
            mot_file.write(str(vel_x_atoms_in_mot[i]) + ";" + str(vel_y_atoms_in_mot[i]) + ";" + str(vel_z_atoms_in_mot[i]) + "\n")
    print("Plotting...")
    # deprecated since line_plotting is a better way for plotting line plots
    # eval_plotting(number_of_atoms, v_min, v_max, bin_count, atoms_in_mot, observing_z_pos, observing_magnetic_field,
    #                         excitation_freq_development, excitation_probability_development,
    #                         vel_z_atoms_in_mot, start_z_vel, zeeman_shift, observing_z_velocity, startTime)
    line_plotting(observing_z_position, observing_z_velocity, 'z position', 'z velocity', 0.0, 0.55, 0.0, 1000.0, startTime, False)
    # line_plotting(observing_z_position, excitation_freq_development, 'z position', 'excitation_freq_development', 0.0, 0.7, -1E10, 1E10, startTime)
    line_plotting(observing_z_position, excitation_probability_development, 'z position', 'excitation probability', 0.0, 0.7, 0.0, 0.55, startTime, False)
    print(laser_detuning)
    print("Average velocity of atoms in trap center: ", sum(vel_z_atoms_in_mot) / len(vel_z_atoms_in_mot))
    print(len(start_z_vel_atoms_in_mot), len(vel_z_atoms_in_mot))
    print(min(start_vel_upper_state))

    str_plane_slice_pos = sim_param_data['positions_for_slicing']

    slice_plotting(str_plane_slice_pos, vel_z_plane_slices_upper_gs, vel_z_plane_slices_lower_gs, v_min, v_max,
                       n, bin_count, startTime)

    hist_plotting('Upper groundstate vs lower groundstate', 'velocity', 'count', 0, 8000, 0, 0.1*n, bin_count, ["Upper state", "Lower state"], start_vel_upper_state, start_vel_lower_state, startTime, vel_upper_groundstate, vel_lower_groundstate)
    # state_occupation_development_plot(ground_state_m_J, excited_state_m_J, excitation_probability_development, startTime)
