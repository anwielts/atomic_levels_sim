# numpy needed for arrays with routines written in C
import numpy as np
# module for parsing JSON files
import json
# import pickle module for saving python objects
import pickle
# module for creating pathes, directories and folders
import pathlib
# from magnetic_field import Magnetic_Field
from magnetic_field import magnetic_field_spline_fit, spline_fit_field_function, maximum_distance_setter


# setting of all variables and assigning them corresponding values in __init__ function
def load_files(sim_path, exp_path, atom_path, magn_path, max_step_path):
    # create lists for storing the magnetic field strength and the corresponding distance value
    magnetic_field_distance = []
    magnetic_strength = []

    max_distance = 0

    with open(magn_path) as magnetic_file:
        header_line = next(magnetic_file)
        for line in magnetic_file:
            line = line.split(";")
            magnetic_field_distance.append(float(line[0]))
            magnetic_strength.append(float(line[1]) * 1E-4)

        max_distance = magnetic_field_distance[-1]

        magnetic_file.close()

    maximum_distance_setter(max_distance)

    # get file to magnetic field/maxiumum step length txt-file by splitting the path file string at the ifle ending (e.g split path/test.txt at '.' to get path/test as the file name)
    path_file_mag = magn_path.split(".")
    path_file_max_step_length = max_step_path.split(".")

    # create a path object using the pathlib module for the file name now with .pickle as file
    # pickle objects can save python objects without further transformtions. E.g. one can simply save an array with all values in it to a pickle, load it from there and use it.
    file_name_mag = pathlib.Path(path_file_mag[0] + ".pickle")
    file_name_max_step_length = pathlib.Path(path_file_max_step_length[0] + ".pickle")

    # check if a pickle with the polynomial fit from the magnetic field information already exists. If so, then load this fit.
    if file_name_mag.exists():
        print("Loading magnetic field pickle...")
        with open(path_file_mag[0] + ".pickle", 'rb') as load_magn_fit:
            spline_fit = pickle.load(load_magn_fit)

    # if the pickle with the polynomial fit does not exists, load magnetic field strength and the corresponding distance from file, call the fit function from magnetic_field.py.
    # Save the fit as a pickle for later use in additional simulation runs with the same magnetic field.
    else:
        # call fit function of class Magnetic_Field from magnetic_field.py. Increases degree of polynom until some error bound is exceeded
        spline_fit = magnetic_field_spline_fit(magnetic_field_distance, magnetic_strength)

        # save polynomial fit with all its information to a pickle object
        with open((path_file_mag[0] + ".pickle"), 'wb') as pickled_magn_fit:
            pickle.dump(spline_fit, pickled_magn_fit, protocol=pickle.HIGHEST_PROTOCOL)

    if file_name_max_step_length.exists():
        print("Loading max step length pickle...")
        with open(path_file_max_step_length[0] + ".pickle", 'rb') as load_max_step_length_fit:
            max_step_length_fit = pickle.load(load_max_step_length_fit)

    else:
        max_step_length_fit = []
        indizes = []
        with open(magn_path) as magnetic_file:
            header_line = next(magnetic_file)
            for line in magnetic_file:
                line = line.split(";")
                magnetic_field_distance.append(float(line[0]))
                magnetic_strength.append(float(line[1]) * 1E-4)

            magnetic_file.close()

            distance = float(magnetic_field_distance[-1] - magnetic_field_distance[0])
            distance_steps = np.linspace(0, distance, len(spline_fit))
            # print(spline_fit_field_function(spline_fit, distance_steps[0])+1E-35/spline_fit_field_function(spline_fit, distance_steps[2])+1E-35)
            # print(len(distance_steps))
            # print(spline_fit_field_function(spline_fit, distance_steps[100000]))
            for i in range(len(distance_steps)):
                b_initial = spline_fit[i]
                # print("Initial B field", b_initial)
                for j in range(i + 1, len(distance_steps)):
                    # print("Change  in B field", (b_initial/spline_fit[j]))
                    if (b_initial / (spline_fit[j] + 1E-35)) < 0.999 or (b_initial / (spline_fit[j] + 1E-35)) > 1.001:
                        # print("High difference")
                        max_step_length_fit.append(((j - i) / 1E4) * distance)
                        indizes.append(i)
                        break
                    if j == len(distance_steps) - 1:
                        max_step_length_fit.append(1E-6)
            max_step_length_fit.append(1E-6)

            with open((path_file_max_step_length[0] + ".pickle"), 'wb') as pickled_max_step_length_fit:
                pickle.dump(max_step_length_fit, pickled_max_step_length_fit, protocol=pickle.HIGHEST_PROTOCOL)

    print(spline_fit_field_function(spline_fit, (0.4999 - 0.5)))
    print(len(spline_fit))
    print(len(max_step_length_fit))

    # load simulation parameters from simulation file
    with open(sim_path) as sim_file:
        sim_param_data = json.load(sim_file)
        # print(sim_param_data)
        sim_file.close()

    # load the experimental setup parameters fron the corresponding file
    with open(exp_path) as exp_file:
        exp_param_data = json.load(exp_file)
        # print(exp_param_data)
        exp_file.close()

    with open(atom_path) as atom_file:
        atom_data = json.load(atom_file)
        atom_file.close()

    return sim_param_data, exp_param_data, atom_data, spline_fit, file_name_mag, max_step_length_fit, max_distance
