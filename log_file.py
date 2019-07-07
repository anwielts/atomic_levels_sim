# module for creating pathes, directories and folders
import pathlib

def create_log_file(file_name, atoms_in_mot, excitation_counter, capture_velocity, start_vel_z_mean,
               capture_count_z_velocity, exp_param_config, sim_param_config, running_time, starting_time,
               number_of_bins, max_velocity):
    # create the path to and the directory of the resulting log file
    pathlib.Path('simulation_results/' + str(starting_time.strftime("%Y_%m_%d %H_%M_%S"))).mkdir(parents=True,
                                                                                             exist_ok=True)
    with open('simulation_results/' + str(starting_time.strftime("%Y_%m_%d %H_%M_%S")) + '/simulation_results_log_file.txt',
              'w') as log_file:
        log_file.write("Overview of simulation parameters and results of simulation run on " + str(
            starting_time.strftime("%Y_%m_%d %H_%M_%S")) + "\n\n")

        # write the parameters of the experimental setup JSON to the log file
        log_file.write("Parameters of the experimental setup\n")
        for key in exp_param_config:
            log_file.write(str(key) + ": " + str(exp_param_config[key]) + "\n")

        # write the parameters of the simulation parameter JSON to the log file
        log_file.write("\nParameters of the simulation run\n")
        for key in sim_param_config:
            log_file.write(str(key) + ": " + str(sim_param_config[key]) + "\n")

        log_file.write("\nMagnetic field information from: " + str(file_name) + "\n")

        log_file.write("\nSimulation results:\n")
        # write the run time to the log file
        log_file.write("Overall simulation runtime: " + str(running_time) + "\n")
        # save the number of atoms in the MOT
        log_file.write("Atoms in MOT: " + str(atoms_in_mot) + "\n")
        # save the number of atoms at elast once excited
        # log_file.write("Excited: " + str(np.count_nonzero(state_information == 1)) + ", ground state: " + str(len(state_information) - np.count_nonzero(state_information == 1)) + "\n")
        # log_file.write("At least once excited: " + str(len(excitment_list_x)) + "\n")
        # write the total number of excitments into the log file
        log_file.write("Total count of excitments: " + str(excitation_counter) + '\n')
        log_file.write("Given capturing velocity: " + str(capture_velocity) + '\n')
        log_file.write(
            "Number of atoms with z-velocity below capture velocity: " + str(capture_count_z_velocity) + '\n')
        # log_file.write("Mean velocity of start distribution: " + str(start_vel_mean) + '\n')
        # log_file.write("Mean velocity of after simulation: " + str(sum(total_velocity_after_sim)/len(total_velocity_after_sim)) + '\n')
        # log_file.write("Mean velocity of x-component of start distribution: " + str(start_vel_x_mean) + '\n')
        # log_file.write("Mean velocity of y-component of start distribution: " + str(start_vel_y_mean) + '\n')
        log_file.write("Mean velocity of z-component of start distribution: " + str(
            sum(start_vel_z_mean) / len(start_vel_z_mean)) + '\n')
        # if len(vel_atoms_in_mot) > 0:
        # log_file.write("Mean velocity of atoms in MOT: " + str(sum(vel_atoms_in_mot)/len(vel_atoms_in_mot)) + '\n')
        # if len(vel_z_atoms_in_mot) > 0:
        # log_file.write("Mean velocity of z-component of atoms in MOT: " + str(sum(vel_z_atoms_in_mot)/len(vel_z_atoms_in_mot)) + '\n')
        # log_file.write("Count of atoms with z-component of velocity below capturing velocity: " + str(capture_count_z_vel) + '\n')
        # log_file.write("Count of atoms with total velocity below capturing velocity: " + str(capture_count_total_vel) + '\n')
        log_file.write("Atoms dead if: Outside laser beam\n")
        log_file.write("Bin width in histogram " + str(max_velocity / number_of_bins) + '\n')
        log_file.close()