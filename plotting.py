import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
# for plotting several plots into one pdf
from matplotlib.backends.backend_pdf import PdfPages


def line_plotting(x_values, y_values, x_label, y_label, x_lim_min, x_lim_max, y_lim_min, y_lim_max, starting_time, preserve_color_flag):
    '''
    :param x_values: List of floats, containing the x values for the plot (e.g. distance values)
    :param y_values: List of floats, containing the y values for the plot (e.g. velocity values)
    :param x_label: String, the label for the x-axis.
    :param y_label: String, the label for the y-axis.
    :param x_lim_min: Float, minimum x value which is plotted.
    :param x_lim_max: Float, maximum x value which is plotted.
    :param y_lim_min: Float, minimum y value which is plotted.
    :param y_lim_max: Float, maximum y value which is plotted.
    :param starting_time: Datetime object, starting time of the simulation run.
    :param preserve_color_flag: Boolean, True preserves the color of a given atom between several simulation runs.
    True can only be used if the number of atoms smaller/equal to 10. False enables the plotting of more than 10 atoms.
    :return: Nothing, a matplotlib.pyplot plot object is saved to the folder of the simulation run.

    Plots a line plot of e.g. the velocity value of atoms versus the position. The dependent variable could also
    be the excitation probability or the magnetic field strength. If several dependent variables should be plotted
    depending on the position on the z-axis the eval_plotting function should be used.
    '''

    x_values_dict = defaultdict(list)
    y_values_dict = defaultdict(list)
    color_dict = {0:'green', 1:'blue', 2:'red',
                  3:'gold', 4:'black', 5:'orange',
                  6:'teal', 7:'aqua', 8:'purple', 9:'yellowgreen'}

    pos_counter = 0
    diff = len(x_values)-len(y_values)

    for i in range(0, len(x_values)-diff):
        if x_values[i] == -1:
            pos_counter += 1
        else:
            x_values_dict[pos_counter].append(x_values[i])
            y_values_dict[pos_counter].append(y_values[i])

    for key in x_values_dict:
        if preserve_color_flag is True:
            plt.plot(x_values_dict[key], y_values_dict[key], linewidth=3, color=color_dict[key])
        else:
            plt.plot(x_values_dict[key], y_values_dict[key], linewidth=3)
    plt.xlabel(x_label + ' (m)', fontweight='bold', fontsize=16)
    plt.ylabel(y_label + ' (m/s)', fontweight='bold', fontsize=16)
    # plt.plot((0.5, 0.5), (0, y_lim_max), linestyle='-', color='grey', alpha=0.5)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(x_lim_min, x_lim_max)
    plt.ylim(y_lim_min, y_lim_max)
    plt.tight_layout()
    plt.savefig(
        'simulation_results/' + str(starting_time.strftime("%Y_%m_%d %H_%M_%S")) + "/" + x_label + "_vs_" + y_label + ".png")
    # plt.show()
    plt.close()


def hist_plotting(title, x_label, y_label, x_lim_min, x_lim_max, y_lim_min, y_lim_max, bin_count, legend, start_dist_upper_state, start_dist_lower_state, starting_time, *args):
    '''
    :param title: String, the title of the plot.
    :param x_label: String, the label for the x-axis.
    :param y_label: String, the label for the y-axis.
    :param x_lim_min: Float, minimum x value which is plotted.
    :param x_lim_max: Float, maximum x value which is plotted.
    :param y_lim_min: Float, minimum y value which is plotted.
    :param y_lim_max: Float, maximum y value which is plotted.
    :param bin_count: Integer, number of bins.
    :param legend: List of strings, containing the labels of the two distributions.
    :param start_dist_upper_state: List of floats, containing the distribution of the upper ground state.
    :param start_dist_lower_state: List of floats, containing the distribution of the lower ground state.
    :param starting_time: Datetime object, starting time of the simulation run.
    :param args: Lists of floats, containing velocity distributions.
    :return: Nothing, a matplotlib.pyplot histogram object is saved to the folder of the simulation run.

    Right now only simulations of alkali atoms are supported and therefor the start and final velocity distribution
    of the upper and lower substates of the splitted ground state are interesting for evaluating the performance of
    a Zeeman slower design. This function plots a histogram of the start and final (at the MOT) velocity distribution
    for each of the ground states.
    '''

    color_list = ['blue','red']
    i = 0
    for arg in args:
        plt.hist(arg, alpha=0.4, bins=bin_count, range=(x_lim_min, x_lim_max), label=legend[i], color=color_list[i])
        i += 1
    plt.hist(args, bin_count, histtype='step', range=(x_lim_min, x_lim_max), stacked=True, fill=False, label=[legend[0], 'sum'])
    plt.hist(start_dist_upper_state+start_dist_lower_state, bin_count, histtype='step', range=(x_lim_min, x_lim_max), stacked=True, fill=False,
             label='sum start distributions')
    plt.hist(start_dist_upper_state, bins=bin_count, histtype='step', range=(x_lim_min, x_lim_max), color=color_list[0], label='Start distribution upper state')
    plt.hist(start_dist_lower_state, bins=bin_count, histtype='step', range=(x_lim_min, x_lim_max), color=color_list[1], label='Start distribution lower state')
    # plt.plot(x_axes_upper, np.histogram(start_dist_upper_state, bins=int(x_lim_max/bin_count))[0], linestyle='-', label='Start distribution upper state')
    # plt.plot(x_axes_lower, np.histogram(start_dist_lower_state, bins=int(x_lim_max/bin_count))[0], linestyle='-', label='Start distribution lower state')
    plt.plot((2000, 2000), (0, y_lim_max), label='Zero velocity of upper state', linestyle='--', color='grey', alpha=0.5)
    # plt.axvline(x=2000, label='Zero velocity of upper state', linestyle='-', color='grey', alpha=0.5)
    plt.xlabel(x_label + '(m/s)', fontweight='bold')
    plt.ylabel(y_label, fontweight='bold')
    plt.title(title, fontweight='bold')
    plt.xlim(x_lim_min, x_lim_max)
    plt.ylim(y_lim_min, y_lim_max)
    plt.legend(loc='upper right')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.tight_layout()
    # plt.show()
    plt.savefig('simulation_results/' + str(starting_time.strftime("%Y_%m_%d %H_%M_%S")) + "/" + title + "_" + x_label + "_vs_" + y_label + ".pdf")
    plt.close()


def make_patch_spines_invisible(ax):
    '''
    :param ax: Ax-object.
    :return: Nothing.

    A helper function for the slice_plotting function.
    '''

    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def slice_plotting(slice_positions, slice_pos_vel_upper_gs, slice_pos_vel_lower_gs, min_velocity, max_velocity, number_of_atoms, bin_number, starting_time):
    '''
    :param slice_positions: List of floats, containing the positions at which velocity distributions were observed.
    :param slice_pos_vel_upper_gs: List of floats, containing the velocity values at the slicing positions of the upper ground state.
    :param slice_pos_vel_lower_gs: List of floats, containing the velocity values at the slicing positions of the lower ground state.
    :param min_velocity: Integer, minimum velocity plotted on the x-axis.
    :param max_velocity: Integer, maximum velocity plotted on the x-axis.
    :param number_of_atoms: Integer, number of atoms used in the simulation run.
    :param bin_number: Integer, number of bins.
    :param starting_time: Datetime object, starting time of the simulation run.
    :return: Nothing, several matplotlib.pyplot histogram objects are saved to the folder of the simulation run.
    Number of plots = len(slice_positions)

    Plots histograms of the velocity distribution of the upper and lower ground state at several positions specified
    with slice_positions. Through this the development of the distributions is made visible and optical pumping effects
    can be observed. The x-axes of the plot show the fluorescence frequency of the atom and the two velocity axes
    for the lower and upper ground state, respectively. These two axes are shifted by 228 MHz which is the splitting
    between the two ground states in Lithium-6.
    '''

    # str_plane_slice_pos = [0.0, 0.1, 0.2, 0.3, 0.4, 0.49, 0.495, 0.496, 0.5]
    labelsize_dist_plot = 12
    with PdfPages('simulation_results/' + str(starting_time.strftime(
                "%Y_%m_%d %H_%M_%S")) + "/" + "distribution_development" + ".pdf") as pdf:
        for i in range(len(slice_pos_vel_upper_gs)):
            fig, ax1 = plt.subplots()
            fig.subplots_adjust(top=0.75)
            ax1.set_xlim(-115, 800)
            ax1.set_ylim(0, 0.05 * number_of_atoms)
            ax1.xaxis.set_tick_params(labelsize=labelsize_dist_plot)
            ax1.yaxis.set_tick_params(labelsize=labelsize_dist_plot)
            # ax1.set_title(str(str_plane_slice_pos[i]) + " m", fontweight='bold')
            ax1.set_xlabel("Doppler laser detuning (MHz)", fontweight='bold')
            ax1.set_ylabel("Number of atoms", fontweight='bold')
            ax2 = ax1.twiny()
            ax2.hist(slice_pos_vel_upper_gs[i][1:], alpha=0.8,
                     bins=np.linspace(min_velocity - 100, max_velocity, 2 * bin_number + 2), label='Upper ground state', color='blue')
            ax2.hist(slice_pos_vel_lower_gs[i][1:], alpha=0.8,
                     bins=np.linspace(min_velocity - 100, max_velocity, 2 * bin_number + 2), label='Lower ground state', color='red')
            ax2.plot((2000, 2000), (0, 0.05 * number_of_atoms), label='Zero velocity of upper state', linestyle='--', color='grey',
                     alpha=0.5)
            # ax2.plot((0, 0), (0, 0.05 * n), linestyle='--', color='grey', alpha=0.5)
            ax2.legend(loc='upper right', fontsize='large')
            ax2.set_xlabel("Velocity lower ground state (m/s)", fontweight='bold', color='red')
            ax2.spines["top"].set_color('red')
            ax2.set_xlim(-1000, 7000)
            ax2.xaxis.set_tick_params(labelsize=labelsize_dist_plot, colors='red')
            ax3 = ax1.twiny()
            ax3.set_xlabel("Velocity upper ground state (m/s)", fontweight='bold', color='blue')
            ax3.spines["top"].set_position(("axes", 1.2))
            ax3.spines["top"].set_color('blue')
            make_patch_spines_invisible(ax3)
            ax3.spines["top"].set_visible(True)
            ax3.set_xlim(-3000, 5000)
            ax3.xaxis.set_tick_params(labelsize=labelsize_dist_plot, colors='blue')
            ax3.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
            ax3.spines['top'].set_bounds(0, 5000)
            # plt.show()
            pdf.savefig()
            plt.savefig('simulation_results/' + str(starting_time.strftime(
                "%Y_%m_%d %H_%M_%S")) + "/" + "distribution_development_" + str(slice_positions[i]).replace('.', '_') + "_m.png")
            plt.close()

            # plt.hist(vel_dead_atoms_upper, alpha=0.5, bins=np.linspace(v_min-100, v_max, 2*bin_count+2), label='Dead atoms upper groundstate')
            # plt.hist(vel_dead_atoms_lower, alpha=0.5, bins=np.linspace(v_min-100, v_max, 2*bin_count+2), label='Dead atoms lower groundstate')
            # plt.xlim(-100, 8000)
            # plt.ylim(0, 0.25 * n)
            # plt.xlabel("Velocity (m/s)", fontweight='bold')
            # plt.ylabel("Number of atoms", fontweight='bold')
            # plt.plot((2000, 2000), (0, 0.1*n), label='Zero velocity of upper state', linestyle='--', color='grey',
            #          alpha=0.5)
            # plt.title("End of MOT", fontweight='bold')
            # plt.legend(loc='upper right', fontsize='large')
            # # plt.show()
            # pdf.savefig()
            # plt.close()


def state_occupation_development_plot(ground_state_m_j, excited_state_m_j, excitation_probability_dev, starting_time):
    '''
    :param ground_state_m_j: Float, quantum number m_j of the ground state
    :param excited_state_m_j: Float, quantum number of the excietd state
    :param excitation_probability_dev: List of floats, contains the excitation probability along the
    trajectory of the atom.
    :param starting_time: Datetime object, starting time of the simulation run.
    :return: Nothing, a matplotlib.pyplot plot object is saved to the folder of the simulation run.

    This plot shows the transition between the sub states of the excited and ground state along a trajectory
    of an atom.
    '''

    fig, ax1 = plt.subplots()
    ax1.plot(ground_state_m_j, 'b+')
    ax1.plot(excited_state_m_j, 'b*')
    ax1.set_xlabel('loops')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('states', color='b')
    ax1.tick_params('y', colors='b')
    ax2 = ax1.twinx()
    ax2.plot(excitation_probability_dev, 'r.')
    ax2.set_ylabel('excitation probability', color='r')
    ax2.set_ylim(0, 1)
    ax2.tick_params('y', colors='r')
    fig.tight_layout()
    plt.savefig('simulation_results/' + str(starting_time.strftime("%Y_%m_%d %H_%M_%S")) + "/" + 'excitation_probability_development' + '_vs states' + ".png")
    # plt.show()
    plt.close()


# method for plotting evaluation plots, e.g. the velocity distribution in z-direction at the start of the siumatlion and in the MOT
def eval_plotting(number_of_atoms, min_velocity, max_velocity, bin_count, atoms_in_mot, observing_z_pos, observing_magnetic_field,
                        excitation_freq_development, excitation_probability_development,
                        vel_z_atoms_in_mot, start_z_vel, zeeman_shift, observing_z_vel, starting_time):

    '''
    :param number_of_atoms:
    :param min_velocity: Integer, minimum velocity plotted on the x-axis.
    :param max_velocity: Integer, maximum velocity plotted on the x-axis.
    :param bin_count: Integer, number of bins.
    :param atoms_in_mot: Integer, number of atoms which reached the MOT.
    :param observing_z_pos: List of floats, containing the position values along the z-axis.
    :param observing_magnetic_field: List of floats, containing the magnetic field strengths
    along the trajectory of an atom.
    :param excitation_freq_development: List of floats, containing the excitation frequencies
    along the trajectory of an atom.
    :param excitation_probability_development: List of floats, containing the excitation probability
    along the trajectory of an atom.
    :param vel_z_atoms_in_mot: List of floats, containing the velocities of all atoms which reached the MOT.
    :param start_z_vel: List of floats, containing the start velocities of all atoms which reached the MOT.
    :param zeeman_shift: List of floats, containg the Zeeman shift (frequency shift caused by the magnetic field)
    along the trajectory of an atom.
    :param observing_z_vel: List of floats, containing the velocities along the trajectory of an atom.
    :param starting_time: Datetime object, starting time of the simulation run.
    :return: Nothing, several matplotlib.pyplot plot objects are saved to the folder of the simulation run.

    This function creates several line plots of interesting dependent variables such as the excitation frequency
    or the Zeeman shift depending on the position values of the atom.
    '''

    y_limit = 0.5 * number_of_atoms
    # print some informations to the command line
    print("Atoms in MOT:", atoms_in_mot)
    # print("Excited:", np.count_nonzero(state_information == 1), "ground state:", len(state_information) - np.count_nonzero(state_information == 1), "At least once excited:", len(excitment_list_x))
    # print("z velocity below 0:", np.sum(z_vel < 0, axis=0))

    # plt.plot(observing_specific_atoms)
    # plt.xlabel('Number of excitements')
    # plt.ylabel('velocity (m/s)')
    # # plt.xlim(0, 300)
    # # plt.ylim(475, 505)
    # plt.tight_layout()
    # # plt.show()
    # plt.savefig('simulation_results/' + str(startTime.strftime("%Y_%m_%d %H_%M_%S")) + "/Number_of_excitements_one_atom.pdf")
    # plt.close()

    plt.plot(observing_z_pos, observing_magnetic_field)
    plt.xlabel('z position')
    plt.ylabel('Magnetic field')
    plt.xlim(0.1, 0.6)
    plt.ylim(-0.05, 0.07)
    plt.tight_layout()
    # plt.show()
    plt.savefig('simulation_results/' + str(starting_time.strftime("%Y_%m_%d %H_%M_%S")) + "/observing_magnetic_field.pdf")
    plt.close()

    plt.plot(observing_z_pos, excitation_freq_development)
    plt.xlabel('z position')
    plt.ylabel('excitation_freq_development')
    plt.xlim(0, 1.1 * max(observing_z_pos))
    plt.ylim(1.1 * min(excitation_freq_development), 1.1 * max(excitation_freq_development))
    plt.tight_layout()
    # plt.show()
    plt.savefig('simulation_results/' + str(
        starting_time.strftime("%Y_%m_%d %H_%M_%S")) + "/excitation_freq_development_vs_pos.pdf")
    plt.close()

    plt.plot(observing_z_pos, excitation_probability_development)
    plt.xlabel('z position')
    plt.ylabel('excitation_probability_development')
    plt.xlim(0, 1.1 * max(observing_z_pos))
    plt.ylim(0, 1.1 * max(excitation_probability_development))
    plt.tight_layout()
    # plt.show()
    plt.savefig('simulation_results/' + str(
        starting_time.strftime("%Y_%m_%d %H_%M_%S")) + "/excitation_probability_development_vs_pos.pdf")
    plt.close()

    # plt.plot(observing_z_pos, excitation_probability_development, label='excitation_probability')
    # plt.plot(observing_z_pos, observing_z_vel, label='z-velocity')
    # plt.plot(observing_z_pos, observing_magnetic_field, label='magnetic field strength')
    # plt.plot(observing_z_pos, excitation_freq_development, label='excitation frequency')
    # plt.xlabel('z position')
    # plt.ylabel('diverse')
    # # plt.xlim(0, 300)
    # # plt.ylim(475, 505)
    # plt.tight_layout()
    # #plt.show()
    # plt.savefig('simulation_results/' + str(starting_time.strftime("%Y_%m_%d %H_%M_%S")) + "/diverse_vs_pos.pdf")
    # plt.close()

    # plt.plot(zeeman_shift)
    # plt.xlabel('Number of loops')
    # plt.ylabel('zeeman_shift')
    # # plt.xlim(0, 300)
    # # plt.ylim(475, 505)
    # plt.tight_layout()
    # #plt.show()
    # plt.savefig('simulation_results/' + str(starting_time.strftime("%Y_%m_%d %H_%M_%S")) + "/zeeman_shift.pdf")
    # plt.close()

    # plot of the velocity distribution in z-direction for atoms in the MOT
    plt.hist(vel_z_atoms_in_mot, bins=bin_count)
    plt.xlabel('velocity (m/s)')
    plt.ylabel('count')
    plt.xlim(min_velocity, max_velocity)
    plt.ylim(0, y_limit)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.tight_layout()
    # plt.show()
    plt.savefig(
        'simulation_results/' + str(starting_time.strftime("%Y_%m_%d %H_%M_%S")) + "/z_velocity_distribution_in_mot.pdf")
    plt.close()

    # plot of the velocity distribution in z-direction at the starting point of the atoms
    plt.hist(start_z_vel, bins=bin_count)
    plt.xlabel('velocity (m/s)')
    plt.ylabel('count')
    plt.xlim(min_velocity, max_velocity)
    plt.ylim(0, y_limit)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.tight_layout()
    # plt.show()
    plt.savefig('simulation_results/' + str(starting_time.strftime("%Y_%m_%d %H_%M_%S")) + "/z_vel_start_distribution.pdf")
    plt.close()

    # plot of the velocity distribution in z-direction at the starting point of the atoms
    plt.hist(start_z_vel, alpha=0.4, bins=np.linspace(min_velocity, max_velocity, bin_count), label='start')
    plt.hist(vel_z_atoms_in_mot, alpha=0.4, bins=np.linspace(min_velocity, max_velocity, bin_count), label='MOT')
    plt.xlabel('velocity (m/s)')
    plt.ylabel('count')
    # plt.xlim(v_min, v_max)
    plt.ylim(0, y_limit)
    plt.legend(loc='upper right')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.tight_layout()
    # plt.show()
    plt.savefig('simulation_results/' + str(
        starting_time.strftime("%Y_%m_%d %H_%M_%S")) + "/z_vel_distribution_comparison_start_mot.pdf")
    plt.close()

    print(max_velocity / bin_count)

# plot function for generating ta 3D plot of the simulation
# legacy code, TODO: Include it into simulation
'''
def plot_3d():
    fig = plt.figure(figsize=plt.figaspect(1.0))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(x_init_pos, y_init_pos, z_init_pos, c='red')
    ax.scatter3D(x_pos, y_pos, z_pos, c='b')
    ax.scatter3D(dead_atom_list_x, dead_atom_list_y, dead_atom_list_z, c='black')
    ax.scatter3D(excitment_list_x, excitment_list_y, excitment_list_z, c='g')
    #ax.plot_surface(xx, yy, z, alpha=0.0)
    # Cylinder 1: Tube in which atoms can move
    x=np.linspace(-max_radius, max_radius, 100)
    z=np.linspace(0, aperture_distance, 100)
    Xc, Zc = np.meshgrid(x, z)
    Yc = np.sqrt(max_radius**2-(Xc**2))
    # Draw cylinder 1
    rstride = 20
    cstride = 10
    ax.plot_surface(Xc, Yc, Zc, alpha=0.4, rstride=rstride, cstride=cstride)
    ax.plot_surface(Xc, -Yc, Zc, alpha=0.4, rstride=rstride, cstride=cstride)
    # Cylinder 2: Plotting the laser light
    x_leaser_beam=np.linspace(-Light_Atom_Interaction.beam_size, Light_Atom_Interaction.beam_size, 100)
    z_leaser_beam=np.linspace(0, exp_param_data["cooling_laser_distance"], 100)
    Xc_leaser_beam, Zc_leaser_beam = np.meshgrid(x_leaser_beam, z_leaser_beam)
    Yc_leaser_beam = np.sqrt(Light_Atom_Interaction.beam_size**2-(Xc_leaser_beam**2))
    # Draw clyinder 2
    rstride_leaser_beam = 20
    cstride_leaser_beam = 10
    ax.plot_surface(Xc_leaser_beam, Yc_leaser_beam, Zc_leaser_beam, alpha=0.5, rstride=rstride_leaser_beam, cstride=cstride_leaser_beam, color='r')
    ax.plot_surface(Xc_leaser_beam, -Yc_leaser_beam, Zc_leaser_beam, alpha=0.5, rstride=rstride_leaser_beam, cstride=cstride_leaser_beam, color='r')
    # Cylinder 3: Plotting the zeeman slower
    x_zeeman_slower=np.linspace(-zeeman_radius, zeeman_radius, 100)
    z_zeeman_slower=np.linspace(zeeman_distance, zeeman_distance + zeeman_length, 100)
    Xc_zeeman_slower, Zc_zeeman_slower = np.meshgrid(x_zeeman_slower, z_zeeman_slower)
    Yc_zeeman_slower = np.sqrt(zeeman_radius**2-(Xc_zeeman_slower**2))
    # Draw clyinder 3
    rstride_leaser_beam = 20
    cstride_leaser_beam = 10
    ax.plot_surface(Xc_zeeman_slower, Yc_zeeman_slower, Zc_zeeman_slower, alpha=0.9, rstride=rstride_leaser_beam, cstride=cstride_leaser_beam, color='grey')
    ax.plot_surface(Xc_zeeman_slower, -Yc_zeeman_slower, Zc_zeeman_slower, alpha=0.9, rstride=rstride_leaser_beam, cstride=cstride_leaser_beam, color='grey')
    # ax.quiver(x_pos, y_pos, z_pos, 0, 0, (z_vel/max_z_vel), length=0.1, arrow_length_ratio=0.3, normalize=False, color='blue')
    # ax.quiver(x_pos, y_pos, z_pos, (x_vel/max_x_vel), (y_vel/max_y_vel), (z_vel/max_z_vel), length=0.1, arrow_length_ratio=0.3, normalize=False, color='orange')
    #ax.scatter3D(, , , c='y')
    # draw diaphragm
    p = Circle((0, 0), aperture_radius, alpha=1, fill=False)
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=aperture_distance, zdir="z")
    # draw target volume
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    X = target_radius * np.cos(u)*np.sin(v) + target_center_x
    Y = target_radius * np.sin(u)*np.sin(v) + target_center_y
    Z = target_radius * np.cos(v) + target_center_z
    # Plot the surface with face colors taken from the array we made.
    surf = ax.plot_surface(X, Y, Z, color='b', linewidth=0, alpha=0.2)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    # ax2 = fig.add_subplot(121, projection='3d')
    # ax2.view_init(azim=90, elev=0)
    plt.show()
    #plt.savefig('simulation_results/' + str(startTime.strftime("%Y_%m_%d %H_%M_%S").pdf")
    plt.close()
'''
