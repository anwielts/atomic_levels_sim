import scipy.constants as scc
import random
# module for creating dictonaries which hold dictionaries
import collections

# classes for providing the necessary informations about the atom species. A new atom species can easily be added as a new class
class Lithium_6:
    wavelength_ground_excited = 670.977E-9 # later comment this out, information is contained in transition dict
    radius_atom = 145E-12
    mass_lithium_six = 6.015 * scc.physical_constants['atomic mass constant'][0]
    niveau_transition_dict = collections.defaultdict(dict)
    
	
    fine_niveau_transition_dict = collections.defaultdict(dict)
    # fine_niveau_transition_dict[0][0.5] = scc.c/670.992E-9
    fine_niveau_transition_dict[0][1.5] = scc.c/670.977E-9
    hyperfine_niveau_transition_dict = collections.defaultdict(dict)
    # hyperfine_niveau_transition_dict[1.5][0.5] = 446789502.616E6
    # hyperfine_niveau_transition_dict[1.5][1.5] = 446789528.716E6
    # hyperfine_niveau_transition_dict[0.5][0.5] = 446789730.821E6
    # hyperfine_niveau_transition_dict[0.5][1.5] = 446789756.942E6
    #hyperfine_niveau_transition_dict[1.5][2.5] = 446799571.082E6
    hyperfine_niveau_transition_dict[1.5][1.5] = 446799573.977E6
    #hyperfine_niveau_transition_dict[1.5][0.5] = 446799575.689E6
    hyperfine_niveau_transition_dict[0.5][1.5] = 446799802.200E6
    #hyperfine_niveau_transition_dict[0.5][0.5] = 446799803.912E6
    groundstates_numbers = [0.5, 1.5]
	
    einstein_coeff_b12 = random.random()
    einstein_coeff_b21 = random.random()
    einstein_coeff_a21 = random.random()
	
	
class Lithium_7:
    wavelength_ground_excited = 670.961E-9
    mass_lithium_six = 2 * scc.physical_constants['atomic mass constant'][0]
    fine_niveau_transition_dict = collections.defaultdict(dict)
    # fine_niveau_transition_dict[0][0.5] = scc.c/670.977E-9
    fine_niveau_transition_dict[0][1.5] = scc.c/670.961E-9
    hyperfine_niveau_transition_dict = collections.defaultdict(dict)
    hyperfine_niveau_transition_dict[2][1] = 446799771.121E6
    hyperfine_niveau_transition_dict[2][2] = 446799862.994E6
    hyperfine_niveau_transition_dict[1][1] = 446800574.608E6
    hyperfine_niveau_transition_dict[1][2] = 446800666.494E6
    hyperfine_niveau_transition_dict[2][3] = 446809874.988E6
    hyperfine_niveau_transition_dict[2][2] = 446809884.450E6
    hyperfine_niveau_transition_dict[2][1] = 446809890.263E6
    hyperfine_niveau_transition_dict[1][2] = 446810687.944E6
    hyperfine_niveau_transition_dict[1][1] = 446810693.757E6
    hyperfine_niveau_transition_dict[1][0] = 446810969.516E6
    groundstates_numbers = [1, 2]
    
	

# class for atoms, needed for object oriented approach
class Atom:
    
    def __init__(self, x_position, y_position, z_position, x_velocity, y_velocity, z_velocity, exciting_frequency, alive, excited):
        self.x_position = x_position
        self.y_position = y_position
        self.z_position = z_position
        self.x_velocity = x_velocity
        self.y_velocity = y_velocity
        self.z_velocity = z_velocity
        self.exciting_frequency = exciting_frequency
        self.alive = alive
        self.excited = excited
		
