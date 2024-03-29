# Numerical simulation of transitions between quantum states of an atom.
The main components of this simulation program were developed as a master thesis in the working group of Professor Pohl 
at the University of Mainz. One project of the group aims at decelerating Lithium atoms using a Zeeman slower and 
catch the slowed atoms in a magneto-optical trap (MOT). Therefor, the atom species, simulation parameters and experiental 
setup used in this repository as an example are those of the Lithium experiment.

## Getting Started

Using the setup files in the example folder sim_setup you can run
```
python main.py --sim_params_file sim_setup/sim_parameter_Li_6.json --exp_params_file sim_setup/exp_setup_Li_6.json --atom_params_file sim_setup/atom_parameter_Li_6.json --raw_magnetic_field_data sim_setup/example_magnetic_field.txt --max_step_size sim_setup/example_magnetic_field_maximum_step_length.txt
```
and get simulation results (log file, plots) in a result folder (simulation_results). A short help is provided running the command
```
python main.py -h
```

## Prerequisites

Python 3 has to be installed to use the simulation. All python package prerequisites are listed in the requirements.txt. Install the 
packages via
```
pip install requirements.txt
```

## Authors

* **Andreas Wieltsch** - [anwielts](https://github.com/anwielts)

## License

This project is licensed under the GNU License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments
* Prof. Randolf Pohl, Institute of Physics, University of Mainz
* Prof. Elmar Schömer, Institute of Computer Science, University of Mainz
* Dr. Stefan Schmidt, Institute of Physics, University of Mainz
* Marcel Willig, Institute of Physics, University of Mainz
* Jan Haack, Institute of Physics, University of Mainz
