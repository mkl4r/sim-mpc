# sim-mpc

This repository contains source code for the Model Predictive Control framework __SimMPC__, which can be used to simulate HCI movements using biomechanical models and the physics simulation [MuJoCo](https://mujoco.org/). This framework is described in detail in the [TOCHI](https://dl.acm.org/journal/tochi) submission *SimMPC: Simulating Interaction Movements with Model Predictive Control* (just submitted). All simulation results used in this work are available in the [SIM-MPC Dataset](https://zenodo.org/record/7304381). The simulation uses the [ISO-VR-Pointing Dataset](https://zenodo.org/record/7300062). To obtain the maximum feasible torques used in the models, the [CFAT method](https://github.com/fl0fischer/cfat) was used.
  
## Structure
This repository is divided in three folders:
- $\texttt{core}$ - contains the main code to simulate movements via MPC, optimize cost function parameters, and tools for visualization 
- $\texttt{scripts}$ - contains different scripts to start simulations, compute initial activations using CFAT, and generate plots
- $\texttt{data}$ - contains the biomechanical models as MuJoCo .xml files, optimized cost function parameters, initial activations obtained via CFAT, and targets for a pointing task

## Installation
To install sim-mpc as a package in a virtual environment, clone this repository, change the directory to the root directory and run 
```bash
pip install -e .
```

## Example
A minimal example simulation can be run simply by executing the file run_movement.py
```bash
python scripts/run_movement.py
```

## Contributors
Markus Klar  
Florian Fischer  
Arthur Fleig  
Miroslav Bachinski  
Jörg Müller  
