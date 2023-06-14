# sim-mpc

This repository contains source code for the Model Predictive Control framework __SimMPC__, which can be used to simulate HCI movements using biomechanical models and the physics simulation [MuJoCo](https://mujoco.org/). This framework is described in detail in the [TOCHI submission](https://dl.acm.org/doi/10.1145/3577016) *SimMPC: Simulating Interaction Movements with Model Predictive Control*. All simulation results used in this work are available in the [SIM-MPC Dataset](https://zenodo.org/record/7304381). The simulation uses the [ISO-VR-Pointing Dataset](https://zenodo.org/record/7300062). To obtain the maximum feasible torques used in the models, the [CFAT method](https://github.com/fl0fischer/cfat) was used.

## Paper

### [Simulating Interaction Movements via Model Predictive Control](https://dl.acm.org/doi/10.1145/3577016)

[Click here for a YouTube video](https://youtu.be/6xbYUfsTvaY)

## Cite
If you use our framework/toolbox in your work, please cite this paper as \n
Markus Klar, Florian Fischer, Arthur Fleig, Miroslav Bachinski, and Jörg Müller. 2023. Simulating Interaction Movements via Model Predictive Control. ACM Trans. Comput.-Hum. Interact. 30, 3, Article 44 (June 2023), 50 pages. https://doi.org/10.1145/3577016
```
@article{10.1145/3577016,
author = {Klar, Markus and Fischer, Florian and Fleig, Arthur and Bachinski, Miroslav and M\"{u}ller, J\"{o}rg},
title = {Simulating Interaction Movements via Model Predictive Control},
year = {2023},
issue_date = {June 2023},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {30},
number = {3},
issn = {1073-0516},
url = {https://doi.org/10.1145/3577016},
doi = {10.1145/3577016},
journal = {ACM Trans. Comput.-Hum. Interact.},
month = {jun},
articleno = {44},
numpages = {50},
keywords = {interaction techniques, optimal feedback control, maximum voluntary torques, Simulation, biomechanics, mid-air pointing, model predictive control, AR/VR environments}
}
```

  
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

