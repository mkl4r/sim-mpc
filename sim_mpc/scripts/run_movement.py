''' 
Script to run an example pointing movement.

Authors: Markus Klar
Date: 11.2022
'''
from sim_mpc.core.simulator import Simulator
import sim_mpc.core.utils as utils
import sim_mpc.core.visualize_backend as vsb
import sim_mpc.scripts.iso_task as iso

import pandas as pd
import logging as log
import numpy as np

log.basicConfig(level=log.ERROR, format=' %(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":

    # Available conditions, participants and costfunctions
    CONDITIONS = ['Virtual_Cursor_ID_ISO_15_plane', 'Virtual_Cursor_Ergonomic_ISO_15_plane', 'Virtual_Pad_ID_ISO_15_plane', 'Virtual_Pad_Ergonomic_ISO_15_plane']
    USER = ['U1', 'U2', 'U3', 'U4', 'U5', 'U6']
    COSTFCTS = ["DC","CTC","JAC"]

    # Mainfolder for data and output, has to contain the initial posture files
    modelfolder = "../data/models"
    paramfolder = "../data/parameters"
    target_file = "../data/targets/iso_targets_15_plane.csv"

    # Output folder
    out_name = "testrun"
    out_folder = '../_results'

    # Choose user model, condition and costfunction
    user = USER[0]
    condition = CONDITIONS[0]
    costfct = COSTFCTS[2]

    # Model file
    model_file = f"{modelfolder}/OriginExperiment_{user}.xml"
    model_short = user

    # Create simulator
    simulator = Simulator(model_file, model_short)

    # Define simulation options 
    #print(list(simulator.param_dict.keys())) # Uncomment to see all possible parameters
    simulator.param_dict["num_steps"] = 20
    simulator.param_dict["max_iterations"] = 30
    simulator.param_dict["N"] = 8

    # Cost weight parameter file
    paramfile = utils.get_paramfile(costfct, paramfolder)

    # Load and set cost weights
    paramcsv = pd.read_csv(paramfile)
    params = paramcsv.loc[paramcsv["condition"]==condition].loc[paramcsv["participant"]==user]

    simulator.param_dict["r"] = float(params["param_0"])
    cost_add = float(params["param_1"])

    # Set cost function
    simulator.param_dict["commanded_tc_cost"] = costfct == "CTC"
    simulator.param_dict["commanded_tc_cost_weight"] = cost_add

    simulator.param_dict["acceleration_cost"] = costfct == "JAC"
    simulator.param_dict["acceleration_cost_weight"] = cost_add

    # Set initial posture and activation
    simulator.param_dict["init_qpos"] = np.array([ 0.15451593,  1.47094627,  0.27911389,  0.30494186, -1.53176394,
        0.33613836, -0.1999706 ,  0.08469811, -0.08469811,  0.1999706 ,
       -0.04048998,  0.32722577,  0.14708573, -0.14708573, -0.32722577,
        0.04048998,  0.44532354,  0.82632295, -0.44532354, -0.18092168,
        1.33108504,  0.65566144,  0.26331317, -0.00377068,  0.30060259,
       -0.00377068])

    simulator.param_dict["init_qvel"] = np.array([ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.00395193, -0.00167376,  0.00167376, -0.00395193,
        0.00080008, -0.00646669, -0.00290682,  0.00290682,  0.00646669,
       -0.00080008, -0.01030709, -0.01632963,  0.01030709,  0.03782211,
        0.05073423,  0.06985526,  0.06307044,  0.02625947,  0.05839269,
        0.02625947])

    simulator.activation = np.array([ 0.12301466,  0.30900481, -0.31716495,  0.38470065, -0.03781062,
        0.08961887, -0.23574512])
    simulator.d_activation = np.array([  3.99522067,   6.64765178,  -3.85148581,  10.0758441 ,
       -11.68752819,  -1.23287101,  -8.20764071])


    # Fix thorax
    utils.adjust_thorax_constraints(simulator)

    # Set target
    simulator.param_dict["target"] = np.array([-0.01310367,  1.4340095 ,  0.80739   ])

    # Set termination conditions
    simulator.param_dict["stoptol_pos"] = 0.025
    simulator.param_dict["stoptol_vel"] = 0.5

    # RUN SIMULATION
    X, QACC, U, XPOS, XVEL, XACC, TORQUE, JN, NFEV, COST, ACT, EXT, totaltime, successful = simulator.run_movement() #used_transfer
    ###

    # Save and visualize simulation
    log.debug('saving...')
    out_folder = utils.export_simulation_to_csv(X, QACC, U, XPOS, XVEL, XACC, TORQUE, JN, NFEV, COST, ACT, EXT, simulator, totaltime, successful, out_folder, out_name)
    log.debug("Result saved in " + out_folder)

    vsb.visualize_run(out_folder)