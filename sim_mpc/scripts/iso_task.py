''' 
Functions for running a ISO type pointing task.

Authors: Markus Klar
Date: 11.2022
'''
from sim_mpc.core.simulator import Simulator
import sim_mpc.core.utils as utils
import sim_mpc.core.transferfunctions as tf
import sim_mpc.core.visualize_backend as vsb

import numpy as np
import pandas as pd
from pathlib import Path
import logging as log
import os

# Change current working directory to file directory
os.chdir(Path(__file__).parent)
log.info(Path(__file__).parent)

DIRNAME_STUDY = "../data/study/"
DIRNAME_MODEL = "../data/models/"
DIRNAME_RESULTS = "../_results/"
DIRNAME_INITACTS = "../data/initacts/"
TARGETFILE = "../data/targets/iso_targets_15_plane.csv"


def run_iso_task(participant, condition, costfct, paramfolder, outputprefix='simulation', createvideos=True, resettoexperiment=True, moveid=-1, totalid=-1, sim_premovement=False, verbosity=False, use_N = 8, use_r1=-1, use_r2=-1, use_noise_seed_add=0, use_noise=True, debugmode=False):

    # Check if data is available
    utils.check_study_dataset_dir(DIRNAME_STUDY)

    # Create Simulator for the correct model
    simulator = Simulator(f"{DIRNAME_MODEL}/OriginExperiment_{participant}.xml", participant)
    outputprefix += "_" + participant

    # Get paramfile
    paramfile = utils.get_paramfile(costfct, paramfolder)
  
    # Set the cost weights 
    outputprefix = set_cost_weights(simulator, costfct, participant, paramfile, condition, outputprefix, use_r1, use_r2)

    if debugmode:
        simulator.param_dict["debugmode"] = True
        simulator.param_dict["opts"]["disp"] =True
        
    simulator.param_dict["displaysteps"] = False
    simulator.param_dict["verbosity"] = verbosity

    simulator.param_dict["max_iterations"] = 30
    simulator.param_dict["N"] = use_N
    outputprefix += f"_N{use_N}"
    simulator.param_dict["num_steps"] = 20

    if use_r1 >= 0 and use_r2 >= 0:
        outputprefix += f"_r_{use_r1:.8e}_r2_{use_r2:.8e}"


    # Set termination at time step from user data
    if totalid >= 0:
        submovement_times = get_submovement_times(participant, condition)
        
        simulator.param_dict["terminate_at_timestep"]  = int((submovement_times[totalid][1]-submovement_times[totalid][0])//(simulator.param_dict["h"]*simulator.param_dict["num_steps"]))
        simulator.param_dict["continue_after_target_reach"] = True
        
    simulator.param_dict["noise"] = use_noise

    if simulator.param_dict["noise"]:
        simulator.param_dict["signal_dependent_noise"] = 0.103
        simulator.param_dict["constantnoise_param"] = 0.185
        simulator.param_dict["noise_seed"] = 1337 + use_noise_seed_add + moveid
    
    if simulator.param_dict["noise"]:
        outputprefix += "_noise"
    
    simulator.param_dict["stay_at_target"] = 0 
    
    simulator.param_dict["stoptol_vel"] = 0.5

        
    # Outputfolder
    outputfolder = f"{DIRNAME_RESULTS}/{participant}/{outputprefix}/{condition}"

    # Get initial posture and activations
    Xqpos, Xqvel = get_init_postures(participant, condition)

    ACT, EXT = get_act_dact(participant, condition)

    # Setup targets
    # Get shoulder position from user data
    rotshoulder = get_init_shoulder(participant, condition) #getRotShoulder(param, Xqpos)
    
    numrepeats = 5
    skiptrials = 0

    # Skip already simulated movements
    while os.path.exists("{}/{}".format(outputfolder, skiptrials)): 
        skiptrials += 1

    targets, width = get_iso_targets(rotshoulder)
 

    simulator.param_dict["iso_targets"] = True
    #simulator.param_dict["fittsgoals"] = targets

    targets = targets * numrepeats
    width = width * numrepeats

     # Create output directory
    Path(outputfolder).mkdir(parents=True, exist_ok=True)
    
    # Get transferfunction used in the condition
    used_transfer = get_used_transfer(condition, simulator, rotshoulder)

    SSUCCESS = []
    
    if not resettoexperiment:
        xqpos = Xqpos[skiptrials] 
        xqvel = Xqvel[skiptrials]
    
    # Run all movements subsequently
    if moveid < 0:
        for k in range(skiptrials, min(len(targets)-1, len(Xqpos))):
            if resettoexperiment:
                    _, succ,_ = run_movement(simulator, targets[k+1], width[k+1], k, outputfolder, Xqpos[k], Xqvel[k], ACT[k], EXT[k], used_transfer, createvideos=createvideos)
            else:
                if k == skiptrials:
                    # Use initial condition for first trial
                    X, succ ,_ = run_movement(simulator, targets[k+1], width[k+1], k, outputfolder, xqpos, xqvel, ACT[k], EXT[k],  used_transfer, createvideos=createvideos)
                else:
                    X, succ,_ = run_movement(simulator, targets[k+1], width[k+1], k, outputfolder, X[-1].qpos, X[-1].qvel, ACT[k], EXT[k],  used_transfer, createvideos=createvideos)
                        
            SSUCCESS.append(succ)

        df = pd.DataFrame(SSUCCESS)
        df.to_csv(outputfolder + "/SSUCCESS.csv")

    
    # Run only one movement
    else:
        k = moveid
        gID = moveid
        
        if sim_premovement:
                
            simulator.param_dict["stoptol_pos"] = width[k]
            
            simulator.param_dict["init_qpos"] = xqpos
            simulator.param_dict["init_qvel"] = xqvel
            X, QACC, U, XPOS, XVEL, XACC, TORQUE, JN, NFEV, COST,  ACT, EXT,  totaltime, succesfulstart = simulator.run_movement(used_transfer)

        if resettoexperiment:
            _, succ,_ = run_movement(simulator, targets[k+1], width[k+1], gID, outputfolder, Xqpos[totalid], Xqvel[totalid], ACT[totalid], EXT[totalid], used_transfer, createvideos=createvideos)
        else:
            log.error("Can't simulate single movement without reset to experiment start. Aborting.")


def set_cost_weights(simulator, costfct, participant, paramfile, condition, outputprefix, use_r1, use_r2):

    if use_r1 < 0 and use_r2 < 0:
    # Obtain cost weights from param file
        paramcsv = pd.read_csv(paramfile)
        params = paramcsv.loc[paramcsv["condition"]==condition].loc[paramcsv["participant"]==participant]

        use_r1 = simulator.param_dict["r"] = float(params["param_0"])
        use_r2 = float(params["param_1"])
        
    if costfct == "JAC":
        simulator.param_dict["r"] = use_r1
        simulator.param_dict["acceleration_cost_weight"] = use_r2 
        simulator.param_dict["acceleration_cost"] = True
        outputprefix += "_JAC"
    elif costfct == "CTC":
        simulator.param_dict["r"] = use_r1
        simulator.param_dict["commanded_tc_cost"] = True
        simulator.param_dict["commanded_tc_cost_weight"] = use_r2
        outputprefix += "_CTC" 
    else:
        simulator.param_dict["r"] = use_r1

    return outputprefix


def run_movement(simulator, target, width, gID, outputfolder, xqpos, xqvel, act, dact, used_transfer, createvideos, save_results=True):
        
    simulator.param_dict["stoptol_pos"] = width 
    
    simulator.param_dict["init_qpos"] = xqpos
    simulator.param_dict["init_qvel"] = xqvel

    simulator.activation = act 
    simulator.d_activation = dact 

    utils.adjust_thorax_constraints(simulator)
     
    simulator.param_dict["target"] = target

    X, QACC, U, XPOS, XVEL, XACC, TORQUE, JN, NFEV, COST,  ACT, EXT,  totaltime, succesfulstart = simulator.run_movement(used_transfer)

    if save_results:
        log.info('saving...')
        out_folder = utils.export_simulation_to_csv(X, QACC, U, XPOS, XVEL, XACC, TORQUE, JN, NFEV, COST,  ACT, EXT,  simulator, totaltime, succesfulstart, outputfolder, "{}".format(gID))
        log.info("Result saved in " + out_folder)
    
    if createvideos:
        log.info('visualizing...')
        vsb.visualize_run(out_folder)

    return X, succesfulstart, XPOS#


def get_iso_targets(rotshoulder):
    target_df = pd.read_csv(TARGETFILE, index_col=0)
    target_df.reset_index(inplace=True)
    target_df.rename(columns={"0": "Target_x", "1": "Target_y", "2":"Target_z", "3":"Width"}, inplace=True)

    targets = [np.array([-target_df["Target_x"][k],target_df["Target_y"][k],target_df["Target_z"][k]]) + rotshoulder  for k in range(len(target_df.index))]
    width = [target_df["Width"][k] for k in range(len(target_df.index))]

    return targets, width

def get_act_dact(participant, condition): 

    submovtimes = get_submovement_times(participant, condition)

    initial_ActEx_table = pd.read_csv(f'{DIRNAME_INITACTS}/{participant}_0.002s_initacts/{participant}_{condition}_CFAT_initacts.csv', dtype=float)

    initial_ActEx_table = initial_ActEx_table.dropna(subset=[cn for cn in initial_ActEx_table.columns if
                                       (cn.startswith('A_') or cn.startswith('Adot_')) and not cn.endswith('_cp')])

    initial_ActEx_table = initial_ActEx_table.iloc[[(initial_ActEx_table.time - i).abs().argsort()[0] for i in submovtimes[:, 0]]].set_index("time")

    initial_ActEx_table = initial_ActEx_table[[cn for cn in initial_ActEx_table.columns if (cn.startswith('A_') or cn.startswith('Adot_')) and not cn.endswith('_cp')]] 

    ACT = [initial_ActEx_table.loc[i][[col for col in initial_ActEx_table.columns if "A_" in col]].to_numpy() for i in initial_ActEx_table.index]
    # DACT = [initial_ActEx_table.iloc[initial_ActEx_table.index.get_loc(i)-1][[col for col in initial_ActEx_table.columns if "Adot_" in col]].to_numpy() for i in initial_ActEx_table.index]
    DACT = [initial_ActEx_table.loc[i][[col for col in initial_ActEx_table.columns if "Adot_" in col]].to_numpy() for i in initial_ActEx_table.index]

    return ACT, DACT


def get_init_postures(participant="", condition=""):

    indicesfile = f"{DIRNAME_STUDY}/_trialIndices/{participant}_{condition}_SubMovIndices.npy"
    xposfile = f"{DIRNAME_STUDY}/IK/{participant}_{condition}.csv"
    
    submovindices = np.load(indicesfile)
    xposdf = pd.read_csv(xposfile)
    
    jointnames = ['thorax_tx', 'thorax_ty', 'thorax_tz', 'thorax_rx', 'thorax_ry', 'thorax_rz',  'sternoclavicular_r2', 'sternoclavicular_r3', 'unrotscap_r3', 'unrotscap_r2', 'acromioclavicular_r2', 'acromioclavicular_r3', 'acromioclavicular_r1', 'unrothum_r1', 'unrothum_r3', 'unrothum_r2', 'elv_angle', 'shoulder_elv', 'shoulder1_r2', 'shoulder_rot', 'elbow_flexion', 'pro_sup', 'deviation', 'flexion', 'wrist_hand_r1', 'wrist_hand_r3']
    posnames = [jn + "_pos" for jn in jointnames]
    velnames = [jn + "_vel" for jn in jointnames]
    
    startingpoints = xposdf.iloc[submovindices[:,0]]

    Xqpos = [np.array(startingpoints[posnames].loc[index]) for index in startingpoints[posnames].index]
    Xqvel = [np.array(startingpoints[velnames].loc[index]) for index in startingpoints[velnames].index]
    for i in range(len(Xqvel)):
        Xqvel[i][:6] = np.zeros((6,))

    return Xqpos, Xqvel


def get_init_shoulder(participant, condition, ):
    experiment_file_name = f"{DIRNAME_STUDY}/_trialData/Experiment_{participant}_{condition}.csv"
    experiment_info = pd.read_csv(experiment_file_name)
    shoulder = np.array(experiment_info.loc[0, "Shoulder.Position.x":"Shoulder.Position.z"] * np.array([-1, 1, 1]), dtype="float")
    return shoulder

def get_used_transfer(condition, simulator, rotshoulder):
    if "Virtual_Cursor_Ergonomic" in condition:
        simulator.param_dict["origin_i"] = np.array([-0.1, -0.4, 0.45]) + rotshoulder
        simulator.param_dict["origin_o"] = np.array([-0.1, 0, 0.55]) + rotshoulder
        simulator.param_dict["transferdilation"] = 1

        F = [simulator.param_dict["origin_i"], simulator.param_dict["origin_o"], simulator.param_dict["transferdilation"]]
        used_transfer = lambda sim,simulator: tf.outputspace_transfer(F, sim, simulator)
    elif "Virtual_Pad_ID" in condition:
        simulator.param_dict["point_i"] = np.array([-0.1, 0, 0.55]) + rotshoulder
        simulator.param_dict["normal_i"] = np.array([0, 0, -1])

        simulator.param_dict["point_o"] = np.array([-0.1, 0, 0.55]) + rotshoulder
        simulator.param_dict["normal_o"] = np.array([0, 0, -1])

        F = [simulator.param_dict["point_i"], simulator.param_dict["normal_i"], simulator.param_dict["point_o"], simulator.param_dict["normal_o"]]
        used_transfer = lambda sim,simulator: tf.plane_transfer(F, sim, simulator)
    elif "Virtual_Pad_Ergonomic" in condition:
        simulator.param_dict["point_i"] = np.array([-0.1, -0.3, 0.55]) + rotshoulder
        simulator.param_dict["normal_i"] = np.array([0, 0, -1])

        simulator.param_dict["point_o"] = np.array([-0.1, 0, 0.55]) + rotshoulder
        simulator.param_dict["normal_o"] = np.array([0, 0, -1])

        F = [simulator.param_dict["point_i"], simulator.param_dict["normal_i"], simulator.param_dict["point_o"], simulator.param_dict["normal_o"]]
        used_transfer = lambda sim,simulator: tf.plane_transfer(F, sim, simulator)
    else:
        used_transfer = tf.transfer_dflt
    
    return used_transfer

def get_submovement_times(participant, condition):
    submovement_indices = np.load(f"{DIRNAME_STUDY}/_trialIndices/{participant}_{condition}_SubMovIndices.npy")
    table_filename = f"{DIRNAME_STUDY}/IK_raw/{participant}_{condition}.mot"
    trajectories_table = pd.read_csv(table_filename, skiprows=10, delimiter="\t", index_col="time")
    submovement_times = submovement_indices.astype('float64')
    submovement_times[:,0] = trajectories_table.iloc[submovement_indices[:,0]].index
    submovement_times[:,1] = trajectories_table.iloc[submovement_indices[:,1]].index


    return submovement_times
