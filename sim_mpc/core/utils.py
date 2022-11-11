''' 
Collection of helper functions for the sim-mpc project.

Authors: Markus Klar, Florian Fischer
Date: 11.2022
'''
import logging as log
import os
import pathlib
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import mujoco_py
import numpy as np
import pandas as pd

log.basicConfig(level=log.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')


def updatebodypos(simulator, name, pos):
    body_pos = simulator.model.body_pos
    body_pos[simulator.model.body_name2id(name)] = pos
    
    
def filename_adds(simulator, add_csv):
    add = str(simulator.param_dict["h"]) + '_' +  str(simulator.param_dict["iterations"]) + '_' + str(simulator.param_dict["N"]) + '_' + str(simulator.param_dict["num_steps"]) + '_' + simulator.param_dict["model_short"] 
    
    if simulator.param_dict["control_cost"]:
        add += f'_r_{simulator.param_dict["r"]:.6}'
    if simulator.param_dict["commanded_tc_cost"]:
        add += f'_ctc_{simulator.param_dict["commanded_tc_cost_weight"]:.6}'
    if simulator.param_dict["acceleration_cost"]:
        add += f'_jac_{simulator.param_dict["acceleration_cost_weight"]:.6}'
    if not simulator.param_dict["commanded_tc_cost"] and not simulator.param_dict["acceleration_cost"]:
        add += '_dc'
    if add_csv:
        add += '.csv'
    return add

def get_paramfile(costfct, paramfolder):
    # Weight parameter file
    paramfile = "params" 

    if costfct == "JAC":
        paramfile += "_jac" 
    elif costfct == "CTC":
        paramfile += "_ctc"
    elif costfct == "DC":
        paramfile += "_dc"
    else:
        log.error("Wrong costfunction. Use DC, CTC or JAC.")
        raise NotImplementedError

    paramfile += ".csv"

    return os.path.join(paramfolder, paramfile)


def get_motor_noise(ctrl, simulator):
    """Sample motor noise. Former: get_signaldependent_noise
    
    """
    return np.random.normal(0.0, scale=np.sqrt(((simulator.param_dict["signal_dependent_noise"] * ctrl) ** 2) + (simulator.param_dict["constantnoise_param"] ** 2))) 

def muscle_activation_model_secondorder(activation, activation_first_derivative, control, dt, t_activation=0.04, t_excitation=0.03):
    """
    Returns activation/torque signal and its first derivative at next time step using second-order (muscle) activation dynamics.
    For details see https://journals.physiology.org/doi/pdf/10.1152/jn.00652.2003.
    """
    assert len(activation) == len(control)  # == sim.sim.model.nu

    return activation + dt * activation_first_derivative,\
           -(dt/(t_excitation * t_activation)) * activation + \
           (1 - ((dt*(t_excitation + t_activation))/(t_excitation * t_activation))) * activation_first_derivative + \
           dt * (control/(t_excitation * t_activation))

# Adjust thorax constraints to keep initial thorax posture
def adjust_thorax_constraints(simulator):
    for column_name in ["thorax_rx", "thorax_ry", "thorax_rz", "thorax_tx", "thorax_ty", "thorax_tz"]:
        simulator.model.eq_data[
            (simulator.model.eq_obj1id[:] == simulator.model.joint_name2id(column_name)) & (
                    simulator.model.eq_type[:] == 2), 0] = simulator.param_dict["init_qpos"][
            simulator.model.joint_name2id(column_name)]
        
# Export simulation results to csv
def export_simulation_to_csv(X, QACC, U, XPOS, XVEL, XACC, TORQUE, JN, NFEV, COST,  ACT, DACT,  simulator, totaltime, is_successful, folder, main_path, out_name = '', generate_complete_csv = True, ):
    # Create mainfolder for export
    if len(out_name) == 0: 
        folder += '/' + filename_adds(simulator, False)
    else:
        folder += '/' + out_name
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
    
    # Extract joint positions and velocities form mujoco states
    steps = len(X)
    
    log.debug(f"Steps: {steps}, number of joints: {len(X[0].qpos)}")
    
    X_qpos = np.zeros((steps,len(X[0].qpos)))
    X_qvel = np.zeros((steps,len(X[0].qvel)))
    X_qacc = np.zeros((steps-1,len(QACC[0])))
    
    for i in range(steps):
        X_qpos[i] = np.array(X[i].qpos)
        X_qvel[i] = np.array(X[i].qvel)
        if i < steps-1:
            X_qacc[i] = np.array(QACC[i])

    if ACT == []:
        ACT = np.zeros((steps,U[0].shape[0]))
    if DACT == []:
        DACT = np.zeros((steps,U[0].shape[0]))
 
    #S ave parmeters
    np.save(folder + '/param.npy', simulator.param_dict) 

    if generate_complete_csv:
        trajectory_table = data2table(X, X_qacc, U, XPOS, XVEL, TORQUE, simulator.param_dict, ACT, DACT, main_path)
        trajectory_table.to_csv(folder + '/complete.csv')
    
    return folder

# Generates csv with complete information
def data2table(X, X_qacc, U, XPOS, XVEL, TORQUE, param, ACT, DACT, main_path):

    time = np.array(range(len(X))) * param["h"]  

    trajectory_table_columns = ["time"] + [i + suffix for suffix in ('_pos', '_vel', '_acc') for i in param["physical_joints"]] + [i + suffix for suffix in ('_pos', '_vel', '_acc') for i in param["virtual_joints"]]  + [
        'end-effector_xpos_x', 'end-effector_xpos_y', 'end-effector_xpos_z', 'end-effector_xvel_x', 'end-effector_xvel_y', 'end-effector_xvel_z', 'target_xpos_x',
        'target_xpos_y', 'target_xpos_z'] + ['thorax_rx_pos', 'thorax_ry_pos', 'thorax_rz_pos',
                                             'thorax_tx_pos', 'thorax_ty_pos', 'thorax_tz_pos'] + [
                                                'A_' + i for i in param["physical_joints"]]+ [
                                   i + '_frc' for i in param["physical_joints"]] + ['target_switch'] + ['shoulder_offset_x', 'shoulder_offset_y', 'shoulder_offset_z'] + \
                                                                         ["ACT_" + i for i in param["physical_joints"]] + \
                                                                             ["D_ACT_" + i for i in param["physical_joints"]] 
    trajectory_table = pd.DataFrame(columns=trajectory_table_columns)

    model = mujoco_py.load_model_from_path(param["model_file"])

    shoulder_offset = np.array([model.body_pos[model.body_name2id(body)] for body in ["clavicle", "clavphant", "scapphant"]]).sum(axis=0)
    for step_index in range(len(X) - 1):
        trajectory_table.loc[len(trajectory_table), :] = np.concatenate(([time[step_index]],
                                                                         X[step_index].qpos[param["physical_joint_ids"]],
                                                                         X[step_index].qvel[param["physical_joint_ids"]],
                                                                         X_qacc[step_index][param["physical_joint_ids"]],
                                                                         X[step_index].qpos[param["virtual_joint_ids"]],
                                                                         X[step_index].qvel[param["virtual_joint_ids"]],
                                                                         X_qacc[step_index][param["virtual_joint_ids"]],
                                                                         XPOS[step_index], XVEL[step_index], param["target"], X[step_index].qpos[3:6],
                                                                         X[step_index].qpos[:3], U[step_index],
                                                                         TORQUE[step_index][param["physical_joint_ids"]],
                                                                         [step_index == 0], shoulder_offset,
                                                                         ACT[step_index],
                                                                         DACT[step_index]))

    trajectory_table.loc[len(trajectory_table), :] = np.concatenate(
        ([time[len(X) - 1]], X[len(X) - 1].qpos[param["physical_joint_ids"]], X[len(X) - 1].qvel[param["physical_joint_ids"]],np.zeros(len(param["physical_joint_ids"]), ),
        X[len(X) - 1].qpos[param["virtual_joint_ids"]], X[len(X) - 1].qvel[param["virtual_joint_ids"]],np.zeros(len(param["virtual_joint_ids"]), ),
         XPOS[len(X) - 1], XVEL[len(X) - 1], param["target"], X[len(X) - 1].qpos[3:6], X[len(X) - 1].qpos[:3], np.zeros(len(U[len(X) - 2]), ),
         np.zeros(len(param["physical_joint_ids"]), ),
         [1], shoulder_offset, ACT[len(X) - 1], DACT[len(X) - 1]))

    return trajectory_table

def check_simulation_dataset_dir(DIRNAME_SIMULATION):
    if not os.path.exists(DIRNAME_SIMULATION):
        download_datasets = input(
            "Could not find reference to the SIM-MPC Dataset. Do you want to download it (~4.5GB after unpacking)? (y/N) ")
        if download_datasets.lower().startswith("y"):
            print(f"Will download and unzip to '{os.path.abspath(DIRNAME_SIMULATION)}'.")
            print("Downloading archive... This can take several minutes. ", end='', flush=True)
            resp = urlopen("https://zenodo.org/record/7304381/files/SIM_MPC_Dataset.zip?download=1")
            zipfile = ZipFile(BytesIO(resp.read()))
            print("unzip archive... ", end='', flush=True)
            for file in zipfile.namelist():
                if file.startswith('simulation/'):
                    zipfile.extract(file, path=os.path.dirname(os.path.normpath(DIRNAME_SIMULATION)) if file.split("/")[0] == os.path.basename(os.path.normpath(DIRNAME_SIMULATION)) else DIRNAME_SIMULATION)
            print("done.")
            assert os.path.exists(DIRNAME_SIMULATION), "Internal Error during unpacking of SIM-MPC Dataset."
        else:
            raise FileNotFoundError("Please ensure that 'DIRNAME_SIMULATION' points to a valid directory containing the SIM-MPC Dataset.")


def check_study_dataset_dir(DIRNAME_STUDY):
    if not os.path.exists(DIRNAME_STUDY):
        download_datasets = input(
            "Could not find reference to the ISO-VR-Pointing Dataset. Do you want to download it (~3.9GB after unpacking)? (y/N) ")
        if download_datasets.lower().startswith("y"):
            print(f"Will download and unzip to '{os.path.abspath(DIRNAME_STUDY)}'.")
            print("Downloading archive... This can take several minutes. ", end='', flush=True)
            resp = urlopen("https://zenodo.org/record/7300062/files/ISO_VR_Pointing_Dataset.zip?download=1")
            zipfile = ZipFile(BytesIO(resp.read()))
            print("unzip archive... ", end='', flush=True)
            for file in zipfile.namelist():
                if file.startswith('study/'):
                    zipfile.extract(file, path=os.path.dirname(os.path.normpath(DIRNAME_STUDY)) if file.split("/")[0] == os.path.basename(os.path.normpath(DIRNAME_STUDY)) else DIRNAME_STUDY)
            print("done.")
            assert os.path.exists(DIRNAME_STUDY), "Internal Error during unpacking of ISO-VR-Pointing Dataset."
        else:
            raise FileNotFoundError("Please ensure that 'DIRNAME_STUDY' points to a valid directory containing the ISO-VR-Pointing Dataset.")
