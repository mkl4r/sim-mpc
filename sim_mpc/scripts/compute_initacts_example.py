''' 
Script to calculate an example initial activation.

Authors: Florian Fischer
Date: 11.2022
'''
import os
from pathlib import Path
import logging
import numpy as np
import argparse
from cfat.main import CFAT_initacts_only
from sim_mpc.core.utils import check_study_dataset_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run CFAT on initial frames of an Inverse Kinematics file from the ISO-VR-Pointing Dataset.')
    parser.add_argument('--dirname', dest='DIRNAME_STUDY', default='../data/study/',
                        help='Directory path of the ISO-VR-Pointing Dataset.')
    parser.add_argument('--username', dest='username', default='U1',
                        help='Username (U1-U6); used for model file (and table_filename, if not specified).')
    parser.add_argument('--task_condition', dest='task_condition', default='Virtual_Cursor_ID_ISO_15_plane',
                        help='Task condition; used for table_filename.')
    parser.add_argument('--table_filename', dest='table_filename', help='Filename to run CFAT with.')
    mujocopy_parser = parser.add_mutually_exclusive_group(required=False)
    mujocopy_parser.add_argument('--mujoco-py', dest='use_mujoco_py', action='store_true',
                                 help='Whether to use mujoco-py or MuJoCo Python bindings.')
    mujocopy_parser.add_argument('--mujoco', dest='use_mujoco_py', action='store_false')
    parser.set_defaults(use_mujoco_py=True)
    args = parser.parse_args()

    # Change current working directory to file directory
    os.chdir(Path(__file__).parent)
    logging.info(Path(__file__).parent)

    DIRNAME_STUDY = args.DIRNAME_STUDY
    DIRNAME_STUDY_IK = os.path.join(DIRNAME_STUDY, "IK_raw/")
    check_study_dataset_dir(DIRNAME_STUDY)

    username = args.username
    task_condition = args.task_condition

    if args.table_filename is not None:
        table_filename = args.table_filename
    else:
        table_filename = os.path.join(DIRNAME_STUDY_IK, f"{username}_{task_condition}.mot")

    model_filename = f"../data/models/OriginExperiment_{username}.xml"

    physical_joints = ["elv_angle", "shoulder_elv", "shoulder_rot", "elbow_flexion", "pro_sup", "deviation", "flexion"]
    param_t_activation = 0.04
    param_t_excitation = 0.03

    timestep = 0.002  # in seconds
    results_dir = f"../_results/{username}_{timestep}s_initacts/"

    ## Define submovements as first two frames of each user data submovement:
    # WARNING: Use _SubMovIndices.npy files instead of _SubMovTimes.npy (should be compensated in CFAT_algorithm()...)
    submovement_times = np.load(os.path.join(os.path.dirname(os.path.dirname(table_filename)), "_trialIndices",
                                             os.path.splitext(os.path.basename(table_filename))[
                                                 0] + '_SubMovIndices.npy'))
    submovement_times[:, 1] = submovement_times[:,
                              0] + 2  # replace end time of individual movements by two time steps (should be sufficient to compute excitation as well)


    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    CFAT_initacts_only(table_filename,
                       model_filename,
                       submovement_times,
                       physical_joints=physical_joints,
                       param_t_activation=param_t_activation,
                       param_t_excitation=param_t_excitation,
                       num_target_switches=None,
                       ensure_constraints=False,
                       reset_pos_and_vel=False,
                       useexcitationcontrol=True,
                       usecontrollimits=True,
                       optimize_excitations=False,
                       use_qacc=False,
                       timestep=timestep,
                       results_dir=results_dir,
                       use_mujoco_py=args.use_mujoco_py)