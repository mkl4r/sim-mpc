''' 
Script to calculate the initial activations locally.

Authors: Florian Fischer
Date: 11.2022
'''
import os
from pathlib import Path
import logging
import numpy as np
from cfat.main import CFAT_initacts_only
from sim_mpc.core.utils import check_study_dataset_dir

if __name__ == "__main__":

    # Change current working directory to file directory
    os.chdir(Path(__file__).parent)
    logging.info(Path(__file__).parent)

    DIRNAME_STUDY = "../data/study/"
    DIRNAME_STUDY_IK = "../data/study/IK_raw/"

    check_study_dataset_dir(DIRNAME_STUDY)

    print('\n\n         +++ Computation of Feasible Applied Torques (CFAT) -- Initial Actuations only +++')
    filelist = [(username, os.path.abspath(os.path.join(DIRNAME_STUDY_IK, f))) for username in
                [f"U{i}" for i in range(1, 7)]
                for f in os.listdir(DIRNAME_STUDY_IK) if
                os.path.isfile(os.path.abspath(os.path.join(DIRNAME_STUDY_IK, f))) and
                f.startswith(username) and f.endswith('.mot')]

    timestep = 0.002

    physical_joints = ["elv_angle", "shoulder_elv", "shoulder_rot", "elbow_flexion", "pro_sup", "deviation", "flexion"]
    param_t_activation = 0.04
    param_t_excitation = 0.03

    use_mujoco_py = True  # whether to use mujoco-py or mujoco

    for trial_id, (username, table_filename) in enumerate(filelist):
        if not any([taskcondition in table_filename for taskcondition in ("Virtual_Cursor_ID_ISO_15_plane",
                                                                          "Virtual_Cursor_Ergonomic_ISO_15_plane",
                                                                          "Virtual_Pad_ID_ISO_15_plane",
                                                                          "Virtual_Pad_Ergonomic_ISO_15_plane")]):
            continue

        results_dir = f"../_results/{username}_{timestep}s_initacts/"
        if not os.path.exists(os.path.expanduser(results_dir)):
            os.makedirs(os.path.expanduser(results_dir), exist_ok=True)
        print(f'\nCOMPUTING FEASIBLE CONTROLS for {table_filename} with constant controls for {timestep} seconds...')

        model_filename = f"../data/models/OriginExperiment_{username}.xml"

        ## Define submovements as first two frames of each user data submovement:
        # WARNING: Use _SubMovIndices.npy files instead of _SubMovTimes.npy (should be compensated in CFAT_algorithm()...)
        submovement_times = np.load(os.path.join(os.path.dirname(os.path.dirname(table_filename)), "_trialIndices",
                           os.path.splitext(os.path.basename(table_filename))[0] + '_SubMovIndices.npy'))
        submovement_times[:, 1] = submovement_times[:, 0] + 2  # replace end time of individual movements by two time steps (should be sufficient to compute excitation as well)


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
                       use_mujoco_py=use_mujoco_py)
