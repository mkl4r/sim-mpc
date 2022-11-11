import os, time
from pathlib import Path
import logging
import itertools
import numpy as np
import pandas as pd
from uitb_evaluate.trajectory_data import TrajectoryData_MPC, TrajectoryData_STUDY
from uitb_evaluate.trajectory_comparisons import QuantitativeComparison
from uitb_evaluate.evaluate_main import trajectoryplot
from uitb_evaluate.evaluate_comparisons import quantitativecomparisonplot
from uitb_evaluate.utils import preprocess_movement_data_simulation_complete
from sim_mpc.core.utils import check_simulation_dataset_dir, check_study_dataset_dir

# Change current working directory to file directory
os.chdir(Path(__file__).parent)
logging.info(Path(__file__).parent)

DIRNAME_SIMULATION = "../data/simulation/"
DIRNAME_STUDY = "../data/study/"
FILENAME_STUDY_TARGETPOSITIONS = "../data/targets/iso_targets_15_plane.csv"

if __name__ == "__main__":

    check_simulation_dataset_dir(DIRNAME_SIMULATION)
    check_study_dataset_dir(DIRNAME_STUDY)

    #################################

    _active_parts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

    #################################

    if 1 in _active_parts:
        # Figure 6/B.2
        _subtitle = """PART 1: MPC - COMPARISON BETWEEN 3 COST FUNCTIONS"""
        print(f"\n\n+++++++++++++++++++++++++++++++++++++\n{_subtitle}\n++++++++++++++++++++++++++++++++++++++\n")

        USER_ID = "U4"
        SIMULATION_SUBDIR_LIST = ["DC",
                                  "CTC",
                                  "JAC"]

        TASK_CONDITION = "Virtual_Pad_ID_ISO_15_plane"
        # ["Virtual_Cursor_ID_ISO_15_plane", "Virtual_Cursor_Ergonomic_ISO_15_plane", "Virtual_Pad_ID_ISO_15_plane", "Virtual_Pad_Ergonomic_ISO_15_plane"]

        filename = ""
        # AGGREGATE_TRIALS = False
        #########################

        REPEATED_MOVEMENTS = False

        trajectories_SIMULATION1 = TrajectoryData_MPC(DIRNAME_SIMULATION, SIMULATION_SUBDIR_LIST[0], USER_ID=USER_ID,
                                                      TASK_CONDITION=TASK_CONDITION,
                                                      FILENAME_STUDY_TARGETPOSITIONS=FILENAME_STUDY_TARGETPOSITIONS)
        trajectories_SIMULATION1.preprocess()  # AGGREGATE_TRIALS=AGGREGATE_TRIALS)

        trajectories_SIMULATION2 = TrajectoryData_MPC(DIRNAME_SIMULATION, SIMULATION_SUBDIR_LIST[1], USER_ID=USER_ID,
                                                      TASK_CONDITION=TASK_CONDITION,
                                                      FILENAME_STUDY_TARGETPOSITIONS=FILENAME_STUDY_TARGETPOSITIONS)
        trajectories_SIMULATION2.preprocess()  # AGGREGATE_TRIALS=AGGREGATE_TRIALS)

        trajectories_SIMULATION3 = TrajectoryData_MPC(DIRNAME_SIMULATION, SIMULATION_SUBDIR_LIST[2], USER_ID=USER_ID,
                                                      TASK_CONDITION=TASK_CONDITION,
                                                      FILENAME_STUDY_TARGETPOSITIONS=FILENAME_STUDY_TARGETPOSITIONS)
        trajectories_SIMULATION3.preprocess()  # AGGREGATE_TRIALS=AGGREGATE_TRIALS)

        trajectories_SIMULATION = [trajectories_SIMULATION1, trajectories_SIMULATION2, trajectories_SIMULATION3]

        # Preprocess experimental trajectories (ISO Task User Study):
        trajectories_STUDY = TrajectoryData_STUDY(DIRNAME_STUDY, USER_ID=USER_ID, TASK_CONDITION=TASK_CONDITION,
                                                  FILENAME_STUDY_TARGETPOSITIONS=FILENAME_STUDY_TARGETPOSITIONS)
        trajectories_STUDY.preprocess()  # AGGREGATE_TRIALS=AGGREGATE_TRIALS)

        #########################

        PLOTTING_ENV = "MPC-costs"

        common_simulation_subdir = "".join([i for i, j, k in zip(*SIMULATION_SUBDIR_LIST) if i == j == k])
        if len(common_simulation_subdir) == 0:
            common_simulation_subdir = "ALLCOSTS"
        elif common_simulation_subdir[-1] in ["-", "_"]:
            common_simulation_subdir = common_simulation_subdir[:-1]

        #################################

        #################################
        ### SET PLOTTING PARAM VALUES ###
        #################################

        # WHICH PARTS OF DATASET? #only used if PLOTTING_ENV == "RL-UIB"
        MOVEMENT_IDS = None  # range(1,9) #[i for i in range(10) if i != 1]
        RADIUS_IDS = None
        EPISODE_IDS = [7]

        r1_FIXED = None  # r1list[5]  #only used if PLOTTING_ENV == "MPC-costweights"
        r2_FIXED = None  # r2list[-1]  #only used if PLOTTING_ENV == "MPC-costweights"

        # TASK_CONDITION_LIST_SELECTED = ["Virtual_Cursor_ID_ISO_15_plane", "Virtual_Cursor_Ergonomic_ISO_15_plane"]  #only used if PLOTTING_ENV == "MPC-taskconditions"

        # WHAT TO COMPUTE?
        EFFECTIVE_PROJECTION_PATH = (
                    PLOTTING_ENV == "RL-UIB")  # if True, projection path connects effective initial and final position instead of nominal target center positions
        USE_TARGETBOUND_AS_DIST = False  # True/False or "MinJerk-only"; if True, only plot trajectory until targetboundary is reached first (i.e., until dwell time begins); if "MinJerk-only", complete simulation trajectories are shown, but MinJerk trajectories are aligned to simulation trajectories without dwell time
        MINJERK_USER_CONSTRAINTS = True

        # WHICH/HOW MANY MOVS?
        """
        IMPORTANT INFO:
        if isinstance(trajectories, TrajectoryData_RL):
            -> TRIAL_IDS/META_IDS/N_MOVS are (meta-)indices of respective episode (or rather, of respective row of trajectories.indices)
            -> TRIAL_IDS and META_IDS are equivalent
        if isinstance(trajectories, TrajectoryData_STUDY) or isinstance(trajectories, TrajectoryData_MPC):
            -> TRIAL_IDS/META_IDS/N_MOVS are (global) indices of entire dataset
            -> TRIAL_IDS correspond to trial indices assigned during user study (last column of trajectories.indices), while META_IDS correspond to (meta-)indices of trajectories.indices itself (i.e., if some trials were removed during previous pre-processing steps, TRIAL_IDS and META_IDS differ!)
        In general, only the first not-None parameter is used, in following order: TRIAL_IDS, META_IDS, N_MOVS.
        """
        TARGET_IDS = None  # corresponds to target IDs (if PLOTTING_ENV == "RL-UIB": only for iso-task)
        TRIAL_IDS = [
            7]  # [i for i in range(65) if i not in range(28, 33)] #range(1,4) #corresponds to trial IDs [last column of self.indices] or relative (meta) index per row in self.indices; either a list of indices, "different_target_sizes" (choose N_MOVS conditions with maximum differences in target size), or None (use META_IDS)
        META_IDS = None  # index positions (i.e., sequential numbering of trials [aggregated trials, if AGGREGATE_TRIALS==True] in indices, without counting removed outliers); if None: use N_MOVS
        N_MOVS = None  # number of movements to visualize (only used, if TRIAL_IDS and META_IDS are both None (or TRIAL_IDS == "different_target_sizes"))
        AGGREGATION_VARS = []  # ["all"]  #["episode", "movement"]  #["episode", "targetoccurrence"]  #["targetoccurrence"] #["episode", "movement"]  #["episode", "radius", "movement", "target", "targetoccurrence"]

        # WHAT TO PLOT?
        PLOT_TRACKING_DISTANCE = False  # if True, plot distance between End-effector and target position instead of position (only reasonable for tracking tasks)
        PLOT_ENDEFFECTOR = True  # if True plot End-effector position and velocity, else plot qpos and qvel for joint with index JOINT_ID (see independent_joints below)
        JOINT_ID = 2  # only used if PLOT_ENDEFFECTOR == False
        PLOT_DEVIATION = False  # only if PLOT_ENDEFFECTOR == True

        # HOW TO PLOT?
        NORMALIZE_TIME = False
        # DWELL_TIME = 0.3  #tail of the trajectories that is not shown (in seconds)
        PLOT_TIME_SERIES = True  # if True plot Position/Velocity/Acceleration Time Series, else plot Phasespace and Hooke plots
        PLOT_VEL_ACC = False  # if True, plot Velocity and Acceleration Time Series, else plot Position and Velocity Time Series (only used if PLOT_TIME_SERIES == True)
        PLOT_RANGES = False
        CONF_LEVEL = "min/max"  # might be between 0 and 1, or "min/max"; only used if PLOT_RANGES==True

        # WHICH BASELINE?
        SHOW_MINJERK = False
        SHOW_STUDY = True
        STUDY_ONLY = False  # only used if PLOTTING_ENV == "MPC-taskconditions"

        # PLOT (WHICH) LEGENDS AND COLORBARS?
        ENABLE_LEGENDS_AND_COLORBARS = True  # if False, legends (of axis 0) and colobars are removed
        ALLOW_DUPLICATES_BETWEEN_LEGENDS = False  # if False, legend of axis 1 only contains values not included in legend of axis 0

        # STORE PLOT?
        STORE_PLOT = True
        STORE_AXES_SEPARATELY = True  # if True, store left and right axis to separate figures

        ####

        trajectoryplot(PLOTTING_ENV, USER_ID, TASK_CONDITION,
                       common_simulation_subdir, filename, trajectories_SIMULATION,
                       trajectories_STUDY=trajectories_STUDY,
                       REPEATED_MOVEMENTS=REPEATED_MOVEMENTS,
                       MOVEMENT_IDS=MOVEMENT_IDS,
                       RADIUS_IDS=RADIUS_IDS,
                       EPISODE_IDS=EPISODE_IDS,
                       r1_FIXED=r1_FIXED,
                       r2_FIXED=r2_FIXED,
                       EFFECTIVE_PROJECTION_PATH=EFFECTIVE_PROJECTION_PATH,
                       USE_TARGETBOUND_AS_DIST=USE_TARGETBOUND_AS_DIST,
                       MINJERK_USER_CONSTRAINTS=MINJERK_USER_CONSTRAINTS,
                       TARGET_IDS=TARGET_IDS,
                       TRIAL_IDS=TRIAL_IDS,
                       META_IDS=META_IDS,
                       N_MOVS=N_MOVS,
                       AGGREGATION_VARS=AGGREGATION_VARS,
                       PLOT_TRACKING_DISTANCE=PLOT_TRACKING_DISTANCE,
                       PLOT_ENDEFFECTOR=PLOT_ENDEFFECTOR,
                       JOINT_ID=JOINT_ID,
                       PLOT_DEVIATION=PLOT_DEVIATION,
                       NORMALIZE_TIME=NORMALIZE_TIME,
                       # DWELL_TIME=#DWELL_TIME,
                       PLOT_TIME_SERIES=PLOT_TIME_SERIES,
                       PLOT_VEL_ACC=PLOT_VEL_ACC,
                       PLOT_RANGES=PLOT_RANGES,
                       CONF_LEVEL=CONF_LEVEL,
                       SHOW_MINJERK=SHOW_MINJERK,
                       SHOW_STUDY=SHOW_STUDY,
                       STUDY_ONLY=STUDY_ONLY,
                       ENABLE_LEGENDS_AND_COLORBARS=ENABLE_LEGENDS_AND_COLORBARS,
                       ALLOW_DUPLICATES_BETWEEN_LEGENDS=ALLOW_DUPLICATES_BETWEEN_LEGENDS,
                       STORE_PLOT=STORE_PLOT,
                       STORE_AXES_SEPARATELY=STORE_AXES_SEPARATELY)

        PLOT_ENDEFFECTOR = False
        for JOINT_ID in range(7):
            trajectoryplot(PLOTTING_ENV, USER_ID, TASK_CONDITION,
                           common_simulation_subdir, filename, trajectories_SIMULATION,
                           trajectories_STUDY=trajectories_STUDY,
                           REPEATED_MOVEMENTS=REPEATED_MOVEMENTS,
                           MOVEMENT_IDS=MOVEMENT_IDS,
                           RADIUS_IDS=RADIUS_IDS,
                           EPISODE_IDS=EPISODE_IDS,
                           r1_FIXED=r1_FIXED,
                           r2_FIXED=r2_FIXED,
                           EFFECTIVE_PROJECTION_PATH=EFFECTIVE_PROJECTION_PATH,
                           USE_TARGETBOUND_AS_DIST=USE_TARGETBOUND_AS_DIST,
                           MINJERK_USER_CONSTRAINTS=MINJERK_USER_CONSTRAINTS,
                           TARGET_IDS=TARGET_IDS,
                           TRIAL_IDS=TRIAL_IDS,
                           META_IDS=META_IDS,
                           N_MOVS=N_MOVS,
                           AGGREGATION_VARS=AGGREGATION_VARS,
                           PLOT_TRACKING_DISTANCE=PLOT_TRACKING_DISTANCE,
                           PLOT_ENDEFFECTOR=PLOT_ENDEFFECTOR,
                           JOINT_ID=JOINT_ID,
                           PLOT_DEVIATION=PLOT_DEVIATION,
                           NORMALIZE_TIME=NORMALIZE_TIME,
                           # DWELL_TIME=#DWELL_TIME,
                           PLOT_TIME_SERIES=PLOT_TIME_SERIES,
                           PLOT_VEL_ACC=PLOT_VEL_ACC,
                           PLOT_RANGES=PLOT_RANGES,
                           CONF_LEVEL=CONF_LEVEL,
                           SHOW_MINJERK=SHOW_MINJERK,
                           SHOW_STUDY=SHOW_STUDY,
                           STUDY_ONLY=STUDY_ONLY,
                           ENABLE_LEGENDS_AND_COLORBARS=JOINT_ID==0,
                           ALLOW_DUPLICATES_BETWEEN_LEGENDS=ALLOW_DUPLICATES_BETWEEN_LEGENDS,
                           STORE_PLOT=STORE_PLOT,
                           STORE_AXES_SEPARATELY=STORE_AXES_SEPARATELY)

    ########################################

    if 2 in _active_parts:
        # Figure 7/B.3
        _subtitle = """PART 2: MPC (QUANTITATIVE) - RMSE COMPUTATIONS"""
        print(f"\n\n+++++++++++++++++++++++++++++++++++++\n{_subtitle}\n++++++++++++++++++++++++++++++++++++++\n")

        USER_ID_LIST = [f"U{i}" for i in range(1, 7)]
        SIMULATION_SUBDIR_LIST = ["DC",
                                  "CTC",
                                  "JAC"]

        TASK_CONDITION_LIST = ["Virtual_Cursor_ID_ISO_15_plane",
                               "Virtual_Cursor_Ergonomic_ISO_15_plane", "Virtual_Pad_ID_ISO_15_plane",
                               "Virtual_Pad_Ergonomic_ISO_15_plane"]

        logging.getLogger().setLevel(logging.ERROR)

        start_time = time.time()

        res_dict_pos = {SIMULATION_SUBDIR: [] for SIMULATION_SUBDIR in SIMULATION_SUBDIR_LIST}
        res_dict_vel = {SIMULATION_SUBDIR: [] for SIMULATION_SUBDIR in SIMULATION_SUBDIR_LIST}
        res_dict_acc = {SIMULATION_SUBDIR: [] for SIMULATION_SUBDIR in SIMULATION_SUBDIR_LIST}
        res_dict_qpos = {SIMULATION_SUBDIR: [] for SIMULATION_SUBDIR in SIMULATION_SUBDIR_LIST}
        res_dict_qvel = {SIMULATION_SUBDIR: [] for SIMULATION_SUBDIR in SIMULATION_SUBDIR_LIST}
        res_dict_qacc = {SIMULATION_SUBDIR: [] for SIMULATION_SUBDIR in SIMULATION_SUBDIR_LIST}

        for USER_ID in USER_ID_LIST:
            for TASK_CONDITION in TASK_CONDITION_LIST:
                # Preprocess simulation trajectories (ISO Task User Study):
                trajectories_STUDY = TrajectoryData_STUDY(DIRNAME_STUDY, USER_ID=USER_ID, TASK_CONDITION=TASK_CONDITION,
                                                          FILENAME_STUDY_TARGETPOSITIONS=FILENAME_STUDY_TARGETPOSITIONS)
                trajectories_STUDY.preprocess()
                # Use all available trials
                trajectories_STUDY.compute_indices(TARGET_IDS=None, TRIAL_IDS=None, META_IDS=None, N_MOVS=None,
                                                   AGGREGATION_VARS=[], ignore_trainingset_trials=True)

                for SIMULATION_SUBDIR in SIMULATION_SUBDIR_LIST:
                    trajectories_SIMULATION = TrajectoryData_MPC(DIRNAME_SIMULATION, SIMULATION_SUBDIR, USER_ID=USER_ID,
                                                                 TASK_CONDITION=TASK_CONDITION,
                                                                 FILENAME_STUDY_TARGETPOSITIONS=FILENAME_STUDY_TARGETPOSITIONS)
                    trajectories_SIMULATION.preprocess()
                    # Use all available trials
                    trajectories_SIMULATION.compute_indices(TARGET_IDS=None, TRIAL_IDS=None, META_IDS=None, N_MOVS=None,
                                                            AGGREGATION_VARS=[], ignore_trainingset_trials=True)

                    sim_vs_study = QuantitativeComparison(trajectories_SIMULATION, trajectories_STUDY)
                    # sim_vs_study.compare("projected_trajectories_pos_trial")

                    try:
                        res_pos = sim_vs_study.compare_all_trials("position_series_trial", cols=None, metric="RMSE",
                                                                  mean_axis=None, ignore_unpaired_trials=True)
                        res_vel = sim_vs_study.compare_all_trials("velocity_series_trial", cols=None, metric="RMSE",
                                                                  mean_axis=None, ignore_unpaired_trials=True)
                        res_acc = sim_vs_study.compare_all_trials("acceleration_series_trial", cols=None, metric="RMSE",
                                                                  mean_axis=None, ignore_unpaired_trials=True)
                        res_qpos = sim_vs_study.compare_all_trials("qpos_series_trial", cols=None, metric="RMSE",
                                                                   mean_axis=None, ignore_unpaired_trials=True)
                        res_qvel = sim_vs_study.compare_all_trials("qvel_series_trial", cols=None, metric="RMSE",
                                                                   mean_axis=None, ignore_unpaired_trials=True)
                        res_qacc = sim_vs_study.compare_all_trials("qacc_series_trial", cols=None, metric="RMSE",
                                                                   mean_axis=None, ignore_unpaired_trials=True)

                        res_dict_pos[SIMULATION_SUBDIR].append(res_pos)
                        res_dict_vel[SIMULATION_SUBDIR].append(res_vel)
                        res_dict_acc[SIMULATION_SUBDIR].append(res_acc)
                        res_dict_qpos[SIMULATION_SUBDIR].append(res_qpos)
                        res_dict_qvel[SIMULATION_SUBDIR].append(res_qvel)
                        res_dict_qacc[SIMULATION_SUBDIR].append(res_qacc)
                    except AssertionError as e:
                        print((f"{USER_ID}, {TASK_CONDITION}, {SIMULATION_SUBDIR}: {e}"))
                        print("Will ignore this sub-dataset and continue...")

        end_time = time.time()
        logging.getLogger().setLevel(logging.INFO)

        logging.info(f"RMSE computation took {end_time - start_time} seconds.")

        #########################

        PLOTTING_ENV_COMPARISON = "MPC"

        #########################

        # STORE PLOT?
        STORE_PLOT = True

        USER2USER_FIXED = False  # if True, compare predictability of user movements of USER_ID_FIXED between simulation and other users
        USER2USER = False  # if True, compare predictability of user movements (of all users) between simulation and respective other users
        #SIM2USER_PER_USER = False  # if True, compare cost function simulation for each user separately

        quantity_list = ["pos", "vel", "acc", "qpos", "qvel", "qacc"]
        for QUANTITY in quantity_list:
            if USER2USER_FIXED:
                res_dict = locals()[f"res_dict_predict_fixeduser_{QUANTITY}"]
            elif USER2USER:
                res_dict = locals()[f"res_dict_predict_{QUANTITY}"]
                # res_dict = {k: v[1::6] for k,v in locals()[f"res_dict_predict_{QUANTITY}"].items()}
            else:
                res_dict = locals()[f"res_dict_{QUANTITY}"]

            quantitativecomparisonplot(PLOTTING_ENV_COMPARISON, QUANTITY, res_dict,
                                       SIMULATION_SUBDIR_LIST,
                                       TASK_CONDITION_LIST,
                                       USER2USER_FIXED=USER2USER_FIXED,
                                       USER2USER=USER2USER,
                                       SIM2USER_PER_USER=False,
                                       STORE_PLOT=STORE_PLOT)

            # Additionally, create "simonly_peruser" plots:
            quantitativecomparisonplot(PLOTTING_ENV_COMPARISON, QUANTITY, res_dict,
                                       SIMULATION_SUBDIR_LIST,
                                       TASK_CONDITION_LIST,
                                       USER2USER_FIXED=USER2USER_FIXED,
                                       USER2USER=USER2USER,
                                       SIM2USER_PER_USER=True,
                                       ENABLE_LEGEND=QUANTITY=="pos",
                                       STORE_PLOT=STORE_PLOT)

    ########################################

    if 3 in _active_parts:
        # Figure 8/B.4
        _subtitle = """PART 3: MPC - Simulation vs. User comparisons of single cost function (baseline only, i.e., other users should be ignored in plots)"""
        print(f"\n\n+++++++++++++++++++++++++++++++++++++\n{_subtitle}\n++++++++++++++++++++++++++++++++++++++\n")

        USER_ID_FIXED = "U2"  # (only used if SHOW_STUDY == True)

        SIMULATION_SUBDIR = "JAC"

        TASK_CONDITION_LIST = ["Virtual_Cursor_ID_ISO_15_plane", "Virtual_Cursor_Ergonomic_ISO_15_plane"]

        filename = ""

        # AGGREGATE_TRIALS = True
        #########################

        REPEATED_MOVEMENTS = False

        #########################

        PLOTTING_ENV = "MPC-userstudy-baselineonly"

        common_simulation_subdir = SIMULATION_SUBDIR

        #################################

        #################################
        ### SET PLOTTING PARAM VALUES ###
        #################################

        # WHICH PARTS OF DATASET? #only used if PLOTTING_ENV == "RL-UIB"
        MOVEMENT_IDS = None  # range(1,9) #[i for i in range(10) if i != 1]
        RADIUS_IDS = None
        EPISODE_IDS = [7]

        r1_FIXED = None  # r1list[5]  #only used if PLOTTING_ENV == "MPC-costweights"
        r2_FIXED = None  # r2list[-1]  #only used if PLOTTING_ENV == "MPC-costweights"

        # TASK_CONDITION_LIST_SELECTED = ["Virtual_Cursor_ID_ISO_15_plane", "Virtual_Cursor_Ergonomic_ISO_15_plane"]  #only used if PLOTTING_ENV == "MPC-taskconditions"

        # WHAT TO COMPUTE?
        EFFECTIVE_PROJECTION_PATH = (
                    PLOTTING_ENV == "RL-UIB")  # if True, projection path connects effective initial and final position instead of nominal target center positions
        USE_TARGETBOUND_AS_DIST = False  # True/False or "MinJerk-only"; if True, only plot trajectory until targetboundary is reached first (i.e., until dwell time begins); if "MinJerk-only", complete simulation trajectories are shown, but MinJerk trajectories are aligned to simulation trajectories without dwell time
        MINJERK_USER_CONSTRAINTS = True

        # WHICH/HOW MANY MOVS?
        """
        IMPORTANT INFO:
        if isinstance(trajectories, TrajectoryData_RL):
            -> TRIAL_IDS/META_IDS/N_MOVS are (meta-)indices of respective episode (or rather, of respective row of trajectories.indices)
            -> TRIAL_IDS and META_IDS are equivalent
        if isinstance(trajectories, TrajectoryData_STUDY) or isinstance(trajectories, TrajectoryData_MPC):
            -> TRIAL_IDS/META_IDS/N_MOVS are (global) indices of entire dataset
            -> TRIAL_IDS correspond to trial indices assigned during user study (last column of trajectories.indices), while META_IDS correspond to (meta-)indices of trajectories.indices itself (i.e., if some trials were removed during previous pre-processing steps, TRIAL_IDS and META_IDS differ!)
        In general, only the first not-None parameter is used, in following order: TRIAL_IDS, META_IDS, N_MOVS.
        """
        TARGET_IDS = None  # corresponds to target IDs (if PLOTTING_ENV == "RL-UIB": only for iso-task)
        TRIAL_IDS = None  # [i for i in range(65) if i not in range(28, 33)] #range(1,4) #corresponds to trial IDs [last column of self.indices] or relative (meta) index per row in self.indices; either a list of indices, "different_target_sizes" (choose N_MOVS conditions with maximum differences in target size), or None (use META_IDS)
        META_IDS = None  # index positions (i.e., sequential numbering of trials [aggregated trials, if AGGREGATE_TRIALS==True] in indices, without counting removed outliers); if None: use N_MOVS
        N_MOVS = None  # number of movements to visualize (only used, if TRIAL_IDS and META_IDS are both None (or TRIAL_IDS == "different_target_sizes"))
        AGGREGATION_VARS = [
            "all"]  # ["episode", "movement"]  #["episode", "targetoccurrence"]  #["targetoccurrence"] #["episode", "movement"]  #["episode", "radius", "movement", "target", "targetoccurrence"]

        # WHAT TO PLOT?
        PLOT_TRACKING_DISTANCE = False  # if True, plot distance between End-effector and target position instead of position (only reasonable for tracking tasks)
        PLOT_ENDEFFECTOR = False  # if True plot End-effector position and velocity, else plot qpos and qvel for joint with index JOINT_ID (see independent_joints below)
        # JOINT_ID = 1  #only used if PLOT_ENDEFFECTOR == False
        PLOT_DEVIATION = False  # only if PLOT_ENDEFFECTOR == True

        # HOW TO PLOT?
        NORMALIZE_TIME = False
        PLOT_TIME_SERIES = True  # if True plot Position/Velocity/Acceleration Time Series, else plot Phasespace and Hooke plots
        PLOT_VEL_ACC = False  # if True, plot Velocity and Acceleration Time Series, else plot Position and Velocity Time Series (only used if PLOT_TIME_SERIES == True)
        PLOT_RANGES = True
        CONF_LEVEL = "min/max"  # might be between 0 and 1, or "min/max"; only used if PLOT_RANGES==True

        # WHICH BASELINE?
        SHOW_MINJERK = False
        SHOW_STUDY = True
        STUDY_ONLY = False  # only used if PLOTTING_ENV == "MPC-taskconditions"

        # PLOT (WHICH) LEGENDS AND COLORBARS?
        # ENABLE_LEGENDS_AND_COLORBARS = True  #if False, legends (of axis 0) and colobars are removed
        ALLOW_DUPLICATES_BETWEEN_LEGENDS = False  # if False, legend of axis 1 only contains values not included in legend of axis 0

        # STORE PLOT?
        STORE_PLOT = True
        STORE_AXES_SEPARATELY = True  # if True, store left and right axis to separate figures

        ####

        for TASK_CONDITION in TASK_CONDITION_LIST:
            trajectories_SIMULATION = TrajectoryData_MPC(DIRNAME_SIMULATION, SIMULATION_SUBDIR, USER_ID=USER_ID_FIXED,
                                                         TASK_CONDITION=TASK_CONDITION,
                                                         FILENAME_STUDY_TARGETPOSITIONS=FILENAME_STUDY_TARGETPOSITIONS)
            trajectories_SIMULATION.preprocess()  # AGGREGATE_TRIALS=AGGREGATE_TRIALS)

            trajectories_STUDY = TrajectoryData_STUDY(DIRNAME_STUDY, USER_ID=USER_ID_FIXED,
                                                      TASK_CONDITION=TASK_CONDITION,
                                                      FILENAME_STUDY_TARGETPOSITIONS=FILENAME_STUDY_TARGETPOSITIONS)
            trajectories_STUDY.preprocess()

            for JOINT_ID in range(7):
                trajectoryplot(PLOTTING_ENV, USER_ID_FIXED, TASK_CONDITION,
                               common_simulation_subdir, filename, trajectories_SIMULATION,
                               trajectories_STUDY=trajectories_STUDY,
                               REPEATED_MOVEMENTS=REPEATED_MOVEMENTS,
                               USER_ID_FIXED=USER_ID_FIXED,
                               MOVEMENT_IDS=MOVEMENT_IDS,
                               RADIUS_IDS=RADIUS_IDS,
                               EPISODE_IDS=EPISODE_IDS,
                               r1_FIXED=r1_FIXED,
                               r2_FIXED=r2_FIXED,
                               EFFECTIVE_PROJECTION_PATH=EFFECTIVE_PROJECTION_PATH,
                               USE_TARGETBOUND_AS_DIST=USE_TARGETBOUND_AS_DIST,
                               MINJERK_USER_CONSTRAINTS=MINJERK_USER_CONSTRAINTS,
                               TARGET_IDS=TARGET_IDS,
                               TRIAL_IDS=TRIAL_IDS,
                               META_IDS=META_IDS,
                               N_MOVS=N_MOVS,
                               AGGREGATION_VARS=AGGREGATION_VARS,
                               PLOT_TRACKING_DISTANCE=PLOT_TRACKING_DISTANCE,
                               PLOT_ENDEFFECTOR=PLOT_ENDEFFECTOR,
                               JOINT_ID=JOINT_ID,
                               PLOT_DEVIATION=PLOT_DEVIATION,
                               NORMALIZE_TIME=NORMALIZE_TIME,
                               # DWELL_TIME=#DWELL_TIME,
                               PLOT_TIME_SERIES=PLOT_TIME_SERIES,
                               PLOT_VEL_ACC=PLOT_VEL_ACC,
                               PLOT_RANGES=PLOT_RANGES,
                               CONF_LEVEL=CONF_LEVEL,
                               SHOW_MINJERK=SHOW_MINJERK,
                               SHOW_STUDY=SHOW_STUDY,
                               STUDY_ONLY=STUDY_ONLY,
                               ENABLE_LEGENDS_AND_COLORBARS=(
                                           JOINT_ID in (0, 2) and TASK_CONDITION == "Virtual_Cursor_ID_ISO_15_plane"),
                               ALLOW_DUPLICATES_BETWEEN_LEGENDS=ALLOW_DUPLICATES_BETWEEN_LEGENDS,
                               predefined_ylim=(-0.1, 1.2) if (JOINT_ID == 1) else None,
                               STORE_PLOT=STORE_PLOT,
                               STORE_AXES_SEPARATELY=STORE_AXES_SEPARATELY)

    ########################################

    if 4 in _active_parts:
        # Figure 9
        _subtitle = """PART 4: MPC - RMSE COMPUTATIONS (SIM2FIXEDUSER vs. USER2FIXEDUSER vs. USER2FIXEDUSER vs. ... [for all other users])"""
        print(f"\n\n+++++++++++++++++++++++++++++++++++++\n{_subtitle}\n++++++++++++++++++++++++++++++++++++++\n")

        USER_ID_LIST = [f"U{i}" for i in range(1, 7)]
        SIMULATION_SUBDIR_LIST = ["JAC"]

        TASK_CONDITION_LIST = ["Virtual_Cursor_ID_ISO_15_plane",
                               "Virtual_Cursor_Ergonomic_ISO_15_plane",
                               "Virtual_Pad_ID_ISO_15_plane",
                               "Virtual_Pad_Ergonomic_ISO_15_plane"
                               ]

        USER_ID_FIXED = "U1"

        logging.getLogger().setLevel(logging.ERROR)

        start_time = time.time()

        res_dict_predict_fixeduser_pos = {SIMULATION_SUBDIR: [] for SIMULATION_SUBDIR in
                                          SIMULATION_SUBDIR_LIST + [i for i in USER_ID_LIST if i != USER_ID_FIXED]}
        res_dict_predict_fixeduser_vel = {SIMULATION_SUBDIR: [] for SIMULATION_SUBDIR in
                                          SIMULATION_SUBDIR_LIST + [i for i in USER_ID_LIST if i != USER_ID_FIXED]}
        res_dict_predict_fixeduser_acc = {SIMULATION_SUBDIR: [] for SIMULATION_SUBDIR in
                                          SIMULATION_SUBDIR_LIST + [i for i in USER_ID_LIST if i != USER_ID_FIXED]}
        res_dict_predict_fixeduser_qpos = {SIMULATION_SUBDIR: [] for SIMULATION_SUBDIR in
                                           SIMULATION_SUBDIR_LIST + [i for i in USER_ID_LIST if i != USER_ID_FIXED]}
        res_dict_predict_fixeduser_qvel = {SIMULATION_SUBDIR: [] for SIMULATION_SUBDIR in
                                           SIMULATION_SUBDIR_LIST + [i for i in USER_ID_LIST if i != USER_ID_FIXED]}
        res_dict_predict_fixeduser_qacc = {SIMULATION_SUBDIR: [] for SIMULATION_SUBDIR in
                                           SIMULATION_SUBDIR_LIST + [i for i in USER_ID_LIST if i != USER_ID_FIXED]}

        for TASK_CONDITION in TASK_CONDITION_LIST:
            trajectories_STUDIES = {}
            for USER_ID in USER_ID_LIST:
                # Preprocess simulation trajectories (ISO Task User Study):
                trajectories_STUDIES[USER_ID] = TrajectoryData_STUDY(DIRNAME_STUDY, USER_ID=USER_ID,
                                                                     TASK_CONDITION=TASK_CONDITION,
                                                                     FILENAME_STUDY_TARGETPOSITIONS=FILENAME_STUDY_TARGETPOSITIONS)
                trajectories_STUDIES[USER_ID].preprocess()
                # Use all available trials
                trajectories_STUDIES[USER_ID].compute_indices(TARGET_IDS=None, TRIAL_IDS=None, META_IDS=None,
                                                              N_MOVS=None,
                                                              AGGREGATION_VARS=[], ignore_trainingset_trials=True)

            # Only use trials that are valid for all users
            unique, counts = np.unique(
                np.concatenate([trajectories_STUDIES[USER_ID].indices[:, 4] for USER_ID in USER_ID_LIST]),
                return_counts=True)
            trial_idx_list = unique[counts == len(USER_ID_LIST)]

            for SIMULATION_SUBDIR in SIMULATION_SUBDIR_LIST:
                trajectories_SIMULATION = TrajectoryData_MPC(DIRNAME_SIMULATION, SIMULATION_SUBDIR,
                                                             USER_ID=USER_ID_FIXED,
                                                             TASK_CONDITION=TASK_CONDITION,
                                                             FILENAME_STUDY_TARGETPOSITIONS=FILENAME_STUDY_TARGETPOSITIONS)
                trajectories_SIMULATION.preprocess()
                # Use trials that are valid for all users
                trajectories_SIMULATION.compute_indices(TARGET_IDS=None, TRIAL_IDS=trial_idx_list, META_IDS=None,
                                                        N_MOVS=None,
                                                        AGGREGATION_VARS=[], ignore_trainingset_trials=True)
                trajectories_STUDIES[USER_ID_FIXED].compute_indices(TRIAL_IDS=trial_idx_list, META_IDS=None,
                                                                    N_MOVS=None,
                                                                    AGGREGATION_VARS=[], ignore_trainingset_trials=True)

                sim_vs_study = QuantitativeComparison(trajectories_SIMULATION, trajectories_STUDIES[USER_ID_FIXED])
                try:
                    res_pos = sim_vs_study.compare_all_trials("position_series_trial", cols=None, metric="RMSE",
                                                              mean_axis=None,
                                                              ignore_unpaired_trials=False)
                    res_vel = sim_vs_study.compare_all_trials("velocity_series_trial", cols=None, metric="RMSE",
                                                              mean_axis=None,
                                                              ignore_unpaired_trials=False)
                    res_acc = sim_vs_study.compare_all_trials("acceleration_series_trial", cols=None, metric="RMSE",
                                                              mean_axis=None, ignore_unpaired_trials=False)
                    res_qpos = sim_vs_study.compare_all_trials("qpos_series_trial", cols=None, metric="RMSE",
                                                               mean_axis=None,
                                                               ignore_unpaired_trials=False)
                    res_qvel = sim_vs_study.compare_all_trials("qvel_series_trial", cols=None, metric="RMSE",
                                                               mean_axis=None,
                                                               ignore_unpaired_trials=False)
                    res_qacc = sim_vs_study.compare_all_trials("qacc_series_trial", cols=None, metric="RMSE",
                                                               mean_axis=None,
                                                               ignore_unpaired_trials=False)
                except ValueError as e:
                    print((f"{USER_ID}, {TASK_CONDITION}, {SIMULATION_SUBDIR}: {e}"))
                    print("Will ignore this sub-dataset and continue...")

                res_dict_predict_fixeduser_pos[SIMULATION_SUBDIR].append(res_pos)
                res_dict_predict_fixeduser_vel[SIMULATION_SUBDIR].append(res_vel)
                res_dict_predict_fixeduser_acc[SIMULATION_SUBDIR].append(res_acc)
                res_dict_predict_fixeduser_qpos[SIMULATION_SUBDIR].append(res_qpos)
                res_dict_predict_fixeduser_qvel[SIMULATION_SUBDIR].append(res_qvel)
                res_dict_predict_fixeduser_qacc[SIMULATION_SUBDIR].append(res_qacc)

            for USER_ID in [i for i in USER_ID_LIST if i != USER_ID_FIXED]:
                trajectories_STUDIES[USER_ID].compute_indices(TARGET_IDS=None, TRIAL_IDS=trial_idx_list, META_IDS=None,
                                                              N_MOVS=None, AGGREGATION_VARS=[],
                                                              ignore_trainingset_trials=True)
                sim_vs_study = QuantitativeComparison(trajectories_STUDIES[USER_ID],
                                                      trajectories_STUDIES[USER_ID_FIXED])
                try:
                    res_pos = sim_vs_study.compare_all_trials("position_series_trial", cols=None, metric="RMSE",
                                                              mean_axis=None,
                                                              ignore_unpaired_trials=False)
                    res_vel = sim_vs_study.compare_all_trials("velocity_series_trial", cols=None, metric="RMSE",
                                                              mean_axis=None,
                                                              ignore_unpaired_trials=False)
                    res_acc = sim_vs_study.compare_all_trials("acceleration_series_trial", cols=None, metric="RMSE",
                                                              mean_axis=None, ignore_unpaired_trials=False)
                    res_qpos = sim_vs_study.compare_all_trials("qpos_series_trial", cols=None, metric="RMSE",
                                                               mean_axis=None,
                                                               ignore_unpaired_trials=False)
                    res_qvel = sim_vs_study.compare_all_trials("qvel_series_trial", cols=None, metric="RMSE",
                                                               mean_axis=None,
                                                               ignore_unpaired_trials=False)
                    res_qacc = sim_vs_study.compare_all_trials("qacc_series_trial", cols=None, metric="RMSE",
                                                               mean_axis=None,
                                                               ignore_unpaired_trials=False)

                    res_dict_predict_fixeduser_pos[USER_ID].append(res_pos)
                    res_dict_predict_fixeduser_vel[USER_ID].append(res_vel)
                    res_dict_predict_fixeduser_acc[USER_ID].append(res_acc)
                    res_dict_predict_fixeduser_qpos[USER_ID].append(res_qpos)
                    res_dict_predict_fixeduser_qvel[USER_ID].append(res_qvel)
                    res_dict_predict_fixeduser_qacc[USER_ID].append(res_qacc)
                except ValueError as e:
                    print((f"{USER_ID}, {TASK_CONDITION}, {SIMULATION_SUBDIR}: {e}"))
                    print("Will ignore this sub-dataset and continue...")

        end_time = time.time()
        logging.getLogger().setLevel(logging.INFO)

        logging.info(f"RMSE computation took {end_time - start_time} seconds.")

        #########################

        PLOTTING_ENV_COMPARISON = "MPC"

        #########################

        # STORE PLOT?
        STORE_PLOT = True

        USER2USER_FIXED = True  # if True, compare predictability of user movements of USER_ID_FIXED between simulation and other users
        USER2USER = False  # if True, compare predictability of user movements (of all users) between simulation and respective other users
        SIM2USER_PER_USER = False  # if True, compare cost function simulation for each user separately

        quantity_list = ["pos", "vel", "acc", "qpos", "qvel", "qacc"]
        for QUANTITY in quantity_list:
            if USER2USER_FIXED:
                res_dict = locals()[f"res_dict_predict_fixeduser_{QUANTITY}"]
            elif USER2USER:
                res_dict = locals()[f"res_dict_predict_{QUANTITY}"]
                # res_dict = {k: v[1::6] for k,v in locals()[f"res_dict_predict_{QUANTITY}"].items()}
            else:
                res_dict = locals()[f"res_dict_{QUANTITY}"]

            quantitativecomparisonplot(PLOTTING_ENV_COMPARISON, QUANTITY, res_dict,
                                       SIMULATION_SUBDIR_LIST,
                                       TASK_CONDITION_LIST,
                                       USER2USER_FIXED=USER2USER_FIXED,
                                       USER2USER=USER2USER,
                                       SIM2USER_PER_USER=SIM2USER_PER_USER,
                                       USER_ID_FIXED=USER_ID_FIXED,
                                       STORE_PLOT=STORE_PLOT)

    ########################################

    if 5 in _active_parts:
        # Figure 10
        _subtitle = """PART 5: MPC - RMSE COMPUTATIONS (SIM2USER vs. USER2USER)"""
        print(f"\n\n+++++++++++++++++++++++++++++++++++++\n{_subtitle}\n++++++++++++++++++++++++++++++++++++++\n")

        USER_ID_LIST = [f"U{i}" for i in range(1, 7)]
        SIMULATION_SUBDIR_LIST = ["JAC"]

        TASK_CONDITION_LIST = ["Virtual_Cursor_ID_ISO_15_plane",
                               "Virtual_Cursor_Ergonomic_ISO_15_plane",
                               "Virtual_Pad_ID_ISO_15_plane",
                               "Virtual_Pad_Ergonomic_ISO_15_plane"
                               ]

        logging.getLogger().setLevel(logging.ERROR)

        start_time = time.time()

        res_dict_predict_pos = {SIMULATION_SUBDIR: [] for SIMULATION_SUBDIR in
                                SIMULATION_SUBDIR_LIST + ["User vs. User"]}
        res_dict_predict_vel = {SIMULATION_SUBDIR: [] for SIMULATION_SUBDIR in
                                SIMULATION_SUBDIR_LIST + ["User vs. User"]}
        res_dict_predict_acc = {SIMULATION_SUBDIR: [] for SIMULATION_SUBDIR in
                                SIMULATION_SUBDIR_LIST + ["User vs. User"]}
        res_dict_predict_qpos = {SIMULATION_SUBDIR: [] for SIMULATION_SUBDIR in
                                 SIMULATION_SUBDIR_LIST + ["User vs. User"]}
        res_dict_predict_qvel = {SIMULATION_SUBDIR: [] for SIMULATION_SUBDIR in
                                 SIMULATION_SUBDIR_LIST + ["User vs. User"]}
        res_dict_predict_qacc = {SIMULATION_SUBDIR: [] for SIMULATION_SUBDIR in
                                 SIMULATION_SUBDIR_LIST + ["User vs. User"]}

        for TASK_CONDITION in TASK_CONDITION_LIST:
            trajectories_STUDIES = {}
            for USER_ID in USER_ID_LIST:
                # Preprocess simulation trajectories (ISO Task User Study):
                trajectories_STUDIES[USER_ID] = TrajectoryData_STUDY(DIRNAME_STUDY, USER_ID=USER_ID,
                                                                     TASK_CONDITION=TASK_CONDITION,
                                                                     FILENAME_STUDY_TARGETPOSITIONS=FILENAME_STUDY_TARGETPOSITIONS)
                trajectories_STUDIES[USER_ID].preprocess()
                # Use all available trials
                trajectories_STUDIES[USER_ID].compute_indices(TARGET_IDS=None, TRIAL_IDS=None, META_IDS=None,
                                                              N_MOVS=None,
                                                              AGGREGATION_VARS=[], ignore_trainingset_trials=True)

            # Only use trials that are valid for all users
            unique, counts = np.unique(
                np.concatenate([trajectories_STUDIES[USER_ID].indices[:, 4] for USER_ID in USER_ID_LIST]),
                return_counts=True)
            trial_idx_list = unique[counts == len(USER_ID_LIST)]

            for USER_ID in USER_ID_LIST:
                for SIMULATION_SUBDIR in SIMULATION_SUBDIR_LIST:
                    trajectories_SIMULATION = TrajectoryData_MPC(DIRNAME_SIMULATION, SIMULATION_SUBDIR, USER_ID=USER_ID,
                                                                 TASK_CONDITION=TASK_CONDITION,
                                                                 FILENAME_STUDY_TARGETPOSITIONS=FILENAME_STUDY_TARGETPOSITIONS)
                    trajectories_SIMULATION.preprocess()
                    # Use trials that are valid for all users
                    trajectories_SIMULATION.compute_indices(TARGET_IDS=None, TRIAL_IDS=trial_idx_list, META_IDS=None,
                                                            N_MOVS=None, AGGREGATION_VARS=[],
                                                            ignore_trainingset_trials=True)
                    trajectories_STUDIES[USER_ID].compute_indices(TARGET_IDS=None, TRIAL_IDS=trial_idx_list,
                                                                  META_IDS=None,
                                                                  N_MOVS=None, AGGREGATION_VARS=[],
                                                                  ignore_trainingset_trials=True)

                    sim_vs_study = QuantitativeComparison(trajectories_SIMULATION, trajectories_STUDIES[USER_ID])
                    try:
                        res_pos = sim_vs_study.compare_all_trials("position_series_trial", cols=None, metric="RMSE",
                                                                  mean_axis=None, ignore_unpaired_trials=False)
                        res_vel = sim_vs_study.compare_all_trials("velocity_series_trial", cols=None, metric="RMSE",
                                                                  mean_axis=None, ignore_unpaired_trials=False)
                        res_acc = sim_vs_study.compare_all_trials("acceleration_series_trial", cols=None, metric="RMSE",
                                                                  mean_axis=None, ignore_unpaired_trials=False)
                        res_qpos = sim_vs_study.compare_all_trials("qpos_series_trial", cols=None, metric="RMSE",
                                                                   mean_axis=None, ignore_unpaired_trials=False)
                        res_qvel = sim_vs_study.compare_all_trials("qvel_series_trial", cols=None, metric="RMSE",
                                                                   mean_axis=None, ignore_unpaired_trials=False)
                        res_qacc = sim_vs_study.compare_all_trials("qacc_series_trial", cols=None, metric="RMSE",
                                                                   mean_axis=None, ignore_unpaired_trials=False)
                    except AssertionError as e:
                        print((f"{USER_ID}, {TASK_CONDITION}, {SIMULATION_SUBDIR}: {e}"))
                        print("Will ignore this sub-dataset and continue...")

                    res_dict_predict_pos[SIMULATION_SUBDIR].append(res_pos)
                    res_dict_predict_vel[SIMULATION_SUBDIR].append(res_vel)
                    res_dict_predict_acc[SIMULATION_SUBDIR].append(res_acc)
                    res_dict_predict_qpos[SIMULATION_SUBDIR].append(res_qpos)
                    res_dict_predict_qvel[SIMULATION_SUBDIR].append(res_qvel)
                    res_dict_predict_qacc[SIMULATION_SUBDIR].append(res_qacc)

            for USER_ID_1, USER_ID_2 in itertools.combinations(USER_ID_LIST, 2):
                sim_vs_study = QuantitativeComparison(trajectories_STUDIES[USER_ID_1], trajectories_STUDIES[USER_ID_2])
                try:
                    res_pos = sim_vs_study.compare_all_trials("position_series_trial", cols=None, metric="RMSE",
                                                              mean_axis=None, ignore_unpaired_trials=False)
                    res_vel = sim_vs_study.compare_all_trials("velocity_series_trial", cols=None, metric="RMSE",
                                                              mean_axis=None, ignore_unpaired_trials=False)
                    res_acc = sim_vs_study.compare_all_trials("acceleration_series_trial", cols=None, metric="RMSE",
                                                              mean_axis=None, ignore_unpaired_trials=False)
                    res_qpos = sim_vs_study.compare_all_trials("qpos_series_trial", cols=None, metric="RMSE",
                                                               mean_axis=None, ignore_unpaired_trials=False)
                    res_qvel = sim_vs_study.compare_all_trials("qvel_series_trial", cols=None, metric="RMSE",
                                                               mean_axis=None, ignore_unpaired_trials=False)
                    res_qacc = sim_vs_study.compare_all_trials("qacc_series_trial", cols=None, metric="RMSE",
                                                               mean_axis=None, ignore_unpaired_trials=False)

                    res_dict_predict_pos["User vs. User"].append(res_pos)
                    res_dict_predict_vel["User vs. User"].append(res_vel)
                    res_dict_predict_acc["User vs. User"].append(res_acc)
                    res_dict_predict_qpos["User vs. User"].append(res_qpos)
                    res_dict_predict_qvel["User vs. User"].append(res_qvel)
                    res_dict_predict_qacc["User vs. User"].append(res_qacc)
                except AssertionError as e:
                    print((f"{USER_ID}, {TASK_CONDITION}, {SIMULATION_SUBDIR}: {e}"))
                    print("Will ignore this sub-dataset and continue...")

        end_time = time.time()
        logging.getLogger().setLevel(logging.INFO)

        logging.info(f"RMSE computation took {end_time - start_time} seconds.")

        #########################

        PLOTTING_ENV_COMPARISON = "MPC"

        #########################

        # STORE PLOT?
        STORE_PLOT = True

        USER2USER_FIXED = False  # if True, compare predictability of user movements of USER_ID_FIXED between simulation and other users
        USER2USER = True  # if True, compare predictability of user movements (of all users) between simulation and respective other users
        SIM2USER_PER_USER = False  # if True, compare cost function simulation for each user separately

        quantity_list = ["pos", "vel", "acc", "qpos", "qvel", "qacc"]
        for QUANTITY in quantity_list:
            if USER2USER_FIXED:
                res_dict = locals()[f"res_dict_predict_fixeduser_{QUANTITY}"]
            elif USER2USER:
                res_dict = locals()[f"res_dict_predict_{QUANTITY}"]
                # res_dict = {k: v[1::6] for k,v in locals()[f"res_dict_predict_{QUANTITY}"].items()}
            else:
                res_dict = locals()[f"res_dict_{QUANTITY}"]

            quantitativecomparisonplot(PLOTTING_ENV_COMPARISON, QUANTITY, res_dict,
                                       SIMULATION_SUBDIR_LIST,
                                       TASK_CONDITION_LIST,
                                       USER2USER_FIXED=USER2USER_FIXED,
                                       USER2USER=USER2USER,
                                       SIM2USER_PER_USER=SIM2USER_PER_USER,
                                       STORE_PLOT=STORE_PLOT)

    ########################################

    if 6 in _active_parts:
        # Figure 11/B.6
        _subtitle = """PART 6: MPC - Simulation vs. User comparisons of single cost function"""
        print(f"\n\n+++++++++++++++++++++++++++++++++++++\n{_subtitle}\n++++++++++++++++++++++++++++++++++++++\n")

        USER_ID_LIST = [f"U{i}" for i in range(1, 7)]
        USER_ID_FIXED = "U4"  # (only used if SHOW_STUDY == True)

        SIMULATION_SUBDIR = "JAC"

        TASK_CONDITION = "Virtual_Pad_Ergonomic_ISO_15_plane"

        filename = ""

        # AGGREGATE_TRIALS = True
        #########################

        REPEATED_MOVEMENTS = False

        trajectories_SIMULATION = TrajectoryData_MPC(DIRNAME_SIMULATION, SIMULATION_SUBDIR, USER_ID=USER_ID_FIXED,
                                                     TASK_CONDITION=TASK_CONDITION,
                                                     FILENAME_STUDY_TARGETPOSITIONS=FILENAME_STUDY_TARGETPOSITIONS)
        trajectories_SIMULATION.preprocess()  # AGGREGATE_TRIALS=AGGREGATE_TRIALS)

        trajectories_SUPPLEMENTARY = []
        for USER_ID in USER_ID_LIST:
            trajectories_SUPPLEMENTARY.append(
                TrajectoryData_STUDY(DIRNAME_STUDY, USER_ID=USER_ID, TASK_CONDITION=TASK_CONDITION,
                                     FILENAME_STUDY_TARGETPOSITIONS=FILENAME_STUDY_TARGETPOSITIONS))
            trajectories_SUPPLEMENTARY[-1].preprocess()  # AGGREGATE_TRIALS=AGGREGATE_TRIALS)

            # Preprocess experimental trajectories for each user (ISO Task User Study):
            if USER_ID == USER_ID_FIXED:
                trajectories_STUDY = trajectories_SUPPLEMENTARY[-1]

        #########################

        PLOTTING_ENV = "MPC-userstudy"

        ignore_trainingset_trials_mpc_userstudy = False  # will not ignore training trials here

        common_simulation_subdir = SIMULATION_SUBDIR

        #################################

        #################################
        ### SET PLOTTING PARAM VALUES ###
        #################################

        # WHICH PARTS OF DATASET? #only used if PLOTTING_ENV == "RL-UIB"
        MOVEMENT_IDS = None  # range(1,9) #[i for i in range(10) if i != 1]
        RADIUS_IDS = None
        EPISODE_IDS = [7]

        r1_FIXED = None  # r1list[5]  #only used if PLOTTING_ENV == "MPC-costweights"
        r2_FIXED = None  # r2list[-1]  #only used if PLOTTING_ENV == "MPC-costweights"

        # TASK_CONDITION_LIST_SELECTED = ["Virtual_Cursor_ID_ISO_15_plane", "Virtual_Cursor_Ergonomic_ISO_15_plane"]  #only used if PLOTTING_ENV == "MPC-taskconditions"

        # WHAT TO COMPUTE?
        EFFECTIVE_PROJECTION_PATH = (
                    PLOTTING_ENV == "RL-UIB")  # if True, projection path connects effective initial and final position instead of nominal target center positions
        USE_TARGETBOUND_AS_DIST = False  # True/False or "MinJerk-only"; if True, only plot trajectory until targetboundary is reached first (i.e., until dwell time begins); if "MinJerk-only", complete simulation trajectories are shown, but MinJerk trajectories are aligned to simulation trajectories without dwell time
        MINJERK_USER_CONSTRAINTS = True

        # WHICH/HOW MANY MOVS?
        """
        IMPORTANT INFO:
        if isinstance(trajectories, TrajectoryData_RL):
            -> TRIAL_IDS/META_IDS/N_MOVS are (meta-)indices of respective episode (or rather, of respective row of trajectories.indices)
            -> TRIAL_IDS and META_IDS are equivalent
        if isinstance(trajectories, TrajectoryData_STUDY) or isinstance(trajectories, TrajectoryData_MPC):
            -> TRIAL_IDS/META_IDS/N_MOVS are (global) indices of entire dataset
            -> TRIAL_IDS correspond to trial indices assigned during user study (last column of trajectories.indices), while META_IDS correspond to (meta-)indices of trajectories.indices itself (i.e., if some trials were removed during previous pre-processing steps, TRIAL_IDS and META_IDS differ!)
        In general, only the first not-None parameter is used, in following order: TRIAL_IDS, META_IDS, N_MOVS.
        """
        TARGET_IDS = None  # corresponds to target IDs (if PLOTTING_ENV == "RL-UIB": only for iso-task)
        TRIAL_IDS = [
            27]  # [i for i in range(65) if i not in range(28, 33)] #range(1,4) #corresponds to trial IDs [last column of self.indices] or relative (meta) index per row in self.indices; either a list of indices, "different_target_sizes" (choose N_MOVS conditions with maximum differences in target size), or None (use META_IDS)
        META_IDS = None  # index positions (i.e., sequential numbering of trials [aggregated trials, if AGGREGATE_TRIALS==True] in indices, without counting removed outliers); if None: use N_MOVS
        N_MOVS = None  # number of movements to visualize (only used, if TRIAL_IDS and META_IDS are both None (or TRIAL_IDS == "different_target_sizes"))
        AGGREGATION_VARS = []  # ["all"]  #["episode", "movement"]  #["episode", "targetoccurrence"]  #["targetoccurrence"] #["episode", "movement"]  #["episode", "radius", "movement", "target", "targetoccurrence"]

        # WHAT TO PLOT?
        PLOT_TRACKING_DISTANCE = False  # if True, plot distance between End-effector and target position instead of position (only reasonable for tracking tasks)
        PLOT_ENDEFFECTOR = True  # if True plot End-effector position and velocity, else plot qpos and qvel for joint with index JOINT_ID (see independent_joints below)
        JOINT_ID = 2  # only used if PLOT_ENDEFFECTOR == False
        PLOT_DEVIATION = False  # only if PLOT_ENDEFFECTOR == True

        # HOW TO PLOT?
        NORMALIZE_TIME = False
        PLOT_TIME_SERIES = True  # if True plot Position/Velocity/Acceleration Time Series, else plot Phasespace and Hooke plots
        PLOT_VEL_ACC = False  # if True, plot Velocity and Acceleration Time Series, else plot Position and Velocity Time Series (only used if PLOT_TIME_SERIES == True)
        PLOT_RANGES = False
        CONF_LEVEL = "min/max"  # might be between 0 and 1, or "min/max"; only used if PLOT_RANGES==True

        # WHICH BASELINE?
        SHOW_MINJERK = False
        SHOW_STUDY = True
        STUDY_ONLY = False  # only used if PLOTTING_ENV == "MPC-taskconditions"

        # PLOT (WHICH) LEGENDS AND COLORBARS?
        ENABLE_LEGENDS_AND_COLORBARS = True  # if False, legends (of axis 0) and colobars are removed
        ALLOW_DUPLICATES_BETWEEN_LEGENDS = False  # if False, legend of axis 1 only contains values not included in legend of axis 0

        # STORE PLOT?
        STORE_PLOT = True
        STORE_AXES_SEPARATELY = True  # if True, store left and right axis to separate figures

        ####

        trajectoryplot(PLOTTING_ENV, USER_ID, TASK_CONDITION,
                       common_simulation_subdir, filename, trajectories_SIMULATION,
                       trajectories_STUDY=trajectories_STUDY,
                       trajectories_SUPPLEMENTARY=trajectories_SUPPLEMENTARY,
                       REPEATED_MOVEMENTS=REPEATED_MOVEMENTS,
                       USER_ID_FIXED=USER_ID_FIXED,
                       ignore_trainingset_trials_mpc_userstudy=ignore_trainingset_trials_mpc_userstudy,
                       MOVEMENT_IDS=MOVEMENT_IDS,
                       RADIUS_IDS=RADIUS_IDS,
                       EPISODE_IDS=EPISODE_IDS,
                       r1_FIXED=r1_FIXED,
                       r2_FIXED=r2_FIXED,
                       EFFECTIVE_PROJECTION_PATH=EFFECTIVE_PROJECTION_PATH,
                       USE_TARGETBOUND_AS_DIST=USE_TARGETBOUND_AS_DIST,
                       MINJERK_USER_CONSTRAINTS=MINJERK_USER_CONSTRAINTS,
                       TARGET_IDS=TARGET_IDS,
                       TRIAL_IDS=TRIAL_IDS,
                       META_IDS=META_IDS,
                       N_MOVS=N_MOVS,
                       AGGREGATION_VARS=AGGREGATION_VARS,
                       PLOT_TRACKING_DISTANCE=PLOT_TRACKING_DISTANCE,
                       PLOT_ENDEFFECTOR=PLOT_ENDEFFECTOR,
                       JOINT_ID=JOINT_ID,
                       PLOT_DEVIATION=PLOT_DEVIATION,
                       NORMALIZE_TIME=NORMALIZE_TIME,
                       # DWELL_TIME=#DWELL_TIME,
                       PLOT_TIME_SERIES=PLOT_TIME_SERIES,
                       PLOT_VEL_ACC=PLOT_VEL_ACC,
                       PLOT_RANGES=PLOT_RANGES,
                       CONF_LEVEL=CONF_LEVEL,
                       SHOW_MINJERK=SHOW_MINJERK,
                       SHOW_STUDY=SHOW_STUDY,
                       STUDY_ONLY=STUDY_ONLY,
                       ENABLE_LEGENDS_AND_COLORBARS=ENABLE_LEGENDS_AND_COLORBARS,
                       ALLOW_DUPLICATES_BETWEEN_LEGENDS=ALLOW_DUPLICATES_BETWEEN_LEGENDS,
                       STORE_PLOT=STORE_PLOT,
                       STORE_AXES_SEPARATELY=STORE_AXES_SEPARATELY)

        PLOT_ENDEFFECTOR = False
        for JOINT_ID in range(7):
            trajectoryplot(PLOTTING_ENV, USER_ID, TASK_CONDITION,
                           common_simulation_subdir, filename, trajectories_SIMULATION,
                           trajectories_STUDY=trajectories_STUDY,
                           trajectories_SUPPLEMENTARY=trajectories_SUPPLEMENTARY,
                           REPEATED_MOVEMENTS=REPEATED_MOVEMENTS,
                           USER_ID_FIXED=USER_ID_FIXED,
                           ignore_trainingset_trials_mpc_userstudy=ignore_trainingset_trials_mpc_userstudy,
                           MOVEMENT_IDS=MOVEMENT_IDS,
                           RADIUS_IDS=RADIUS_IDS,
                           EPISODE_IDS=EPISODE_IDS,
                           r1_FIXED=r1_FIXED,
                           r2_FIXED=r2_FIXED,
                           EFFECTIVE_PROJECTION_PATH=EFFECTIVE_PROJECTION_PATH,
                           USE_TARGETBOUND_AS_DIST=USE_TARGETBOUND_AS_DIST,
                           MINJERK_USER_CONSTRAINTS=MINJERK_USER_CONSTRAINTS,
                           TARGET_IDS=TARGET_IDS,
                           TRIAL_IDS=TRIAL_IDS,
                           META_IDS=META_IDS,
                           N_MOVS=N_MOVS,
                           AGGREGATION_VARS=AGGREGATION_VARS,
                           PLOT_TRACKING_DISTANCE=PLOT_TRACKING_DISTANCE,
                           PLOT_ENDEFFECTOR=PLOT_ENDEFFECTOR,
                           JOINT_ID=JOINT_ID,
                           PLOT_DEVIATION=PLOT_DEVIATION,
                           NORMALIZE_TIME=NORMALIZE_TIME,
                           # DWELL_TIME=#DWELL_TIME,
                           PLOT_TIME_SERIES=PLOT_TIME_SERIES,
                           PLOT_VEL_ACC=PLOT_VEL_ACC,
                           PLOT_RANGES=PLOT_RANGES,
                           CONF_LEVEL=CONF_LEVEL,
                           SHOW_MINJERK=SHOW_MINJERK,
                           SHOW_STUDY=SHOW_STUDY,
                           STUDY_ONLY=STUDY_ONLY,
                           ENABLE_LEGENDS_AND_COLORBARS=JOINT_ID==0,
                           ALLOW_DUPLICATES_BETWEEN_LEGENDS=ALLOW_DUPLICATES_BETWEEN_LEGENDS,
                           STORE_PLOT=STORE_PLOT,
                           STORE_AXES_SEPARATELY=STORE_AXES_SEPARATELY)

    ########################################

    if 7 in _active_parts:
        # Figure 12/B.11/14/B.12
        _subtitle = """PART 7: MPC - COMPARISON OF COST WEIGHTS OF A GIVEN COST FUNCTION"""
        print(f"\n\n+++++++++++++++++++++++++++++++++++++\n{_subtitle}\n++++++++++++++++++++++++++++++++++++++\n")

        USER_ID = "U4"  # (only used if SHOW_STUDY == True)

        TASK_CONDITION = "Virtual_Cursor_Ergonomic_ISO_15_plane"

        COST_FUNCTION = "accjoint"  # "accjoint", "dist", "ctc"
        #########################

        REPEATED_MOVEMENTS = False

        filename = ""

        mainfolder = DIRNAME_SIMULATION

        pre = "JAC_r_10x10_r1_"
        paramfile = os.path.join(mainfolder, "parameters/params_jac.csv")

        paramdata = pd.read_csv(paramfile)
        param_0 = \
            paramdata.where(paramdata["participant"] == USER_ID).where(paramdata["condition"] == TASK_CONDITION).loc[:,
            "param_0"].dropna().iloc[0]
        param_1 = \
            paramdata.where(paramdata["participant"] == USER_ID).where(paramdata["condition"] == TASK_CONDITION).loc[:,
            "param_1"].dropna().iloc[0]

        r1list = np.hstack([np.arange(0, 200 * param_0 - 1e-12, param_0 / 0.05)])
        r2list = np.hstack([np.arange(0, 10 * param_1 - 1e-12, param_1 / 1)])

        mid = "_r2_"
        post = ""  # "_cso_sae"

        trajectories_SIMULATION = []
        for r1 in r1list:
            for r2 in r2list:
                outputprefix = f"{pre}{float(r1):.8e}{mid}{float(r2):.8e}{post}"
                try:
                    trajectories_SIMULATION.append(
                        TrajectoryData_MPC(DIRNAME_SIMULATION, outputprefix, USER_ID=USER_ID,
                                           TASK_CONDITION=TASK_CONDITION,
                                           FILENAME_STUDY_TARGETPOSITIONS=FILENAME_STUDY_TARGETPOSITIONS))
                    trajectories_SIMULATION[-1].preprocess()
                except FileNotFoundError as e:
                    if e.filename.endswith("/complete.csv"):
                        logging.info(f"Need to create complete.csv for '{outputprefix}'.")
                        trajectory_plots_data_ALL, target_switch_indices_ALL = preprocess_movement_data_simulation_complete(
                            DIRNAME_SIMULATION, DIRNAME_STUDY,
                            user_initial_delay=0.5, start_outside_target=False, initial_acceleration_constraint=1,
                            scale_movement_times=False, scale_to_mean_movement_times=False, participant_list=[USER_ID],
                            task_condition_list=[TASK_CONDITION], simulation_subdir=outputprefix)
                        trajectories_SIMULATION.append(
                            TrajectoryData_MPC(DIRNAME_SIMULATION, outputprefix, USER_ID=USER_ID,
                                               TASK_CONDITION=TASK_CONDITION,
                                               FILENAME_STUDY_TARGETPOSITIONS=FILENAME_STUDY_TARGETPOSITIONS))
                        trajectories_SIMULATION[-1].preprocess()
                    else:
                        raise FileNotFoundError
        #########################

        PLOTTING_ENV = "MPC-costweights"

        common_simulation_subdir = "".join(
            [pre, mid, post])  # "".join([i for i,j,k in zip(*SIMULATION_SUBDIR_LIST) if i==j==k])
        if len(common_simulation_subdir) == 0:
            common_simulation_subdir = "ALLCOSTS"
        elif common_simulation_subdir[-1] in ["-", "_"]:
            common_simulation_subdir = common_simulation_subdir[:-1]

        #################################

        #################################
        ### SET PLOTTING PARAM VALUES ###
        #################################

        # WHICH PARTS OF DATASET? #only used if PLOTTING_ENV == "RL-UIB"
        MOVEMENT_IDS = None  # range(1,9) #[i for i in range(10) if i != 1]
        RADIUS_IDS = None
        EPISODE_IDS = [7]

        r1_FIXED = None  # r1list[5]  #only used if PLOTTING_ENV == "MPC-costweights"
        r2_FIXED = r2list[-1]  # only used if PLOTTING_ENV == "MPC-costweights"

        # TASK_CONDITION_LIST_SELECTED = ["Virtual_Cursor_ID_ISO_15_plane", "Virtual_Cursor_Ergonomic_ISO_15_plane"]  #only used if PLOTTING_ENV == "MPC-taskconditions"

        # WHAT TO COMPUTE?
        EFFECTIVE_PROJECTION_PATH = (
                PLOTTING_ENV == "RL-UIB")  # if True, projection path connects effective initial and final position instead of nominal target center positions
        USE_TARGETBOUND_AS_DIST = False  # True/False or "MinJerk-only"; if True, only plot trajectory until targetboundary is reached first (i.e., until dwell time begins); if "MinJerk-only", complete simulation trajectories are shown, but MinJerk trajectories are aligned to simulation trajectories without dwell time
        MINJERK_USER_CONSTRAINTS = True

        # WHICH/HOW MANY MOVS?
        """
        IMPORTANT INFO:
        if isinstance(trajectories, TrajectoryData_RL):
            -> TRIAL_IDS/META_IDS/N_MOVS are (meta-)indices of respective episode (or rather, of respective row of trajectories.indices)
            -> TRIAL_IDS and META_IDS are equivalent
        if isinstance(trajectories, TrajectoryData_STUDY) or isinstance(trajectories, TrajectoryData_MPC):
            -> TRIAL_IDS/META_IDS/N_MOVS are (global) indices of entire dataset
            -> TRIAL_IDS correspond to trial indices assigned during user study (last column of trajectories.indices), while META_IDS correspond to (meta-)indices of trajectories.indices itself (i.e., if some trials were removed during previous pre-processing steps, TRIAL_IDS and META_IDS differ!)
        In general, only the first not-None parameter is used, in following order: TRIAL_IDS, META_IDS, N_MOVS.
        """
        TARGET_IDS = None  # corresponds to target IDs (if PLOTTING_ENV == "RL-UIB": only for iso-task)
        TRIAL_IDS = [
            4]  # [i for i in range(65) if i not in range(28, 33)] #range(1,4) #corresponds to trial IDs [last column of self.indices] or relative (meta) index per row in self.indices; either a list of indices, "different_target_sizes" (choose N_MOVS conditions with maximum differences in target size), or None (use META_IDS)
        META_IDS = None  # index positions (i.e., sequential numbering of trials [aggregated trials, if AGGREGATE_TRIALS==True] in indices, without counting removed outliers); if None: use N_MOVS
        N_MOVS = None  # number of movements to visualize (only used, if TRIAL_IDS and META_IDS are both None (or TRIAL_IDS == "different_target_sizes"))
        AGGREGATION_VARS = []  # ["all"]  #["episode", "movement"]  #["episode", "targetoccurrence"]  #["targetoccurrence"] #["episode", "movement"]  #["episode", "radius", "movement", "target", "targetoccurrence"]

        # WHAT TO PLOT?
        PLOT_TRACKING_DISTANCE = False  # if True, plot distance between End-effector and target position instead of position (only reasonable for tracking tasks)
        PLOT_ENDEFFECTOR = True  # if True plot End-effector position and velocity, else plot qpos and qvel for joint with index JOINT_ID (see independent_joints below)
        JOINT_ID = 2  # only used if PLOT_ENDEFFECTOR == False
        PLOT_DEVIATION = False  # only if PLOT_ENDEFFECTOR == True

        # HOW TO PLOT?
        NORMALIZE_TIME = False
        PLOT_TIME_SERIES = True  # if True plot Position/Velocity/Acceleration Time Series, else plot Phasespace and Hooke plots
        PLOT_VEL_ACC = False  # if True, plot Velocity and Acceleration Time Series, else plot Position and Velocity Time Series (only used if PLOT_TIME_SERIES == True)
        PLOT_RANGES = False
        CONF_LEVEL = "min/max"  # might be between 0 and 1, or "min/max"; only used if PLOT_RANGES==True

        # WHICH BASELINE?
        SHOW_MINJERK = False
        SHOW_STUDY = False
        STUDY_ONLY = False  # only used if PLOTTING_ENV == "MPC-taskconditions"

        # PLOT (WHICH) LEGENDS AND COLORBARS?
        ENABLE_LEGENDS_AND_COLORBARS = True  # if False, legends (of axis 0) and colobars are removed
        ALLOW_DUPLICATES_BETWEEN_LEGENDS = False  # if False, legend of axis 1 only contains values not included in legend of axis 0

        # STORE PLOT?
        STORE_PLOT = True
        STORE_AXES_SEPARATELY = True  # if True, store left and right axis to separate figures

        ####

        trajectoryplot(PLOTTING_ENV, USER_ID, TASK_CONDITION,
                       common_simulation_subdir, filename, trajectories_SIMULATION,
                       REPEATED_MOVEMENTS=REPEATED_MOVEMENTS,
                       MOVEMENT_IDS=MOVEMENT_IDS,
                       RADIUS_IDS=RADIUS_IDS,
                       EPISODE_IDS=EPISODE_IDS,
                       r1_FIXED=r1_FIXED,
                       r2_FIXED=r2_FIXED,
                       r1list=r1list,
                       r2list=r2list,
                       COST_FUNCTION=COST_FUNCTION,
                       EFFECTIVE_PROJECTION_PATH=EFFECTIVE_PROJECTION_PATH,
                       USE_TARGETBOUND_AS_DIST=USE_TARGETBOUND_AS_DIST,
                       MINJERK_USER_CONSTRAINTS=MINJERK_USER_CONSTRAINTS,
                       TARGET_IDS=TARGET_IDS,
                       TRIAL_IDS=TRIAL_IDS,
                       META_IDS=META_IDS,
                       N_MOVS=N_MOVS,
                       AGGREGATION_VARS=AGGREGATION_VARS,
                       PLOT_TRACKING_DISTANCE=PLOT_TRACKING_DISTANCE,
                       PLOT_ENDEFFECTOR=PLOT_ENDEFFECTOR,
                       JOINT_ID=JOINT_ID,
                       PLOT_DEVIATION=PLOT_DEVIATION,
                       NORMALIZE_TIME=NORMALIZE_TIME,
                       # DWELL_TIME=#DWELL_TIME,
                       PLOT_TIME_SERIES=PLOT_TIME_SERIES,
                       PLOT_VEL_ACC=PLOT_VEL_ACC,
                       PLOT_RANGES=PLOT_RANGES,
                       CONF_LEVEL=CONF_LEVEL,
                       SHOW_MINJERK=SHOW_MINJERK,
                       SHOW_STUDY=SHOW_STUDY,
                       STUDY_ONLY=STUDY_ONLY,
                       ENABLE_LEGENDS_AND_COLORBARS=ENABLE_LEGENDS_AND_COLORBARS,
                       ALLOW_DUPLICATES_BETWEEN_LEGENDS=ALLOW_DUPLICATES_BETWEEN_LEGENDS,
                       STORE_PLOT=STORE_PLOT,
                       STORE_AXES_SEPARATELY=STORE_AXES_SEPARATELY)

        PLOT_ENDEFFECTOR = False
        for JOINT_ID in range(7):
            trajectoryplot(PLOTTING_ENV, USER_ID, TASK_CONDITION,
                           common_simulation_subdir, filename, trajectories_SIMULATION,
                           REPEATED_MOVEMENTS=REPEATED_MOVEMENTS,
                           MOVEMENT_IDS=MOVEMENT_IDS,
                           RADIUS_IDS=RADIUS_IDS,
                           EPISODE_IDS=EPISODE_IDS,
                           r1_FIXED=r1_FIXED,
                           r2_FIXED=r2_FIXED,
                           r1list=r1list,
                           r2list=r2list,
                           COST_FUNCTION=COST_FUNCTION,
                           EFFECTIVE_PROJECTION_PATH=EFFECTIVE_PROJECTION_PATH,
                           USE_TARGETBOUND_AS_DIST=USE_TARGETBOUND_AS_DIST,
                           MINJERK_USER_CONSTRAINTS=MINJERK_USER_CONSTRAINTS,
                           TARGET_IDS=TARGET_IDS,
                           TRIAL_IDS=TRIAL_IDS,
                           META_IDS=META_IDS,
                           N_MOVS=N_MOVS,
                           AGGREGATION_VARS=AGGREGATION_VARS,
                           PLOT_TRACKING_DISTANCE=PLOT_TRACKING_DISTANCE,
                           PLOT_ENDEFFECTOR=PLOT_ENDEFFECTOR,
                           JOINT_ID=JOINT_ID,
                           PLOT_DEVIATION=PLOT_DEVIATION,
                           NORMALIZE_TIME=NORMALIZE_TIME,
                           # DWELL_TIME=#DWELL_TIME,
                           PLOT_TIME_SERIES=PLOT_TIME_SERIES,
                           PLOT_VEL_ACC=PLOT_VEL_ACC,
                           PLOT_RANGES=PLOT_RANGES,
                           CONF_LEVEL=CONF_LEVEL,
                           SHOW_MINJERK=SHOW_MINJERK,
                           SHOW_STUDY=SHOW_STUDY,
                           STUDY_ONLY=STUDY_ONLY,
                           ENABLE_LEGENDS_AND_COLORBARS=JOINT_ID==0,
                           ALLOW_DUPLICATES_BETWEEN_LEGENDS=ALLOW_DUPLICATES_BETWEEN_LEGENDS,
                           STORE_PLOT=STORE_PLOT,
                           STORE_AXES_SEPARATELY=STORE_AXES_SEPARATELY)

        r1_FIXED = r1list[5]  # only used if PLOTTING_ENV == "MPC-costweights"
        r2_FIXED = None  # only used if PLOTTING_ENV == "MPC-costweights"

        PLOT_ENDEFFECTOR = True
        ENABLE_LEGENDS_AND_COLORBARS = True
        trajectoryplot(PLOTTING_ENV, USER_ID, TASK_CONDITION,
                       common_simulation_subdir, filename, trajectories_SIMULATION,
                       REPEATED_MOVEMENTS=REPEATED_MOVEMENTS,
                       MOVEMENT_IDS=MOVEMENT_IDS,
                       RADIUS_IDS=RADIUS_IDS,
                       EPISODE_IDS=EPISODE_IDS,
                       r1_FIXED=r1_FIXED,
                       r2_FIXED=r2_FIXED,
                       r1list=r1list,
                       r2list=r2list,
                       COST_FUNCTION=COST_FUNCTION,
                       EFFECTIVE_PROJECTION_PATH=EFFECTIVE_PROJECTION_PATH,
                       USE_TARGETBOUND_AS_DIST=USE_TARGETBOUND_AS_DIST,
                       MINJERK_USER_CONSTRAINTS=MINJERK_USER_CONSTRAINTS,
                       TARGET_IDS=TARGET_IDS,
                       TRIAL_IDS=TRIAL_IDS,
                       META_IDS=META_IDS,
                       N_MOVS=N_MOVS,
                       AGGREGATION_VARS=AGGREGATION_VARS,
                       PLOT_TRACKING_DISTANCE=PLOT_TRACKING_DISTANCE,
                       PLOT_ENDEFFECTOR=PLOT_ENDEFFECTOR,
                       JOINT_ID=JOINT_ID,
                       PLOT_DEVIATION=PLOT_DEVIATION,
                       NORMALIZE_TIME=NORMALIZE_TIME,
                       # DWELL_TIME=#DWELL_TIME,
                       PLOT_TIME_SERIES=PLOT_TIME_SERIES,
                       PLOT_VEL_ACC=PLOT_VEL_ACC,
                       PLOT_RANGES=PLOT_RANGES,
                       CONF_LEVEL=CONF_LEVEL,
                       SHOW_MINJERK=SHOW_MINJERK,
                       SHOW_STUDY=SHOW_STUDY,
                       STUDY_ONLY=STUDY_ONLY,
                       ENABLE_LEGENDS_AND_COLORBARS=ENABLE_LEGENDS_AND_COLORBARS,
                       ALLOW_DUPLICATES_BETWEEN_LEGENDS=ALLOW_DUPLICATES_BETWEEN_LEGENDS,
                       STORE_PLOT=STORE_PLOT,
                       STORE_AXES_SEPARATELY=STORE_AXES_SEPARATELY)

        PLOT_ENDEFFECTOR = False
        for JOINT_ID in range(7):
            trajectoryplot(PLOTTING_ENV, USER_ID, TASK_CONDITION,
                           common_simulation_subdir, filename, trajectories_SIMULATION,
                           REPEATED_MOVEMENTS=REPEATED_MOVEMENTS,
                           MOVEMENT_IDS=MOVEMENT_IDS,
                           RADIUS_IDS=RADIUS_IDS,
                           EPISODE_IDS=EPISODE_IDS,
                           r1_FIXED=r1_FIXED,
                           r2_FIXED=r2_FIXED,
                           r1list=r1list,
                           r2list=r2list,
                           COST_FUNCTION=COST_FUNCTION,
                           EFFECTIVE_PROJECTION_PATH=EFFECTIVE_PROJECTION_PATH,
                           USE_TARGETBOUND_AS_DIST=USE_TARGETBOUND_AS_DIST,
                           MINJERK_USER_CONSTRAINTS=MINJERK_USER_CONSTRAINTS,
                           TARGET_IDS=TARGET_IDS,
                           TRIAL_IDS=TRIAL_IDS,
                           META_IDS=META_IDS,
                           N_MOVS=N_MOVS,
                           AGGREGATION_VARS=AGGREGATION_VARS,
                           PLOT_TRACKING_DISTANCE=PLOT_TRACKING_DISTANCE,
                           PLOT_ENDEFFECTOR=PLOT_ENDEFFECTOR,
                           JOINT_ID=JOINT_ID,
                           PLOT_DEVIATION=PLOT_DEVIATION,
                           NORMALIZE_TIME=NORMALIZE_TIME,
                           # DWELL_TIME=#DWELL_TIME,
                           PLOT_TIME_SERIES=PLOT_TIME_SERIES,
                           PLOT_VEL_ACC=PLOT_VEL_ACC,
                           PLOT_RANGES=PLOT_RANGES,
                           CONF_LEVEL=CONF_LEVEL,
                           SHOW_MINJERK=SHOW_MINJERK,
                           SHOW_STUDY=SHOW_STUDY,
                           STUDY_ONLY=STUDY_ONLY,
                           ENABLE_LEGENDS_AND_COLORBARS=JOINT_ID==0,
                           ALLOW_DUPLICATES_BETWEEN_LEGENDS=ALLOW_DUPLICATES_BETWEEN_LEGENDS,
                           STORE_PLOT=STORE_PLOT,
                           STORE_AXES_SEPARATELY=STORE_AXES_SEPARATELY)

    ########################################

    if 8 in _active_parts:
        # Figure 15/B.13
        _subtitle = """PART 8: MPC - COMPARISON OF DIFFERENT MPC HORIZONS N"""
        print(f"\n\n+++++++++++++++++++++++++++++++++++++\n{_subtitle}\n++++++++++++++++++++++++++++++++++++++\n")

        USER_ID = "U4"  # (only used if SHOW_STUDY == True)

        HORIZONS_LIST = list(range(2, 7)) + [8, 18]
        SIMULATION_SUBDIR_LIST = [f"JAC_N_N{n}" for n in HORIZONS_LIST]

        TASK_CONDITION = "Virtual_Cursor_Ergonomic_ISO_15_plane"

        # AGGREGATE_TRIALS = False
        #########################

        REPEATED_MOVEMENTS = False

        filename = ""

        trajectories_SIMULATION = []
        for SIMULATION_SUBDIR_CURRENT in SIMULATION_SUBDIR_LIST:
            try:
                trajectories_SIMULATION.append(
                    TrajectoryData_MPC(DIRNAME_SIMULATION, SIMULATION_SUBDIR_CURRENT, USER_ID=USER_ID,
                                       TASK_CONDITION=TASK_CONDITION,
                                       FILENAME_STUDY_TARGETPOSITIONS=FILENAME_STUDY_TARGETPOSITIONS))
                trajectories_SIMULATION[-1].preprocess()
            except FileNotFoundError as e:
                if e.filename.endswith("/complete.csv"):
                    logging.info(f"Need to create complete.csv for '{SIMULATION_SUBDIR_CURRENT}'.")
                    trajectory_plots_data_ALL, target_switch_indices_ALL = preprocess_movement_data_simulation_complete(
                        DIRNAME_SIMULATION, DIRNAME_STUDY,
                        user_initial_delay=0.5, start_outside_target=False, initial_acceleration_constraint=1,
                        scale_movement_times=False, scale_to_mean_movement_times=False, participant_list=[USER_ID],
                        task_condition_list=[TASK_CONDITION], simulation_subdir=SIMULATION_SUBDIR_CURRENT)
                    trajectories_SIMULATION.append(
                        TrajectoryData_MPC(DIRNAME_SIMULATION, SIMULATION_SUBDIR_CURRENT, USER_ID=USER_ID,
                                           TASK_CONDITION=TASK_CONDITION,
                                           FILENAME_STUDY_TARGETPOSITIONS=FILENAME_STUDY_TARGETPOSITIONS))
                    trajectories_SIMULATION[-1].preprocess()
                else:
                    raise FileNotFoundError

        # Preprocess experimental trajectories (ISO Task User Study):
        trajectories_STUDY = TrajectoryData_STUDY(DIRNAME_STUDY, USER_ID=USER_ID, TASK_CONDITION=TASK_CONDITION,
                                                  FILENAME_STUDY_TARGETPOSITIONS=FILENAME_STUDY_TARGETPOSITIONS)
        trajectories_STUDY.preprocess()  # AGGREGATE_TRIALS=AGGREGATE_TRIALS)

        #########################

        PLOTTING_ENV = "MPC-horizons"

        common_simulation_subdir = "".join([i[0] for i in zip(*SIMULATION_SUBDIR_LIST) if all(j == i[0] for j in i)])
        if len(common_simulation_subdir) == 0:
            common_simulation_subdir = "ALLCOSTS"
        elif common_simulation_subdir[-1] in ["-", "_"]:
            common_simulation_subdir = common_simulation_subdir[:-1]

        pprint_range_or_intlist = lambda x: (
            f"{x.start}-{x.stop - 1}" if x.start < x.stop - 1 else f"{x.start}" if x.start == x.stop - 1 else "ERROR") if isinstance(
            x, range) else (
            (f"{min(x)}-{max(x)}" if min(x) != max(x) else f"{min(x)}") if set(range(min(x), max(x) + 1)) == set(
                x) else ",".join([str(i) for i in sorted(set(x))])) if isinstance(x, list) or (
                isinstance(x, np.ndarray) and x.ndim == 1) else f"0-{x - 1}" if isinstance(x, int) else "ERROR"
        common_simulation_subdir += pprint_range_or_intlist(HORIZONS_LIST)

        #################################

        #################################
        ### SET PLOTTING PARAM VALUES ###
        #################################

        # WHICH PARTS OF DATASET? #only used if PLOTTING_ENV == "RL-UIB"
        MOVEMENT_IDS = None  # range(1,9) #[i for i in range(10) if i != 1]
        RADIUS_IDS = None
        EPISODE_IDS = [7]

        r1_FIXED = None  # r1list[5]  #only used if PLOTTING_ENV == "MPC-costweights"
        r2_FIXED = None  # r2list[-1]  # only used if PLOTTING_ENV == "MPC-costweights"

        # TASK_CONDITION_LIST_SELECTED = ["Virtual_Cursor_ID_ISO_15_plane", "Virtual_Cursor_Ergonomic_ISO_15_plane"]  #only used if PLOTTING_ENV == "MPC-taskconditions"

        # WHAT TO COMPUTE?
        EFFECTIVE_PROJECTION_PATH = (
                PLOTTING_ENV == "RL-UIB")  # if True, projection path connects effective initial and final position instead of nominal target center positions
        USE_TARGETBOUND_AS_DIST = False  # True/False or "MinJerk-only"; if True, only plot trajectory until targetboundary is reached first (i.e., until dwell time begins); if "MinJerk-only", complete simulation trajectories are shown, but MinJerk trajectories are aligned to simulation trajectories without dwell time
        MINJERK_USER_CONSTRAINTS = True

        # WHICH/HOW MANY MOVS?
        """
        IMPORTANT INFO:
        if isinstance(trajectories, TrajectoryData_RL):
            -> TRIAL_IDS/META_IDS/N_MOVS are (meta-)indices of respective episode (or rather, of respective row of trajectories.indices)
            -> TRIAL_IDS and META_IDS are equivalent
        if isinstance(trajectories, TrajectoryData_STUDY) or isinstance(trajectories, TrajectoryData_MPC):
            -> TRIAL_IDS/META_IDS/N_MOVS are (global) indices of entire dataset
            -> TRIAL_IDS correspond to trial indices assigned during user study (last column of trajectories.indices), while META_IDS correspond to (meta-)indices of trajectories.indices itself (i.e., if some trials were removed during previous pre-processing steps, TRIAL_IDS and META_IDS differ!)
        In general, only the first not-None parameter is used, in following order: TRIAL_IDS, META_IDS, N_MOVS.
        """
        TARGET_IDS = None  # corresponds to target IDs (if PLOTTING_ENV == "RL-UIB": only for iso-task)
        TRIAL_IDS = [
            4]  # [i for i in range(65) if i not in range(28, 33)] #range(1,4) #corresponds to trial IDs [last column of self.indices] or relative (meta) index per row in self.indices; either a list of indices, "different_target_sizes" (choose N_MOVS conditions with maximum differences in target size), or None (use META_IDS)
        META_IDS = None  # index positions (i.e., sequential numbering of trials [aggregated trials, if AGGREGATE_TRIALS==True] in indices, without counting removed outliers); if None: use N_MOVS
        N_MOVS = None  # number of movements to visualize (only used, if TRIAL_IDS and META_IDS are both None (or TRIAL_IDS == "different_target_sizes"))
        AGGREGATION_VARS = []  # ["all"]  #["episode", "movement"]  #["episode", "targetoccurrence"]  #["targetoccurrence"] #["episode", "movement"]  #["episode", "radius", "movement", "target", "targetoccurrence"]

        # WHAT TO PLOT?
        PLOT_TRACKING_DISTANCE = False  # if True, plot distance between End-effector and target position instead of position (only reasonable for tracking tasks)
        PLOT_ENDEFFECTOR = True  # if True plot End-effector position and velocity, else plot qpos and qvel for joint with index JOINT_ID (see independent_joints below)
        JOINT_ID = 2  # only used if PLOT_ENDEFFECTOR == False
        PLOT_DEVIATION = False  # only if PLOT_ENDEFFECTOR == True

        # HOW TO PLOT?
        NORMALIZE_TIME = False
        PLOT_TIME_SERIES = True  # if True plot Position/Velocity/Acceleration Time Series, else plot Phasespace and Hooke plots
        PLOT_VEL_ACC = False  # if True, plot Velocity and Acceleration Time Series, else plot Position and Velocity Time Series (only used if PLOT_TIME_SERIES == True)
        PLOT_RANGES = False
        CONF_LEVEL = "min/max"  # might be between 0 and 1, or "min/max"; only used if PLOT_RANGES==True

        # WHICH BASELINE?
        SHOW_MINJERK = False
        SHOW_STUDY = False
        STUDY_ONLY = False  # only used if PLOTTING_ENV == "MPC-taskconditions"

        # PLOT (WHICH) LEGENDS AND COLORBARS?
        ENABLE_LEGENDS_AND_COLORBARS = True  # if False, legends (of axis 0) and colobars are removed
        ALLOW_DUPLICATES_BETWEEN_LEGENDS = False  # if False, legend of axis 1 only contains values not included in legend of axis 0

        # STORE PLOT?
        STORE_PLOT = True
        STORE_AXES_SEPARATELY = True  # if True, store left and right axis to separate figures

        ####

        trajectoryplot(PLOTTING_ENV, USER_ID, TASK_CONDITION,
                       common_simulation_subdir, filename, trajectories_SIMULATION,
                       REPEATED_MOVEMENTS=REPEATED_MOVEMENTS,
                       MOVEMENT_IDS=MOVEMENT_IDS,
                       RADIUS_IDS=RADIUS_IDS,
                       EPISODE_IDS=EPISODE_IDS,
                       r1_FIXED=r1_FIXED,
                       r2_FIXED=r2_FIXED,
                       EFFECTIVE_PROJECTION_PATH=EFFECTIVE_PROJECTION_PATH,
                       USE_TARGETBOUND_AS_DIST=USE_TARGETBOUND_AS_DIST,
                       MINJERK_USER_CONSTRAINTS=MINJERK_USER_CONSTRAINTS,
                       TARGET_IDS=TARGET_IDS,
                       TRIAL_IDS=TRIAL_IDS,
                       META_IDS=META_IDS,
                       N_MOVS=N_MOVS,
                       AGGREGATION_VARS=AGGREGATION_VARS,
                       PLOT_TRACKING_DISTANCE=PLOT_TRACKING_DISTANCE,
                       PLOT_ENDEFFECTOR=PLOT_ENDEFFECTOR,
                       JOINT_ID=JOINT_ID,
                       PLOT_DEVIATION=PLOT_DEVIATION,
                       NORMALIZE_TIME=NORMALIZE_TIME,
                       # DWELL_TIME=#DWELL_TIME,
                       PLOT_TIME_SERIES=PLOT_TIME_SERIES,
                       PLOT_VEL_ACC=PLOT_VEL_ACC,
                       PLOT_RANGES=PLOT_RANGES,
                       CONF_LEVEL=CONF_LEVEL,
                       SHOW_MINJERK=SHOW_MINJERK,
                       SHOW_STUDY=SHOW_STUDY,
                       STUDY_ONLY=STUDY_ONLY,
                       ENABLE_LEGENDS_AND_COLORBARS=ENABLE_LEGENDS_AND_COLORBARS,
                       ALLOW_DUPLICATES_BETWEEN_LEGENDS=ALLOW_DUPLICATES_BETWEEN_LEGENDS,
                       STORE_PLOT=STORE_PLOT,
                       STORE_AXES_SEPARATELY=STORE_AXES_SEPARATELY)

        PLOT_ENDEFFECTOR = False
        for JOINT_ID in range(7):
            trajectoryplot(PLOTTING_ENV, USER_ID, TASK_CONDITION,
                           common_simulation_subdir, filename, trajectories_SIMULATION,
                           REPEATED_MOVEMENTS=REPEATED_MOVEMENTS,
                           MOVEMENT_IDS=MOVEMENT_IDS,
                           RADIUS_IDS=RADIUS_IDS,
                           EPISODE_IDS=EPISODE_IDS,
                           r1_FIXED=r1_FIXED,
                           r2_FIXED=r2_FIXED,
                           EFFECTIVE_PROJECTION_PATH=EFFECTIVE_PROJECTION_PATH,
                           USE_TARGETBOUND_AS_DIST=USE_TARGETBOUND_AS_DIST,
                           MINJERK_USER_CONSTRAINTS=MINJERK_USER_CONSTRAINTS,
                           TARGET_IDS=TARGET_IDS,
                           TRIAL_IDS=TRIAL_IDS,
                           META_IDS=META_IDS,
                           N_MOVS=N_MOVS,
                           AGGREGATION_VARS=AGGREGATION_VARS,
                           PLOT_TRACKING_DISTANCE=PLOT_TRACKING_DISTANCE,
                           PLOT_ENDEFFECTOR=PLOT_ENDEFFECTOR,
                           JOINT_ID=JOINT_ID,
                           PLOT_DEVIATION=PLOT_DEVIATION,
                           NORMALIZE_TIME=NORMALIZE_TIME,
                           # DWELL_TIME=#DWELL_TIME,
                           PLOT_TIME_SERIES=PLOT_TIME_SERIES,
                           PLOT_VEL_ACC=PLOT_VEL_ACC,
                           PLOT_RANGES=PLOT_RANGES,
                           CONF_LEVEL=CONF_LEVEL,
                           SHOW_MINJERK=SHOW_MINJERK,
                           SHOW_STUDY=SHOW_STUDY,
                           STUDY_ONLY=STUDY_ONLY,
                           ENABLE_LEGENDS_AND_COLORBARS=JOINT_ID==0,
                           ALLOW_DUPLICATES_BETWEEN_LEGENDS=ALLOW_DUPLICATES_BETWEEN_LEGENDS,
                           STORE_PLOT=STORE_PLOT,
                           STORE_AXES_SEPARATELY=STORE_AXES_SEPARATELY)

    ########################################

    if 9 in _active_parts:
        # Figure B.5
        _subtitle = """PART 9: MPC - Simulation vs. User comparisons for different movement directions"""
        print(f"\n\n+++++++++++++++++++++++++++++++++++++\n{_subtitle}\n++++++++++++++++++++++++++++++++++++++\n")

        USER_ID_LIST = [f"U{i}" for i in range(1, 7)]
        USER_ID_FIXED = "U4"  # (only used if SHOW_STUDY == True)

        SIMULATION_SUBDIR = "JAC"

        TASK_CONDITION = "Virtual_Pad_Ergonomic_ISO_15_plane"

        filename = ""

        # AGGREGATE_TRIALS = True
        #########################

        REPEATED_MOVEMENTS = False

        trajectories_SIMULATION = TrajectoryData_MPC(DIRNAME_SIMULATION, SIMULATION_SUBDIR, USER_ID=USER_ID_FIXED,
                                                     TASK_CONDITION=TASK_CONDITION,
                                                     FILENAME_STUDY_TARGETPOSITIONS=FILENAME_STUDY_TARGETPOSITIONS)
        trajectories_SIMULATION.preprocess()  # AGGREGATE_TRIALS=AGGREGATE_TRIALS)

        trajectories_SUPPLEMENTARY = []
        for USER_ID in USER_ID_LIST:
            trajectories_SUPPLEMENTARY.append(
                TrajectoryData_STUDY(DIRNAME_STUDY, USER_ID=USER_ID, TASK_CONDITION=TASK_CONDITION,
                                     FILENAME_STUDY_TARGETPOSITIONS=FILENAME_STUDY_TARGETPOSITIONS))
            trajectories_SUPPLEMENTARY[-1].preprocess()  # AGGREGATE_TRIALS=AGGREGATE_TRIALS)

            # Preprocess experimental trajectories for each user (ISO Task User Study):
            if USER_ID == USER_ID_FIXED:
                trajectories_STUDY = trajectories_SUPPLEMENTARY[-1]

        #########################

        PLOTTING_ENV = "MPC-simvsuser-colored"

        ignore_trainingset_trials_mpc_userstudy = False  # will not ignore training trials here

        common_simulation_subdir = SIMULATION_SUBDIR

        #################################

        #################################
        ### SET PLOTTING PARAM VALUES ###
        #################################

        # WHICH PARTS OF DATASET? #only used if PLOTTING_ENV == "RL-UIB"
        MOVEMENT_IDS = None  # range(1,9) #[i for i in range(10) if i != 1]
        RADIUS_IDS = None
        EPISODE_IDS = [7]

        r1_FIXED = None  # r1list[5]  #only used if PLOTTING_ENV == "MPC-costweights"
        r2_FIXED = None  # r2list[-1]  #only used if PLOTTING_ENV == "MPC-costweights"

        # TASK_CONDITION_LIST_SELECTED = ["Virtual_Cursor_ID_ISO_15_plane", "Virtual_Cursor_Ergonomic_ISO_15_plane"]  #only used if PLOTTING_ENV == "MPC-taskconditions"

        # WHAT TO COMPUTE?
        EFFECTIVE_PROJECTION_PATH = (
                    PLOTTING_ENV == "RL-UIB")  # if True, projection path connects effective initial and final position instead of nominal target center positions
        USE_TARGETBOUND_AS_DIST = False  # True/False or "MinJerk-only"; if True, only plot trajectory until targetboundary is reached first (i.e., until dwell time begins); if "MinJerk-only", complete simulation trajectories are shown, but MinJerk trajectories are aligned to simulation trajectories without dwell time
        MINJERK_USER_CONSTRAINTS = True

        # WHICH/HOW MANY MOVS?
        """
        IMPORTANT INFO:
        if isinstance(trajectories, TrajectoryData_RL):
            -> TRIAL_IDS/META_IDS/N_MOVS are (meta-)indices of respective episode (or rather, of respective row of trajectories.indices)
            -> TRIAL_IDS and META_IDS are equivalent
        if isinstance(trajectories, TrajectoryData_STUDY) or isinstance(trajectories, TrajectoryData_MPC):
            -> TRIAL_IDS/META_IDS/N_MOVS are (global) indices of entire dataset
            -> TRIAL_IDS correspond to trial indices assigned during user study (last column of trajectories.indices), while META_IDS correspond to (meta-)indices of trajectories.indices itself (i.e., if some trials were removed during previous pre-processing steps, TRIAL_IDS and META_IDS differ!)
        In general, only the first not-None parameter is used, in following order: TRIAL_IDS, META_IDS, N_MOVS.
        """
        TARGET_IDS = range(1, 4)  # corresponds to target IDs (if PLOTTING_ENV == "RL-UIB": only for iso-task)
        TRIAL_IDS = range(13)  # [i for i in range(65) if i not in range(28, 33)] #range(1,4) #corresponds to trial IDs [last column of self.indices] or relative (meta) index per row in self.indices; either a list of indices, "different_target_sizes" (choose N_MOVS conditions with maximum differences in target size), or None (use META_IDS)
        META_IDS = None  # index positions (i.e., sequential numbering of trials [aggregated trials, if AGGREGATE_TRIALS==True] in indices, without counting removed outliers); if None: use N_MOVS
        N_MOVS = None  # number of movements to visualize (only used, if TRIAL_IDS and META_IDS are both None (or TRIAL_IDS == "different_target_sizes"))
        AGGREGATION_VARS = []  # ["all"]  #["episode", "movement"]  #["episode", "targetoccurrence"]  #["targetoccurrence"] #["episode", "movement"]  #["episode", "radius", "movement", "target", "targetoccurrence"]

        # WHAT TO PLOT?
        PLOT_TRACKING_DISTANCE = False  # if True, plot distance between End-effector and target position instead of position (only reasonable for tracking tasks)
        PLOT_ENDEFFECTOR = False  # if True plot End-effector position and velocity, else plot qpos and qvel for joint with index JOINT_ID (see independent_joints below)
        #JOINT_ID = 2  # only used if PLOT_ENDEFFECTOR == False
        PLOT_DEVIATION = False  # only if PLOT_ENDEFFECTOR == True

        # HOW TO PLOT?
        NORMALIZE_TIME = False
        PLOT_TIME_SERIES = True  # if True plot Position/Velocity/Acceleration Time Series, else plot Phasespace and Hooke plots
        PLOT_VEL_ACC = False  # if True, plot Velocity and Acceleration Time Series, else plot Position and Velocity Time Series (only used if PLOT_TIME_SERIES == True)
        PLOT_RANGES = False
        CONF_LEVEL = "min/max"  # might be between 0 and 1, or "min/max"; only used if PLOT_RANGES==True

        # WHICH BASELINE?
        SHOW_MINJERK = False
        SHOW_STUDY = True
        STUDY_ONLY = False  # only used if PLOTTING_ENV == "MPC-taskconditions"

        # PLOT (WHICH) LEGENDS AND COLORBARS?
        #ENABLE_LEGENDS_AND_COLORBARS = True  # if False, legends (of axis 0) and colobars are removed
        ALLOW_DUPLICATES_BETWEEN_LEGENDS = False  # if False, legend of axis 1 only contains values not included in legend of axis 0

        # STORE PLOT?
        STORE_PLOT = True
        STORE_AXES_SEPARATELY = True  # if True, store left and right axis to separate figures

        ####

        for JOINT_ID in range(7):
            trajectoryplot(PLOTTING_ENV, USER_ID, TASK_CONDITION,
                           common_simulation_subdir, filename, trajectories_SIMULATION,
                           trajectories_STUDY=trajectories_STUDY,
                           trajectories_SUPPLEMENTARY=trajectories_SUPPLEMENTARY,
                           REPEATED_MOVEMENTS=REPEATED_MOVEMENTS,
                           USER_ID_FIXED=USER_ID_FIXED,
                           ignore_trainingset_trials_mpc_userstudy=ignore_trainingset_trials_mpc_userstudy,
                           MOVEMENT_IDS=MOVEMENT_IDS,
                           RADIUS_IDS=RADIUS_IDS,
                           EPISODE_IDS=EPISODE_IDS,
                           r1_FIXED=r1_FIXED,
                           r2_FIXED=r2_FIXED,
                           EFFECTIVE_PROJECTION_PATH=EFFECTIVE_PROJECTION_PATH,
                           USE_TARGETBOUND_AS_DIST=USE_TARGETBOUND_AS_DIST,
                           MINJERK_USER_CONSTRAINTS=MINJERK_USER_CONSTRAINTS,
                           TARGET_IDS=TARGET_IDS,
                           TRIAL_IDS=TRIAL_IDS,
                           META_IDS=META_IDS,
                           N_MOVS=N_MOVS,
                           AGGREGATION_VARS=AGGREGATION_VARS,
                           PLOT_TRACKING_DISTANCE=PLOT_TRACKING_DISTANCE,
                           PLOT_ENDEFFECTOR=PLOT_ENDEFFECTOR,
                           JOINT_ID=JOINT_ID,
                           PLOT_DEVIATION=PLOT_DEVIATION,
                           NORMALIZE_TIME=NORMALIZE_TIME,
                           # DWELL_TIME=#DWELL_TIME,
                           PLOT_TIME_SERIES=PLOT_TIME_SERIES,
                           PLOT_VEL_ACC=PLOT_VEL_ACC,
                           PLOT_RANGES=PLOT_RANGES,
                           CONF_LEVEL=CONF_LEVEL,
                           SHOW_MINJERK=SHOW_MINJERK,
                           SHOW_STUDY=SHOW_STUDY,
                           STUDY_ONLY=STUDY_ONLY,
                           ENABLE_LEGENDS_AND_COLORBARS=JOINT_ID==2,
                           ALLOW_DUPLICATES_BETWEEN_LEGENDS=ALLOW_DUPLICATES_BETWEEN_LEGENDS,
                           STORE_PLOT=STORE_PLOT,
                           STORE_AXES_SEPARATELY=STORE_AXES_SEPARATELY)

    ########################################

    if 10 in _active_parts:
        # Figure B.7
        _subtitle = """PART 10: MPC - Simulation vs. User comparisons of single cost function"""
        print(f"\n\n+++++++++++++++++++++++++++++++++++++\n{_subtitle}\n++++++++++++++++++++++++++++++++++++++\n")

        USER_ID_LIST = [f"U{i}" for i in range(1, 7)]
        USER_ID_FIXED = "U2"  # (only used if SHOW_STUDY == True)

        SIMULATION_SUBDIR = "JAC"

        TASK_CONDITION = "Virtual_Cursor_ID_ISO_15_plane"

        filename = ""

        # AGGREGATE_TRIALS = True
        #########################

        REPEATED_MOVEMENTS = False

        trajectories_SIMULATION = TrajectoryData_MPC(DIRNAME_SIMULATION, SIMULATION_SUBDIR, USER_ID=USER_ID_FIXED,
                                                     TASK_CONDITION=TASK_CONDITION,
                                                     FILENAME_STUDY_TARGETPOSITIONS=FILENAME_STUDY_TARGETPOSITIONS)
        trajectories_SIMULATION.preprocess()  # AGGREGATE_TRIALS=AGGREGATE_TRIALS)

        trajectories_SUPPLEMENTARY = []
        for USER_ID in USER_ID_LIST:
            trajectories_SUPPLEMENTARY.append(
                TrajectoryData_STUDY(DIRNAME_STUDY, USER_ID=USER_ID, TASK_CONDITION=TASK_CONDITION,
                                     FILENAME_STUDY_TARGETPOSITIONS=FILENAME_STUDY_TARGETPOSITIONS))
            trajectories_SUPPLEMENTARY[-1].preprocess()  # AGGREGATE_TRIALS=AGGREGATE_TRIALS)

            # Preprocess experimental trajectories for each user (ISO Task User Study):
            if USER_ID == USER_ID_FIXED:
                trajectories_STUDY = trajectories_SUPPLEMENTARY[-1]

        #########################

        PLOTTING_ENV = "MPC-userstudy"

        ignore_trainingset_trials_mpc_userstudy = False  # will not ignore training trials here

        common_simulation_subdir = SIMULATION_SUBDIR

        #################################

        #################################
        ### SET PLOTTING PARAM VALUES ###
        #################################

        # WHICH PARTS OF DATASET? #only used if PLOTTING_ENV == "RL-UIB"
        MOVEMENT_IDS = None  # range(1,9) #[i for i in range(10) if i != 1]
        RADIUS_IDS = None
        EPISODE_IDS = [7]

        r1_FIXED = None  # r1list[5]  #only used if PLOTTING_ENV == "MPC-costweights"
        r2_FIXED = None  # r2list[-1]  #only used if PLOTTING_ENV == "MPC-costweights"

        # TASK_CONDITION_LIST_SELECTED = ["Virtual_Cursor_ID_ISO_15_plane", "Virtual_Cursor_Ergonomic_ISO_15_plane"]  #only used if PLOTTING_ENV == "MPC-taskconditions"

        # WHAT TO COMPUTE?
        EFFECTIVE_PROJECTION_PATH = (
                    PLOTTING_ENV == "RL-UIB")  # if True, projection path connects effective initial and final position instead of nominal target center positions
        USE_TARGETBOUND_AS_DIST = False  # True/False or "MinJerk-only"; if True, only plot trajectory until targetboundary is reached first (i.e., until dwell time begins); if "MinJerk-only", complete simulation trajectories are shown, but MinJerk trajectories are aligned to simulation trajectories without dwell time
        MINJERK_USER_CONSTRAINTS = True

        # WHICH/HOW MANY MOVS?
        """
        IMPORTANT INFO:
        if isinstance(trajectories, TrajectoryData_RL):
            -> TRIAL_IDS/META_IDS/N_MOVS are (meta-)indices of respective episode (or rather, of respective row of trajectories.indices)
            -> TRIAL_IDS and META_IDS are equivalent
        if isinstance(trajectories, TrajectoryData_STUDY) or isinstance(trajectories, TrajectoryData_MPC):
            -> TRIAL_IDS/META_IDS/N_MOVS are (global) indices of entire dataset
            -> TRIAL_IDS correspond to trial indices assigned during user study (last column of trajectories.indices), while META_IDS correspond to (meta-)indices of trajectories.indices itself (i.e., if some trials were removed during previous pre-processing steps, TRIAL_IDS and META_IDS differ!)
        In general, only the first not-None parameter is used, in following order: TRIAL_IDS, META_IDS, N_MOVS.
        """
        TARGET_IDS = None  # corresponds to target IDs (if PLOTTING_ENV == "RL-UIB": only for iso-task)
        TRIAL_IDS = [
            21]  # [i for i in range(65) if i not in range(28, 33)] #range(1,4) #corresponds to trial IDs [last column of self.indices] or relative (meta) index per row in self.indices; either a list of indices, "different_target_sizes" (choose N_MOVS conditions with maximum differences in target size), or None (use META_IDS)
        META_IDS = None  # index positions (i.e., sequential numbering of trials [aggregated trials, if AGGREGATE_TRIALS==True] in indices, without counting removed outliers); if None: use N_MOVS
        N_MOVS = None  # number of movements to visualize (only used, if TRIAL_IDS and META_IDS are both None (or TRIAL_IDS == "different_target_sizes"))
        AGGREGATION_VARS = []  # ["all"]  #["episode", "movement"]  #["episode", "targetoccurrence"]  #["targetoccurrence"] #["episode", "movement"]  #["episode", "radius", "movement", "target", "targetoccurrence"]

        # WHAT TO PLOT?
        PLOT_TRACKING_DISTANCE = False  # if True, plot distance between End-effector and target position instead of position (only reasonable for tracking tasks)
        PLOT_ENDEFFECTOR = True  # if True plot End-effector position and velocity, else plot qpos and qvel for joint with index JOINT_ID (see independent_joints below)
        JOINT_ID = 2  # only used if PLOT_ENDEFFECTOR == False
        PLOT_DEVIATION = False  # only if PLOT_ENDEFFECTOR == True

        # HOW TO PLOT?
        NORMALIZE_TIME = False
        PLOT_TIME_SERIES = True  # if True plot Position/Velocity/Acceleration Time Series, else plot Phasespace and Hooke plots
        PLOT_VEL_ACC = False  # if True, plot Velocity and Acceleration Time Series, else plot Position and Velocity Time Series (only used if PLOT_TIME_SERIES == True)
        PLOT_RANGES = False
        CONF_LEVEL = "min/max"  # might be between 0 and 1, or "min/max"; only used if PLOT_RANGES==True

        # WHICH BASELINE?
        SHOW_MINJERK = False
        SHOW_STUDY = True
        STUDY_ONLY = False  # only used if PLOTTING_ENV == "MPC-taskconditions"

        # PLOT (WHICH) LEGENDS AND COLORBARS?
        ENABLE_LEGENDS_AND_COLORBARS = True  # if False, legends (of axis 0) and colobars are removed
        ALLOW_DUPLICATES_BETWEEN_LEGENDS = False  # if False, legend of axis 1 only contains values not included in legend of axis 0

        # STORE PLOT?
        STORE_PLOT = True
        STORE_AXES_SEPARATELY = True  # if True, store left and right axis to separate figures

        ####

        trajectoryplot(PLOTTING_ENV, USER_ID, TASK_CONDITION,
                       common_simulation_subdir, filename, trajectories_SIMULATION,
                       trajectories_STUDY=trajectories_STUDY,
                       trajectories_SUPPLEMENTARY=trajectories_SUPPLEMENTARY,
                       REPEATED_MOVEMENTS=REPEATED_MOVEMENTS,
                       USER_ID_FIXED=USER_ID_FIXED,
                       ignore_trainingset_trials_mpc_userstudy=ignore_trainingset_trials_mpc_userstudy,
                       MOVEMENT_IDS=MOVEMENT_IDS,
                       RADIUS_IDS=RADIUS_IDS,
                       EPISODE_IDS=EPISODE_IDS,
                       r1_FIXED=r1_FIXED,
                       r2_FIXED=r2_FIXED,
                       EFFECTIVE_PROJECTION_PATH=EFFECTIVE_PROJECTION_PATH,
                       USE_TARGETBOUND_AS_DIST=USE_TARGETBOUND_AS_DIST,
                       MINJERK_USER_CONSTRAINTS=MINJERK_USER_CONSTRAINTS,
                       TARGET_IDS=TARGET_IDS,
                       TRIAL_IDS=TRIAL_IDS,
                       META_IDS=META_IDS,
                       N_MOVS=N_MOVS,
                       AGGREGATION_VARS=AGGREGATION_VARS,
                       PLOT_TRACKING_DISTANCE=PLOT_TRACKING_DISTANCE,
                       PLOT_ENDEFFECTOR=PLOT_ENDEFFECTOR,
                       JOINT_ID=JOINT_ID,
                       PLOT_DEVIATION=PLOT_DEVIATION,
                       NORMALIZE_TIME=NORMALIZE_TIME,
                       # DWELL_TIME=#DWELL_TIME,
                       PLOT_TIME_SERIES=PLOT_TIME_SERIES,
                       PLOT_VEL_ACC=PLOT_VEL_ACC,
                       PLOT_RANGES=PLOT_RANGES,
                       CONF_LEVEL=CONF_LEVEL,
                       SHOW_MINJERK=SHOW_MINJERK,
                       SHOW_STUDY=SHOW_STUDY,
                       STUDY_ONLY=STUDY_ONLY,
                       ENABLE_LEGENDS_AND_COLORBARS=ENABLE_LEGENDS_AND_COLORBARS,
                       ALLOW_DUPLICATES_BETWEEN_LEGENDS=ALLOW_DUPLICATES_BETWEEN_LEGENDS,
                       STORE_PLOT=STORE_PLOT,
                       STORE_AXES_SEPARATELY=STORE_AXES_SEPARATELY)

        PLOT_ENDEFFECTOR = False
        ENABLE_LEGENDS_AND_COLORBARS = False
        for JOINT_ID in range(7):
            trajectoryplot(PLOTTING_ENV, USER_ID, TASK_CONDITION,
                           common_simulation_subdir, filename, trajectories_SIMULATION,
                           trajectories_STUDY=trajectories_STUDY,
                           trajectories_SUPPLEMENTARY=trajectories_SUPPLEMENTARY,
                           REPEATED_MOVEMENTS=REPEATED_MOVEMENTS,
                           USER_ID_FIXED=USER_ID_FIXED,
                           ignore_trainingset_trials_mpc_userstudy=ignore_trainingset_trials_mpc_userstudy,
                           MOVEMENT_IDS=MOVEMENT_IDS,
                           RADIUS_IDS=RADIUS_IDS,
                           EPISODE_IDS=EPISODE_IDS,
                           r1_FIXED=r1_FIXED,
                           r2_FIXED=r2_FIXED,
                           EFFECTIVE_PROJECTION_PATH=EFFECTIVE_PROJECTION_PATH,
                           USE_TARGETBOUND_AS_DIST=USE_TARGETBOUND_AS_DIST,
                           MINJERK_USER_CONSTRAINTS=MINJERK_USER_CONSTRAINTS,
                           TARGET_IDS=TARGET_IDS,
                           TRIAL_IDS=TRIAL_IDS,
                           META_IDS=META_IDS,
                           N_MOVS=N_MOVS,
                           AGGREGATION_VARS=AGGREGATION_VARS,
                           PLOT_TRACKING_DISTANCE=PLOT_TRACKING_DISTANCE,
                           PLOT_ENDEFFECTOR=PLOT_ENDEFFECTOR,
                           JOINT_ID=JOINT_ID,
                           PLOT_DEVIATION=PLOT_DEVIATION,
                           NORMALIZE_TIME=NORMALIZE_TIME,
                           # DWELL_TIME=#DWELL_TIME,
                           PLOT_TIME_SERIES=PLOT_TIME_SERIES,
                           PLOT_VEL_ACC=PLOT_VEL_ACC,
                           PLOT_RANGES=PLOT_RANGES,
                           CONF_LEVEL=CONF_LEVEL,
                           SHOW_MINJERK=SHOW_MINJERK,
                           SHOW_STUDY=SHOW_STUDY,
                           STUDY_ONLY=STUDY_ONLY,
                           ENABLE_LEGENDS_AND_COLORBARS=ENABLE_LEGENDS_AND_COLORBARS,
                           ALLOW_DUPLICATES_BETWEEN_LEGENDS=ALLOW_DUPLICATES_BETWEEN_LEGENDS,
                           STORE_PLOT=STORE_PLOT,
                           STORE_AXES_SEPARATELY=STORE_AXES_SEPARATELY)

    ########################################

    if 11 in _active_parts:
        # Figure B.8
        _subtitle = """PART 11: MPC - Simulation vs. User comparisons of single cost function"""
        print(f"\n\n+++++++++++++++++++++++++++++++++++++\n{_subtitle}\n++++++++++++++++++++++++++++++++++++++\n")

        USER_ID_LIST = [f"U{i}" for i in range(1, 7)]
        USER_ID_FIXED = "U3"  # (only used if SHOW_STUDY == True)

        SIMULATION_SUBDIR = "JAC"

        TASK_CONDITION = "Virtual_Cursor_Ergonomic_ISO_15_plane"

        filename = ""

        # AGGREGATE_TRIALS = True
        #########################

        REPEATED_MOVEMENTS = False

        trajectories_SIMULATION = TrajectoryData_MPC(DIRNAME_SIMULATION, SIMULATION_SUBDIR, USER_ID=USER_ID_FIXED,
                                                     TASK_CONDITION=TASK_CONDITION,
                                                     FILENAME_STUDY_TARGETPOSITIONS=FILENAME_STUDY_TARGETPOSITIONS)
        trajectories_SIMULATION.preprocess()  # AGGREGATE_TRIALS=AGGREGATE_TRIALS)

        trajectories_SUPPLEMENTARY = []
        for USER_ID in USER_ID_LIST:
            trajectories_SUPPLEMENTARY.append(
                TrajectoryData_STUDY(DIRNAME_STUDY, USER_ID=USER_ID, TASK_CONDITION=TASK_CONDITION,
                                     FILENAME_STUDY_TARGETPOSITIONS=FILENAME_STUDY_TARGETPOSITIONS))
            trajectories_SUPPLEMENTARY[-1].preprocess()  # AGGREGATE_TRIALS=AGGREGATE_TRIALS)

            # Preprocess experimental trajectories for each user (ISO Task User Study):
            if USER_ID == USER_ID_FIXED:
                trajectories_STUDY = trajectories_SUPPLEMENTARY[-1]

        #########################

        PLOTTING_ENV = "MPC-userstudy"

        ignore_trainingset_trials_mpc_userstudy = False  # will not ignore training trials here

        common_simulation_subdir = SIMULATION_SUBDIR

        #################################

        #################################
        ### SET PLOTTING PARAM VALUES ###
        #################################

        # WHICH PARTS OF DATASET? #only used if PLOTTING_ENV == "RL-UIB"
        MOVEMENT_IDS = None  # range(1,9) #[i for i in range(10) if i != 1]
        RADIUS_IDS = None
        EPISODE_IDS = [7]

        r1_FIXED = None  # r1list[5]  #only used if PLOTTING_ENV == "MPC-costweights"
        r2_FIXED = None  # r2list[-1]  #only used if PLOTTING_ENV == "MPC-costweights"

        # TASK_CONDITION_LIST_SELECTED = ["Virtual_Cursor_ID_ISO_15_plane", "Virtual_Cursor_Ergonomic_ISO_15_plane"]  #only used if PLOTTING_ENV == "MPC-taskconditions"

        # WHAT TO COMPUTE?
        EFFECTIVE_PROJECTION_PATH = (
                    PLOTTING_ENV == "RL-UIB")  # if True, projection path connects effective initial and final position instead of nominal target center positions
        USE_TARGETBOUND_AS_DIST = False  # True/False or "MinJerk-only"; if True, only plot trajectory until targetboundary is reached first (i.e., until dwell time begins); if "MinJerk-only", complete simulation trajectories are shown, but MinJerk trajectories are aligned to simulation trajectories without dwell time
        MINJERK_USER_CONSTRAINTS = True

        # WHICH/HOW MANY MOVS?
        """
        IMPORTANT INFO:
        if isinstance(trajectories, TrajectoryData_RL):
            -> TRIAL_IDS/META_IDS/N_MOVS are (meta-)indices of respective episode (or rather, of respective row of trajectories.indices)
            -> TRIAL_IDS and META_IDS are equivalent
        if isinstance(trajectories, TrajectoryData_STUDY) or isinstance(trajectories, TrajectoryData_MPC):
            -> TRIAL_IDS/META_IDS/N_MOVS are (global) indices of entire dataset
            -> TRIAL_IDS correspond to trial indices assigned during user study (last column of trajectories.indices), while META_IDS correspond to (meta-)indices of trajectories.indices itself (i.e., if some trials were removed during previous pre-processing steps, TRIAL_IDS and META_IDS differ!)
        In general, only the first not-None parameter is used, in following order: TRIAL_IDS, META_IDS, N_MOVS.
        """
        TARGET_IDS = None  # corresponds to target IDs (if PLOTTING_ENV == "RL-UIB": only for iso-task)
        TRIAL_IDS = [
            47]  # [i for i in range(65) if i not in range(28, 33)] #range(1,4) #corresponds to trial IDs [last column of self.indices] or relative (meta) index per row in self.indices; either a list of indices, "different_target_sizes" (choose N_MOVS conditions with maximum differences in target size), or None (use META_IDS)
        META_IDS = None  # index positions (i.e., sequential numbering of trials [aggregated trials, if AGGREGATE_TRIALS==True] in indices, without counting removed outliers); if None: use N_MOVS
        N_MOVS = None  # number of movements to visualize (only used, if TRIAL_IDS and META_IDS are both None (or TRIAL_IDS == "different_target_sizes"))
        AGGREGATION_VARS = []  # ["all"]  #["episode", "movement"]  #["episode", "targetoccurrence"]  #["targetoccurrence"] #["episode", "movement"]  #["episode", "radius", "movement", "target", "targetoccurrence"]

        # WHAT TO PLOT?
        PLOT_TRACKING_DISTANCE = False  # if True, plot distance between End-effector and target position instead of position (only reasonable for tracking tasks)
        PLOT_ENDEFFECTOR = True  # if True plot End-effector position and velocity, else plot qpos and qvel for joint with index JOINT_ID (see independent_joints below)
        JOINT_ID = 2  # only used if PLOT_ENDEFFECTOR == False
        PLOT_DEVIATION = False  # only if PLOT_ENDEFFECTOR == True

        # HOW TO PLOT?
        NORMALIZE_TIME = False
        PLOT_TIME_SERIES = True  # if True plot Position/Velocity/Acceleration Time Series, else plot Phasespace and Hooke plots
        PLOT_VEL_ACC = False  # if True, plot Velocity and Acceleration Time Series, else plot Position and Velocity Time Series (only used if PLOT_TIME_SERIES == True)
        PLOT_RANGES = False
        CONF_LEVEL = "min/max"  # might be between 0 and 1, or "min/max"; only used if PLOT_RANGES==True

        # WHICH BASELINE?
        SHOW_MINJERK = False
        SHOW_STUDY = True
        STUDY_ONLY = False  # only used if PLOTTING_ENV == "MPC-taskconditions"

        # PLOT (WHICH) LEGENDS AND COLORBARS?
        ENABLE_LEGENDS_AND_COLORBARS = True  # if False, legends (of axis 0) and colobars are removed
        ALLOW_DUPLICATES_BETWEEN_LEGENDS = False  # if False, legend of axis 1 only contains values not included in legend of axis 0

        # STORE PLOT?
        STORE_PLOT = True
        STORE_AXES_SEPARATELY = True  # if True, store left and right axis to separate figures

        ####

        trajectoryplot(PLOTTING_ENV, USER_ID, TASK_CONDITION,
                       common_simulation_subdir, filename, trajectories_SIMULATION,
                       trajectories_STUDY=trajectories_STUDY,
                       trajectories_SUPPLEMENTARY=trajectories_SUPPLEMENTARY,
                       REPEATED_MOVEMENTS=REPEATED_MOVEMENTS,
                       USER_ID_FIXED=USER_ID_FIXED,
                       ignore_trainingset_trials_mpc_userstudy=ignore_trainingset_trials_mpc_userstudy,
                       MOVEMENT_IDS=MOVEMENT_IDS,
                       RADIUS_IDS=RADIUS_IDS,
                       EPISODE_IDS=EPISODE_IDS,
                       r1_FIXED=r1_FIXED,
                       r2_FIXED=r2_FIXED,
                       EFFECTIVE_PROJECTION_PATH=EFFECTIVE_PROJECTION_PATH,
                       USE_TARGETBOUND_AS_DIST=USE_TARGETBOUND_AS_DIST,
                       MINJERK_USER_CONSTRAINTS=MINJERK_USER_CONSTRAINTS,
                       TARGET_IDS=TARGET_IDS,
                       TRIAL_IDS=TRIAL_IDS,
                       META_IDS=META_IDS,
                       N_MOVS=N_MOVS,
                       AGGREGATION_VARS=AGGREGATION_VARS,
                       PLOT_TRACKING_DISTANCE=PLOT_TRACKING_DISTANCE,
                       PLOT_ENDEFFECTOR=PLOT_ENDEFFECTOR,
                       JOINT_ID=JOINT_ID,
                       PLOT_DEVIATION=PLOT_DEVIATION,
                       NORMALIZE_TIME=NORMALIZE_TIME,
                       # DWELL_TIME=#DWELL_TIME,
                       PLOT_TIME_SERIES=PLOT_TIME_SERIES,
                       PLOT_VEL_ACC=PLOT_VEL_ACC,
                       PLOT_RANGES=PLOT_RANGES,
                       CONF_LEVEL=CONF_LEVEL,
                       SHOW_MINJERK=SHOW_MINJERK,
                       SHOW_STUDY=SHOW_STUDY,
                       STUDY_ONLY=STUDY_ONLY,
                       ENABLE_LEGENDS_AND_COLORBARS=ENABLE_LEGENDS_AND_COLORBARS,
                       ALLOW_DUPLICATES_BETWEEN_LEGENDS=ALLOW_DUPLICATES_BETWEEN_LEGENDS,
                       STORE_PLOT=STORE_PLOT,
                       STORE_AXES_SEPARATELY=STORE_AXES_SEPARATELY)

        PLOT_ENDEFFECTOR = False
        ENABLE_LEGENDS_AND_COLORBARS = False
        for JOINT_ID in range(7):
            trajectoryplot(PLOTTING_ENV, USER_ID, TASK_CONDITION,
                           common_simulation_subdir, filename, trajectories_SIMULATION,
                           trajectories_STUDY=trajectories_STUDY,
                           trajectories_SUPPLEMENTARY=trajectories_SUPPLEMENTARY,
                           REPEATED_MOVEMENTS=REPEATED_MOVEMENTS,
                           USER_ID_FIXED=USER_ID_FIXED,
                           ignore_trainingset_trials_mpc_userstudy=ignore_trainingset_trials_mpc_userstudy,
                           MOVEMENT_IDS=MOVEMENT_IDS,
                           RADIUS_IDS=RADIUS_IDS,
                           EPISODE_IDS=EPISODE_IDS,
                           r1_FIXED=r1_FIXED,
                           r2_FIXED=r2_FIXED,
                           EFFECTIVE_PROJECTION_PATH=EFFECTIVE_PROJECTION_PATH,
                           USE_TARGETBOUND_AS_DIST=USE_TARGETBOUND_AS_DIST,
                           MINJERK_USER_CONSTRAINTS=MINJERK_USER_CONSTRAINTS,
                           TARGET_IDS=TARGET_IDS,
                           TRIAL_IDS=TRIAL_IDS,
                           META_IDS=META_IDS,
                           N_MOVS=N_MOVS,
                           AGGREGATION_VARS=AGGREGATION_VARS,
                           PLOT_TRACKING_DISTANCE=PLOT_TRACKING_DISTANCE,
                           PLOT_ENDEFFECTOR=PLOT_ENDEFFECTOR,
                           JOINT_ID=JOINT_ID,
                           PLOT_DEVIATION=PLOT_DEVIATION,
                           NORMALIZE_TIME=NORMALIZE_TIME,
                           # DWELL_TIME=#DWELL_TIME,
                           PLOT_TIME_SERIES=PLOT_TIME_SERIES,
                           PLOT_VEL_ACC=PLOT_VEL_ACC,
                           PLOT_RANGES=PLOT_RANGES,
                           CONF_LEVEL=CONF_LEVEL,
                           SHOW_MINJERK=SHOW_MINJERK,
                           SHOW_STUDY=SHOW_STUDY,
                           STUDY_ONLY=STUDY_ONLY,
                           ENABLE_LEGENDS_AND_COLORBARS=ENABLE_LEGENDS_AND_COLORBARS,
                           ALLOW_DUPLICATES_BETWEEN_LEGENDS=ALLOW_DUPLICATES_BETWEEN_LEGENDS,
                           STORE_PLOT=STORE_PLOT,
                           STORE_AXES_SEPARATELY=STORE_AXES_SEPARATELY)

    ########################################

    if 12 in _active_parts:
        # Figure B.9
        _subtitle = """PART 12: MPC - Simulation vs. User comparisons of single cost function"""
        print(f"\n\n+++++++++++++++++++++++++++++++++++++\n{_subtitle}\n++++++++++++++++++++++++++++++++++++++\n")

        USER_ID_LIST = [f"U{i}" for i in range(1, 7)]
        USER_ID_FIXED = "U4"  # (only used if SHOW_STUDY == True)

        SIMULATION_SUBDIR = "JAC"

        TASK_CONDITION = "Virtual_Pad_ID_ISO_15_plane"

        filename = ""

        # AGGREGATE_TRIALS = True
        #########################

        REPEATED_MOVEMENTS = False

        trajectories_SIMULATION = TrajectoryData_MPC(DIRNAME_SIMULATION, SIMULATION_SUBDIR, USER_ID=USER_ID_FIXED,
                                                     TASK_CONDITION=TASK_CONDITION,
                                                     FILENAME_STUDY_TARGETPOSITIONS=FILENAME_STUDY_TARGETPOSITIONS)
        trajectories_SIMULATION.preprocess()  # AGGREGATE_TRIALS=AGGREGATE_TRIALS)

        trajectories_SUPPLEMENTARY = []
        for USER_ID in USER_ID_LIST:
            trajectories_SUPPLEMENTARY.append(
                TrajectoryData_STUDY(DIRNAME_STUDY, USER_ID=USER_ID, TASK_CONDITION=TASK_CONDITION,
                                     FILENAME_STUDY_TARGETPOSITIONS=FILENAME_STUDY_TARGETPOSITIONS))
            trajectories_SUPPLEMENTARY[-1].preprocess()  # AGGREGATE_TRIALS=AGGREGATE_TRIALS)

            # Preprocess experimental trajectories for each user (ISO Task User Study):
            if USER_ID == USER_ID_FIXED:
                trajectories_STUDY = trajectories_SUPPLEMENTARY[-1]

        #########################

        PLOTTING_ENV = "MPC-userstudy"

        ignore_trainingset_trials_mpc_userstudy = False  # will not ignore training trials here

        common_simulation_subdir = SIMULATION_SUBDIR

        #################################

        #################################
        ### SET PLOTTING PARAM VALUES ###
        #################################

        # WHICH PARTS OF DATASET? #only used if PLOTTING_ENV == "RL-UIB"
        MOVEMENT_IDS = None  # range(1,9) #[i for i in range(10) if i != 1]
        RADIUS_IDS = None
        EPISODE_IDS = [7]

        r1_FIXED = None  # r1list[5]  #only used if PLOTTING_ENV == "MPC-costweights"
        r2_FIXED = None  # r2list[-1]  #only used if PLOTTING_ENV == "MPC-costweights"

        # TASK_CONDITION_LIST_SELECTED = ["Virtual_Cursor_ID_ISO_15_plane", "Virtual_Cursor_Ergonomic_ISO_15_plane"]  #only used if PLOTTING_ENV == "MPC-taskconditions"

        # WHAT TO COMPUTE?
        EFFECTIVE_PROJECTION_PATH = (
                    PLOTTING_ENV == "RL-UIB")  # if True, projection path connects effective initial and final position instead of nominal target center positions
        USE_TARGETBOUND_AS_DIST = False  # True/False or "MinJerk-only"; if True, only plot trajectory until targetboundary is reached first (i.e., until dwell time begins); if "MinJerk-only", complete simulation trajectories are shown, but MinJerk trajectories are aligned to simulation trajectories without dwell time
        MINJERK_USER_CONSTRAINTS = True

        # WHICH/HOW MANY MOVS?
        """
        IMPORTANT INFO:
        if isinstance(trajectories, TrajectoryData_RL):
            -> TRIAL_IDS/META_IDS/N_MOVS are (meta-)indices of respective episode (or rather, of respective row of trajectories.indices)
            -> TRIAL_IDS and META_IDS are equivalent
        if isinstance(trajectories, TrajectoryData_STUDY) or isinstance(trajectories, TrajectoryData_MPC):
            -> TRIAL_IDS/META_IDS/N_MOVS are (global) indices of entire dataset
            -> TRIAL_IDS correspond to trial indices assigned during user study (last column of trajectories.indices), while META_IDS correspond to (meta-)indices of trajectories.indices itself (i.e., if some trials were removed during previous pre-processing steps, TRIAL_IDS and META_IDS differ!)
        In general, only the first not-None parameter is used, in following order: TRIAL_IDS, META_IDS, N_MOVS.
        """
        TARGET_IDS = None  # corresponds to target IDs (if PLOTTING_ENV == "RL-UIB": only for iso-task)
        TRIAL_IDS = [
            21]  # [i for i in range(65) if i not in range(28, 33)] #range(1,4) #corresponds to trial IDs [last column of self.indices] or relative (meta) index per row in self.indices; either a list of indices, "different_target_sizes" (choose N_MOVS conditions with maximum differences in target size), or None (use META_IDS)
        META_IDS = None  # index positions (i.e., sequential numbering of trials [aggregated trials, if AGGREGATE_TRIALS==True] in indices, without counting removed outliers); if None: use N_MOVS
        N_MOVS = None  # number of movements to visualize (only used, if TRIAL_IDS and META_IDS are both None (or TRIAL_IDS == "different_target_sizes"))
        AGGREGATION_VARS = []  # ["all"]  #["episode", "movement"]  #["episode", "targetoccurrence"]  #["targetoccurrence"] #["episode", "movement"]  #["episode", "radius", "movement", "target", "targetoccurrence"]

        # WHAT TO PLOT?
        PLOT_TRACKING_DISTANCE = False  # if True, plot distance between End-effector and target position instead of position (only reasonable for tracking tasks)
        PLOT_ENDEFFECTOR = True  # if True plot End-effector position and velocity, else plot qpos and qvel for joint with index JOINT_ID (see independent_joints below)
        JOINT_ID = 2  # only used if PLOT_ENDEFFECTOR == False
        PLOT_DEVIATION = False  # only if PLOT_ENDEFFECTOR == True

        # HOW TO PLOT?
        NORMALIZE_TIME = False
        PLOT_TIME_SERIES = True  # if True plot Position/Velocity/Acceleration Time Series, else plot Phasespace and Hooke plots
        PLOT_VEL_ACC = False  # if True, plot Velocity and Acceleration Time Series, else plot Position and Velocity Time Series (only used if PLOT_TIME_SERIES == True)
        PLOT_RANGES = False
        CONF_LEVEL = "min/max"  # might be between 0 and 1, or "min/max"; only used if PLOT_RANGES==True

        # WHICH BASELINE?
        SHOW_MINJERK = False
        SHOW_STUDY = True
        STUDY_ONLY = False  # only used if PLOTTING_ENV == "MPC-taskconditions"

        # PLOT (WHICH) LEGENDS AND COLORBARS?
        ENABLE_LEGENDS_AND_COLORBARS = True  # if False, legends (of axis 0) and colobars are removed
        ALLOW_DUPLICATES_BETWEEN_LEGENDS = False  # if False, legend of axis 1 only contains values not included in legend of axis 0

        # STORE PLOT?
        STORE_PLOT = True
        STORE_AXES_SEPARATELY = True  # if True, store left and right axis to separate figures

        ####

        trajectoryplot(PLOTTING_ENV, USER_ID, TASK_CONDITION,
                       common_simulation_subdir, filename, trajectories_SIMULATION,
                       trajectories_STUDY=trajectories_STUDY,
                       trajectories_SUPPLEMENTARY=trajectories_SUPPLEMENTARY,
                       REPEATED_MOVEMENTS=REPEATED_MOVEMENTS,
                       USER_ID_FIXED=USER_ID_FIXED,
                       ignore_trainingset_trials_mpc_userstudy=ignore_trainingset_trials_mpc_userstudy,
                       MOVEMENT_IDS=MOVEMENT_IDS,
                       RADIUS_IDS=RADIUS_IDS,
                       EPISODE_IDS=EPISODE_IDS,
                       r1_FIXED=r1_FIXED,
                       r2_FIXED=r2_FIXED,
                       EFFECTIVE_PROJECTION_PATH=EFFECTIVE_PROJECTION_PATH,
                       USE_TARGETBOUND_AS_DIST=USE_TARGETBOUND_AS_DIST,
                       MINJERK_USER_CONSTRAINTS=MINJERK_USER_CONSTRAINTS,
                       TARGET_IDS=TARGET_IDS,
                       TRIAL_IDS=TRIAL_IDS,
                       META_IDS=META_IDS,
                       N_MOVS=N_MOVS,
                       AGGREGATION_VARS=AGGREGATION_VARS,
                       PLOT_TRACKING_DISTANCE=PLOT_TRACKING_DISTANCE,
                       PLOT_ENDEFFECTOR=PLOT_ENDEFFECTOR,
                       JOINT_ID=JOINT_ID,
                       PLOT_DEVIATION=PLOT_DEVIATION,
                       NORMALIZE_TIME=NORMALIZE_TIME,
                       # DWELL_TIME=#DWELL_TIME,
                       PLOT_TIME_SERIES=PLOT_TIME_SERIES,
                       PLOT_VEL_ACC=PLOT_VEL_ACC,
                       PLOT_RANGES=PLOT_RANGES,
                       CONF_LEVEL=CONF_LEVEL,
                       SHOW_MINJERK=SHOW_MINJERK,
                       SHOW_STUDY=SHOW_STUDY,
                       STUDY_ONLY=STUDY_ONLY,
                       ENABLE_LEGENDS_AND_COLORBARS=ENABLE_LEGENDS_AND_COLORBARS,
                       ALLOW_DUPLICATES_BETWEEN_LEGENDS=ALLOW_DUPLICATES_BETWEEN_LEGENDS,
                       STORE_PLOT=STORE_PLOT,
                       STORE_AXES_SEPARATELY=STORE_AXES_SEPARATELY)

        PLOT_ENDEFFECTOR = False
        ENABLE_LEGENDS_AND_COLORBARS = False
        for JOINT_ID in range(7):
            trajectoryplot(PLOTTING_ENV, USER_ID, TASK_CONDITION,
                           common_simulation_subdir, filename, trajectories_SIMULATION,
                           trajectories_STUDY=trajectories_STUDY,
                           trajectories_SUPPLEMENTARY=trajectories_SUPPLEMENTARY,
                           REPEATED_MOVEMENTS=REPEATED_MOVEMENTS,
                           USER_ID_FIXED=USER_ID_FIXED,
                           ignore_trainingset_trials_mpc_userstudy=ignore_trainingset_trials_mpc_userstudy,
                           MOVEMENT_IDS=MOVEMENT_IDS,
                           RADIUS_IDS=RADIUS_IDS,
                           EPISODE_IDS=EPISODE_IDS,
                           r1_FIXED=r1_FIXED,
                           r2_FIXED=r2_FIXED,
                           EFFECTIVE_PROJECTION_PATH=EFFECTIVE_PROJECTION_PATH,
                           USE_TARGETBOUND_AS_DIST=USE_TARGETBOUND_AS_DIST,
                           MINJERK_USER_CONSTRAINTS=MINJERK_USER_CONSTRAINTS,
                           TARGET_IDS=TARGET_IDS,
                           TRIAL_IDS=TRIAL_IDS,
                           META_IDS=META_IDS,
                           N_MOVS=N_MOVS,
                           AGGREGATION_VARS=AGGREGATION_VARS,
                           PLOT_TRACKING_DISTANCE=PLOT_TRACKING_DISTANCE,
                           PLOT_ENDEFFECTOR=PLOT_ENDEFFECTOR,
                           JOINT_ID=JOINT_ID,
                           PLOT_DEVIATION=PLOT_DEVIATION,
                           NORMALIZE_TIME=NORMALIZE_TIME,
                           # DWELL_TIME=#DWELL_TIME,
                           PLOT_TIME_SERIES=PLOT_TIME_SERIES,
                           PLOT_VEL_ACC=PLOT_VEL_ACC,
                           PLOT_RANGES=PLOT_RANGES,
                           CONF_LEVEL=CONF_LEVEL,
                           SHOW_MINJERK=SHOW_MINJERK,
                           SHOW_STUDY=SHOW_STUDY,
                           STUDY_ONLY=STUDY_ONLY,
                           ENABLE_LEGENDS_AND_COLORBARS=ENABLE_LEGENDS_AND_COLORBARS,
                           ALLOW_DUPLICATES_BETWEEN_LEGENDS=ALLOW_DUPLICATES_BETWEEN_LEGENDS,
                           STORE_PLOT=STORE_PLOT,
                           STORE_AXES_SEPARATELY=STORE_AXES_SEPARATELY)

    ########################################

    if 13 in _active_parts:
        # Figure B.10
        _subtitle = """PART 13: MPC - Simulation vs. User comparisons of single cost function"""
        print(f"\n\n+++++++++++++++++++++++++++++++++++++\n{_subtitle}\n++++++++++++++++++++++++++++++++++++++\n")

        USER_ID_LIST = [f"U{i}" for i in range(1, 7)]
        USER_ID_FIXED = "U6"  # (only used if SHOW_STUDY == True)

        SIMULATION_SUBDIR = "JAC"

        TASK_CONDITION = "Virtual_Pad_Ergonomic_ISO_15_plane"

        filename = ""

        # AGGREGATE_TRIALS = True
        #########################

        REPEATED_MOVEMENTS = False

        trajectories_SIMULATION = TrajectoryData_MPC(DIRNAME_SIMULATION, SIMULATION_SUBDIR, USER_ID=USER_ID_FIXED,
                                                     TASK_CONDITION=TASK_CONDITION,
                                                     FILENAME_STUDY_TARGETPOSITIONS=FILENAME_STUDY_TARGETPOSITIONS)
        trajectories_SIMULATION.preprocess()  # AGGREGATE_TRIALS=AGGREGATE_TRIALS)

        trajectories_SUPPLEMENTARY = []
        for USER_ID in USER_ID_LIST:
            trajectories_SUPPLEMENTARY.append(
                TrajectoryData_STUDY(DIRNAME_STUDY, USER_ID=USER_ID, TASK_CONDITION=TASK_CONDITION,
                                     FILENAME_STUDY_TARGETPOSITIONS=FILENAME_STUDY_TARGETPOSITIONS))
            trajectories_SUPPLEMENTARY[-1].preprocess()  # AGGREGATE_TRIALS=AGGREGATE_TRIALS)

            # Preprocess experimental trajectories for each user (ISO Task User Study):
            if USER_ID == USER_ID_FIXED:
                trajectories_STUDY = trajectories_SUPPLEMENTARY[-1]

        #########################

        PLOTTING_ENV = "MPC-userstudy"

        ignore_trainingset_trials_mpc_userstudy = False  # will not ignore training trials here

        common_simulation_subdir = SIMULATION_SUBDIR

        #################################

        #################################
        ### SET PLOTTING PARAM VALUES ###
        #################################

        # WHICH PARTS OF DATASET? #only used if PLOTTING_ENV == "RL-UIB"
        MOVEMENT_IDS = None  # range(1,9) #[i for i in range(10) if i != 1]
        RADIUS_IDS = None
        EPISODE_IDS = [7]

        r1_FIXED = None  # r1list[5]  #only used if PLOTTING_ENV == "MPC-costweights"
        r2_FIXED = None  # r2list[-1]  #only used if PLOTTING_ENV == "MPC-costweights"

        # TASK_CONDITION_LIST_SELECTED = ["Virtual_Cursor_ID_ISO_15_plane", "Virtual_Cursor_Ergonomic_ISO_15_plane"]  #only used if PLOTTING_ENV == "MPC-taskconditions"

        # WHAT TO COMPUTE?
        EFFECTIVE_PROJECTION_PATH = (
                    PLOTTING_ENV == "RL-UIB")  # if True, projection path connects effective initial and final position instead of nominal target center positions
        USE_TARGETBOUND_AS_DIST = False  # True/False or "MinJerk-only"; if True, only plot trajectory until targetboundary is reached first (i.e., until dwell time begins); if "MinJerk-only", complete simulation trajectories are shown, but MinJerk trajectories are aligned to simulation trajectories without dwell time
        MINJERK_USER_CONSTRAINTS = True

        # WHICH/HOW MANY MOVS?
        """
        IMPORTANT INFO:
        if isinstance(trajectories, TrajectoryData_RL):
            -> TRIAL_IDS/META_IDS/N_MOVS are (meta-)indices of respective episode (or rather, of respective row of trajectories.indices)
            -> TRIAL_IDS and META_IDS are equivalent
        if isinstance(trajectories, TrajectoryData_STUDY) or isinstance(trajectories, TrajectoryData_MPC):
            -> TRIAL_IDS/META_IDS/N_MOVS are (global) indices of entire dataset
            -> TRIAL_IDS correspond to trial indices assigned during user study (last column of trajectories.indices), while META_IDS correspond to (meta-)indices of trajectories.indices itself (i.e., if some trials were removed during previous pre-processing steps, TRIAL_IDS and META_IDS differ!)
        In general, only the first not-None parameter is used, in following order: TRIAL_IDS, META_IDS, N_MOVS.
        """
        TARGET_IDS = None  # corresponds to target IDs (if PLOTTING_ENV == "RL-UIB": only for iso-task)
        TRIAL_IDS = [
            8]  # [i for i in range(65) if i not in range(28, 33)] #range(1,4) #corresponds to trial IDs [last column of self.indices] or relative (meta) index per row in self.indices; either a list of indices, "different_target_sizes" (choose N_MOVS conditions with maximum differences in target size), or None (use META_IDS)
        META_IDS = None  # index positions (i.e., sequential numbering of trials [aggregated trials, if AGGREGATE_TRIALS==True] in indices, without counting removed outliers); if None: use N_MOVS
        N_MOVS = None  # number of movements to visualize (only used, if TRIAL_IDS and META_IDS are both None (or TRIAL_IDS == "different_target_sizes"))
        AGGREGATION_VARS = []  # ["all"]  #["episode", "movement"]  #["episode", "targetoccurrence"]  #["targetoccurrence"] #["episode", "movement"]  #["episode", "radius", "movement", "target", "targetoccurrence"]

        # WHAT TO PLOT?
        PLOT_TRACKING_DISTANCE = False  # if True, plot distance between End-effector and target position instead of position (only reasonable for tracking tasks)
        PLOT_ENDEFFECTOR = True  # if True plot End-effector position and velocity, else plot qpos and qvel for joint with index JOINT_ID (see independent_joints below)
        JOINT_ID = 2  # only used if PLOT_ENDEFFECTOR == False
        PLOT_DEVIATION = False  # only if PLOT_ENDEFFECTOR == True

        # HOW TO PLOT?
        NORMALIZE_TIME = False
        PLOT_TIME_SERIES = True  # if True plot Position/Velocity/Acceleration Time Series, else plot Phasespace and Hooke plots
        PLOT_VEL_ACC = False  # if True, plot Velocity and Acceleration Time Series, else plot Position and Velocity Time Series (only used if PLOT_TIME_SERIES == True)
        PLOT_RANGES = False
        CONF_LEVEL = "min/max"  # might be between 0 and 1, or "min/max"; only used if PLOT_RANGES==True

        # WHICH BASELINE?
        SHOW_MINJERK = False
        SHOW_STUDY = True
        STUDY_ONLY = False  # only used if PLOTTING_ENV == "MPC-taskconditions"

        # PLOT (WHICH) LEGENDS AND COLORBARS?
        ENABLE_LEGENDS_AND_COLORBARS = True  # if False, legends (of axis 0) and colobars are removed
        ALLOW_DUPLICATES_BETWEEN_LEGENDS = False  # if False, legend of axis 1 only contains values not included in legend of axis 0

        # STORE PLOT?
        STORE_PLOT = True
        STORE_AXES_SEPARATELY = True  # if True, store left and right axis to separate figures

        ####

        trajectoryplot(PLOTTING_ENV, USER_ID, TASK_CONDITION,
                       common_simulation_subdir, filename, trajectories_SIMULATION,
                       trajectories_STUDY=trajectories_STUDY,
                       trajectories_SUPPLEMENTARY=trajectories_SUPPLEMENTARY,
                       REPEATED_MOVEMENTS=REPEATED_MOVEMENTS,
                       USER_ID_FIXED=USER_ID_FIXED,
                       ignore_trainingset_trials_mpc_userstudy=ignore_trainingset_trials_mpc_userstudy,
                       MOVEMENT_IDS=MOVEMENT_IDS,
                       RADIUS_IDS=RADIUS_IDS,
                       EPISODE_IDS=EPISODE_IDS,
                       r1_FIXED=r1_FIXED,
                       r2_FIXED=r2_FIXED,
                       EFFECTIVE_PROJECTION_PATH=EFFECTIVE_PROJECTION_PATH,
                       USE_TARGETBOUND_AS_DIST=USE_TARGETBOUND_AS_DIST,
                       MINJERK_USER_CONSTRAINTS=MINJERK_USER_CONSTRAINTS,
                       TARGET_IDS=TARGET_IDS,
                       TRIAL_IDS=TRIAL_IDS,
                       META_IDS=META_IDS,
                       N_MOVS=N_MOVS,
                       AGGREGATION_VARS=AGGREGATION_VARS,
                       PLOT_TRACKING_DISTANCE=PLOT_TRACKING_DISTANCE,
                       PLOT_ENDEFFECTOR=PLOT_ENDEFFECTOR,
                       JOINT_ID=JOINT_ID,
                       PLOT_DEVIATION=PLOT_DEVIATION,
                       NORMALIZE_TIME=NORMALIZE_TIME,
                       # DWELL_TIME=#DWELL_TIME,
                       PLOT_TIME_SERIES=PLOT_TIME_SERIES,
                       PLOT_VEL_ACC=PLOT_VEL_ACC,
                       PLOT_RANGES=PLOT_RANGES,
                       CONF_LEVEL=CONF_LEVEL,
                       SHOW_MINJERK=SHOW_MINJERK,
                       SHOW_STUDY=SHOW_STUDY,
                       STUDY_ONLY=STUDY_ONLY,
                       ENABLE_LEGENDS_AND_COLORBARS=ENABLE_LEGENDS_AND_COLORBARS,
                       ALLOW_DUPLICATES_BETWEEN_LEGENDS=ALLOW_DUPLICATES_BETWEEN_LEGENDS,
                       STORE_PLOT=STORE_PLOT,
                       STORE_AXES_SEPARATELY=STORE_AXES_SEPARATELY)

        PLOT_ENDEFFECTOR = False
        ENABLE_LEGENDS_AND_COLORBARS = False
        for JOINT_ID in range(7):
            trajectoryplot(PLOTTING_ENV, USER_ID, TASK_CONDITION,
                           common_simulation_subdir, filename, trajectories_SIMULATION,
                           trajectories_STUDY=trajectories_STUDY,
                           trajectories_SUPPLEMENTARY=trajectories_SUPPLEMENTARY,
                           REPEATED_MOVEMENTS=REPEATED_MOVEMENTS,
                           USER_ID_FIXED=USER_ID_FIXED,
                           ignore_trainingset_trials_mpc_userstudy=ignore_trainingset_trials_mpc_userstudy,
                           MOVEMENT_IDS=MOVEMENT_IDS,
                           RADIUS_IDS=RADIUS_IDS,
                           EPISODE_IDS=EPISODE_IDS,
                           r1_FIXED=r1_FIXED,
                           r2_FIXED=r2_FIXED,
                           EFFECTIVE_PROJECTION_PATH=EFFECTIVE_PROJECTION_PATH,
                           USE_TARGETBOUND_AS_DIST=USE_TARGETBOUND_AS_DIST,
                           MINJERK_USER_CONSTRAINTS=MINJERK_USER_CONSTRAINTS,
                           TARGET_IDS=TARGET_IDS,
                           TRIAL_IDS=TRIAL_IDS,
                           META_IDS=META_IDS,
                           N_MOVS=N_MOVS,
                           AGGREGATION_VARS=AGGREGATION_VARS,
                           PLOT_TRACKING_DISTANCE=PLOT_TRACKING_DISTANCE,
                           PLOT_ENDEFFECTOR=PLOT_ENDEFFECTOR,
                           JOINT_ID=JOINT_ID,
                           PLOT_DEVIATION=PLOT_DEVIATION,
                           NORMALIZE_TIME=NORMALIZE_TIME,
                           # DWELL_TIME=#DWELL_TIME,
                           PLOT_TIME_SERIES=PLOT_TIME_SERIES,
                           PLOT_VEL_ACC=PLOT_VEL_ACC,
                           PLOT_RANGES=PLOT_RANGES,
                           CONF_LEVEL=CONF_LEVEL,
                           SHOW_MINJERK=SHOW_MINJERK,
                           SHOW_STUDY=SHOW_STUDY,
                           STUDY_ONLY=STUDY_ONLY,
                           ENABLE_LEGENDS_AND_COLORBARS=ENABLE_LEGENDS_AND_COLORBARS,
                           ALLOW_DUPLICATES_BETWEEN_LEGENDS=ALLOW_DUPLICATES_BETWEEN_LEGENDS,
                           STORE_PLOT=STORE_PLOT,
                           STORE_AXES_SEPARATELY=STORE_AXES_SEPARATELY)

    ########################################
