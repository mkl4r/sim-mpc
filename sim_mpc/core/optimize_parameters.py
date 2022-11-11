''' 
Class for parameter optimization using MPC simulation.

Authors: Markus Klar
Date: 11.2022
'''
from concurrent.futures import ProcessPoolExecutor as Pool
from scipy import interpolate
import os
import copy
from pathlib import Path
import logging as log
import pandas as pd
import numpy as np
import sim_mpc.scripts.iso_task as iso
import sim_mpc.core.cmaes as cmaes
from sim_mpc.core.simulator import Simulator



################################
# For parallel simulation
MAX_PROCESSES = 30

def worker_init(func):
    global _func
    _func = func


def worker(x):
    return _func(x)


def xmap(func, iterable, processes=MAX_PROCESSES):
    with Pool(processes, initializer=worker_init, initargs=(func,)) as p:
        return(p.map(worker, iterable))
#################################

# Change current working directory to file directory
os.chdir(Path(__file__).parent)
log.info(Path(__file__).parent)

DIRNAME_STUDY = "../data/study/"
DIRNAME_RESULTS = "../../results/"
DIRNAME_MODEL = "../data/models/"

log.basicConfig(level=log.ERROR, format=' %(asctime)s - %(levelname)s - %(message)s')

class ParamOptimizer():
    def __init__(self, participant, condition, outputprefix, costfct, eermse=False, debug=False, init_params=None, **kwargs):

        self.participant = participant
        self.condition = condition
        self.outputprefix = outputprefix

        # Define minimum successful movements
        self.minsuccessful = 4 #5 
        self.numevalmoves = 5 #6

        self.eermse = eermse
        if eermse:
            outputprefix += "_EE"

        self.init_params = init_params

        # Save Eval results
        self.save_eval = True

        # Create Simulator for the correct model
        self.simulator = Simulator(f"{DIRNAME_MODEL}/OriginExperiment_{self.participant}.xml", self.participant)

        # Pretermination
        self.simulator.param_dict["pretermination"] = True

        self.simulator.param_dict["display_steps"] = False
        self.simulator.param_dict["verbosity"] = False

        self.simulator.param_dict["max_iterations"] = 25
        if debug:
            self.simulator.param_dict["max_iterations"] = 1

        self.simulator.param_dict["num_steps"] = 30
        self.simulator.param_dict["N"] = 8
        if debug:
            self.simulator.param_dict["N"] = 2

        self.opt_single = False
        if costfct == "JAC":
            self.simulator.param_dict["acceleration_cost"] = True
            outputprefix += "_JAC"
        elif costfct == "CTC":
            self.simulator.param_dict["commanded_tc_cost"] = True
            outputprefix += "_ctc"
        elif costfct == "DC":
            self.opt_single = True
            self.simulator.param_dict["act_cost"] = "quad"
        else:
            log.error("Wrong cost function. Use DC, CTC or JAC.")
            raise NotImplementedError


        # Targets and init values
        # Get Initial Posture and Activation
        Xqpos, Xqvel = iso.get_init_postures(self.participant, self.condition)
        ACT, EXT = iso.get_act_dact(self.participant, self.condition)

        self.rotshoulder = iso.get_init_shoulder(self.participant, self.condition)

        numrepeats = 5

        targets, width = iso.get_iso_targets(self.rotshoulder)

        self.simulator.param_dict["iso_targets"] = True

        self.targets = targets * numrepeats
        self.width = width * numrepeats

        # Setup initial values for every target
        mtinfo = iso.get_submovement_times(participant, condition)
        self.usedXqpos = []
        self.usedXqvel = []
        self.usedACT = []
        self.usedEXT = []
        self.target_ids = []
        self.movementid = []
        self.lineids = []
        self.iterations = []

        # Choose targets out of available targets
        totalid = int((len(mtinfo)-1)//2)
        while len(self.target_ids) < self.numevalmoves:
            if mtinfo[totalid][3] not in self.target_ids:
                self.target_ids.append(int(mtinfo[totalid][3]))
                self.movementid.append(int(mtinfo[totalid][4]))
                self.usedXqpos.append(Xqpos[totalid])
                self.usedXqvel.append(Xqvel[totalid])
                self.usedACT.append(ACT[totalid])
                self.usedEXT.append(EXT[totalid])
                self.lineids.append(totalid)
                self.iterations.append(int((mtinfo[totalid][1]-mtinfo[totalid][0])//(self.simulator.param_dict["h"]*self.simulator.param_dict["num_steps"])))
            totalid -= 1

        # Setup folder
        self.outputfolder = "{}/{}/{}/{}".format(
            DIRNAME_RESULTS, participant, outputprefix, condition)
        Path(self.outputfolder).mkdir(parents=True, exist_ok=True)

        self.used_transfer = iso.get_used_transfer(self.condition, self.simulator, self.rotshoulder)

    def optimize(self):

        # DEFINE OUTPUT
        self.JJ = []
        self.XX = []
        self.out_columns_fit =  ["rmse"]
        self.out_columns_x = ["param_0"] if self.opt_single else ["param_0", "param_1"]

        # DEFINE INITIAL COSTPARAMS
        def cmaes_job(x, id, args):
            return self.eval_cost_parameters(x, id)
        
        # (CMAES) OPTIMIZATION
        log.debug("Starting parameter optimization with CMAES.")
        if self.opt_single:
            if self.init_params:
                initparamdf = pd.read_csv(self.init_params)
                if len(initparamdf.loc[initparamdf["participant"]==self.participant].loc[initparamdf["condition"]==self.condition].index) > 0:
                    xstart = np.log10(np.reshape(initparamdf.loc[initparamdf["participant"]==self.participant].loc[initparamdf["condition"]==self.condition, ["param_0"]].to_numpy(),(1,)))
                else:
                    xstart = np.array([-4.2211255279972604])
            else:
                xstart = np.array([-4.2211255279972604])
        else:
            if self.init_params:
                initparamdf = pd.read_csv(self.init_params)
                if len(initparamdf.loc[initparamdf["participant"]==self.participant].loc[initparamdf["condition"]==self.condition].index) > 0:
                    xstart = np.log10(np.reshape(initparamdf.loc[initparamdf["participant"]==self.participant].loc[initparamdf["condition"]==self.condition, ["param_0","param_1"]].to_numpy(),(2,)))
                else:
                    xstart = np.array([-1.898459127441651, -4.2211255279972604])
            else:
                xstart = np.array([-1.898459127441651, -4.2211255279972604])
        start_sigma = 1 
        optx = cmaes.cmaes(xstart, start_sigma, self.outputfolder, cmaes_job, {}, 30, max_iterations=1000) 
        log.debug(f"Optimization with CMAES finished. Best parameters: {optx}")
        np.save(self.outputfolder+ "/optx.npy", optx)
        
        return optx

    def change_cost_parameters(self, cost_params, exp):
        
        if exp:
            self.simulator.param_dict["r"] = 10.0**cost_params[0]
        else:
            self.simulator.param_dict["r"] = cost_params[0]

        if self.opt_single:
            if self.simulator.param_dict["commanded_tc_cost"] and exp:
                self.simulator.param_dict["commanded_tc_cost_weight"] = 10.0**cost_params[0]
        else:
            if exp:
                add_cost = 10.0**cost_params[1]
            else:
                add_cost = cost_params[1]
            if self.simulator.param_dict["acceleration_cost"]:
                self.simulator.param_dict["acceleration_cost_weight"] = add_cost
            elif self.simulator.param_dict["commanded_tc_cost"]:
                self.simulator.param_dict["commanded_tc_cost_weight"] = add_cost
                
    def eval_cost_parameters(self, costParam, id, exp=True):

        costParam = costParam.flatten()

        self.change_cost_parameters(costParam, exp)

        # Run simulations in parallel
        _func = None
        mapout = list(
            xmap(lambda j: self.sim_one_move(j, id), range(self.numevalmoves)))

        RMSE = [mapout[k][0] for k in range(self.numevalmoves)] 
        log.debug(RMSE)
        rmse = sum(RMSE)
        log.debug(f"Parameters {costParam}, rmse = {rmse}") 

        if self.save_eval:
            path=f"{self.outputfolder}/PO_result.csv"
            self.JJ.append(rmse)
            self.XX.append(costParam)
            dffit = pd.DataFrame(data=self.JJ, columns = self.out_columns_fit)
            dfx = pd.DataFrame(data=self.XX, columns = self.out_columns_x)
            df = pd.concat([dffit, dfx], axis=1)
            df.to_csv(path, index=False)


        return rmse

    def sim_one_move(self, runid, optrun):

        simpar = Simulator(
            self.simulator.param_dict["model_file"], self.simulator.param_dict["model_short"])
        simpar.param_dict = copy.deepcopy(self.simulator.param_dict)

        # Set termination to User data and continue after target is reached
        simpar.param_dict["terminate_at_timestep"] = self.iterations[runid]
        simpar.param_dict["continue_after_target_reach"] = True

        target_id = self.target_ids[runid]


        X, successful, XPOS = iso.run_movement(simpar, self.targets[target_id], self.width[target_id], str(optrun) + "_" + str(
            runid), self.outputfolder, self.usedXqpos[runid], self.usedXqvel[runid], self.usedACT[runid], self.usedEXT[runid], self.used_transfer, createvideos=False, save_results=False)

        if self.eermse:
            rmse_mvsu,rmse_uvsm = self.rmse(XPOS, self.movementid[runid])
        else:
            QPOS = np.array([X[k].qpos[self.simulator.param_dict["physical_joint_ids"]] for k in range(len(X))])
            rmse_mvsu,rmse_uvsm = self.rmse_joints(QPOS, self.movementid[runid])

        log.debug(f"rmse_mvsu: {rmse_mvsu}, rmse_uvsm: {rmse_uvsm}.")

        rmse = rmse_uvsm 
        return rmse, successful

    def rmse(self, endeffector_pos_simulation, movement_ID):
        """
        This function can be used for on-line evaluation of simulation trajectories that should approximate a certain user trajectory, using positional RMSE as measure.
        """
        endeffector_pos_simulation = [pos - self.rotshoulder for pos in endeffector_pos_simulation]
        endeffector_pos_simulation = np.array(endeffector_pos_simulation)

        trajectory_plots_data_users_markers_filtered = pd.read_csv(
            f'{DIRNAME_STUDY}/Marker/{self.participant}_{self.condition}.csv')
        target_switch_indices_data_users_markers = np.load(
            f'{DIRNAME_STUDY}/_trialIndices/{self.participant}_{self.condition}_SubMovIndices.npy', allow_pickle=True)

        (trial_start_index_user, trial_end_index_user, _, _, _) = target_switch_indices_data_users_markers[np.where(
            target_switch_indices_data_users_markers[:, 4] == movement_ID)[0][0]]

        endeffector_pos_simulation = pd.DataFrame(endeffector_pos_simulation, columns=[
                                                  "end-effector_xpos_x", "end-effector_xpos_y", "end-effector_xpos_z"], index=0.002*np.arange(endeffector_pos_simulation.shape[0])).rename_axis("time")
        endeffector_pos_user = trajectory_plots_data_users_markers_filtered.loc[trial_start_index_user:trial_end_index_user-1, ["time"] + ["end_effector" + suffix + xyz for suffix in ["_pos_", "_vel_", "_acc_"] for xyz in ["X", "Y", "Z"]] + [prefix + xyz for prefix in ["init_", "target_"] for xyz in ["x", "y", "z"]]].set_index("time").rename({
            "end_effector_pos_X": "end-effector_xpos_x",
            "end_effector_pos_Y": "end-effector_xpos_y",
            "end_effector_pos_Z": "end-effector_xpos_z",
            "end_effector_vel_X": "end-effector_xvel_x",
            "end_effector_vel_Y": "end-effector_xvel_y",
            "end_effector_vel_Z": "end-effector_xvel_z",
            "end_effector_acc_X": "end-effector_xacc_x",
            "end_effector_acc_Y": "end-effector_xacc_y",
            "end_effector_acc_Z": "end-effector_xacc_z"}, axis=1)
        endeffector_pos_user.index -= endeffector_pos_user.index[0]
        user_X = interpolate.interp1d(endeffector_pos_user.index,
                                      endeffector_pos_user.loc[:, "end-effector_xpos_x"])
        user_Y = interpolate.interp1d(endeffector_pos_user.index,
                                      endeffector_pos_user.loc[:, "end-effector_xpos_y"])
        user_Z = interpolate.interp1d(endeffector_pos_user.index,
                                      endeffector_pos_user.loc[:, "end-effector_xpos_z"])
        simulation_X = interpolate.interp1d(endeffector_pos_simulation.index,
                                            endeffector_pos_simulation.loc[:, "end-effector_xpos_x"])
        simulation_Y = interpolate.interp1d(endeffector_pos_simulation.index,
                                            endeffector_pos_simulation.loc[:, "end-effector_xpos_y"])
        simulation_Z = interpolate.interp1d(endeffector_pos_simulation.index,
                                            endeffector_pos_simulation.loc[:, "end-effector_xpos_z"])
        error_pos_MODELvsUSER = np.linalg.norm(endeffector_pos_simulation.loc[:, "end-effector_xpos_x":"end-effector_xpos_z"] - np.array((user_X(
            np.clip(endeffector_pos_simulation.index, endeffector_pos_user.index[0], endeffector_pos_user.index[-1])), user_Y(
            np.clip(endeffector_pos_simulation.index, endeffector_pos_user.index[0], endeffector_pos_user.index[-1])), user_Z(
            np.clip(endeffector_pos_simulation.index, endeffector_pos_user.index[0], endeffector_pos_user.index[-1])))).T, axis=1)
        RMSE_MODELvsUSER = np.sqrt(np.square(error_pos_MODELvsUSER).mean())
        error_pos_USERvsMODEL = np.linalg.norm(endeffector_pos_user.loc[:, "end-effector_xpos_x":"end-effector_xpos_z"] - np.array((simulation_X(
            np.clip(endeffector_pos_user.index, endeffector_pos_simulation.index[0], endeffector_pos_simulation.index[-1])), simulation_Y(
            np.clip(endeffector_pos_user.index, endeffector_pos_simulation.index[0], endeffector_pos_simulation.index[-1])), simulation_Z(
            np.clip(endeffector_pos_user.index, endeffector_pos_simulation.index[0], endeffector_pos_simulation.index[-1])))).T, axis=1)
        RMSE_USERvsMODEL = np.sqrt(np.square(error_pos_USERvsMODEL).mean())

        return RMSE_MODELvsUSER, RMSE_USERvsMODEL

    def rmse_joints(self, joint_angle_simulation, movement_ID, level='pos', dt=0.002):
        """
        This function can be used for on-line evaluation of simulation trajectories that should approximate a certain user trajectory, using positional RMSE of all joints as measure.
        """
        trajectory_plots_data_users_angles_filtered = pd.read_csv(f'{DIRNAME_STUDY}/IK/{self.participant}_{self.condition}.csv')
        trajectory_plots_data_users_angles = trajectory_plots_data_users_angles_filtered  
        trajectory_plots_data_users_torques = pd.read_csv(f'{DIRNAME_STUDY}/ID/{self.participant}_{self.condition}.csv')
        target_switch_indices_data_users_markers = np.load(f'{DIRNAME_STUDY}/_trialIndices/{self.participant}_{self.condition}_SubMovIndices.npy', allow_pickle=True)

        joint_list = ["elv_angle", "shoulder_elv", "shoulder_rot", "elbow_flexion", "pro_sup", "deviation", "flexion"]

        (trial_start_index_user, trial_end_index_user, _, _, _) = target_switch_indices_data_users_markers[np.where(target_switch_indices_data_users_markers[:, 4] == movement_ID)[0][0]]

        joint_angle_simulation = pd.DataFrame(joint_angle_simulation, columns=[f"{cn}_{level}" for cn in joint_list], index=dt * np.arange(joint_angle_simulation.shape[0])).rename_axis("time")
        joint_angle_user = (trajectory_plots_data_users_angles.loc[trial_start_index_user:trial_end_index_user, ['time'] + [joint_name + suffix for joint_name in joint_list for suffix in ['_pos', '_vel', '_acc']]].set_index("time"))
        joint_angle_user.index -= joint_angle_user.index[0]
        joint_torque_user = (trajectory_plots_data_users_torques.loc[trial_start_index_user:trial_end_index_user, ['time'] + [joint_name + suffix for joint_name in joint_list for suffix in ['_moment']]].set_index("time")).rename({joint_name + '_moment': joint_name + '_frc' for joint_name in joint_list}, axis=1)
        joint_torque_user.index -= joint_torque_user.index[0]

        if level in ['pos', 'vel', 'acc']:
            user_JOINT = [interpolate.interp1d(joint_angle_user.index,
                                                joint_angle_user.loc[:, joint_name]) for joint_name in joint_angle_user.columns if joint_name.endswith(f"_{level}")]
            simulation_JOINT = [interpolate.interp1d(joint_angle_simulation.index,
                                                joint_angle_simulation.loc[:, joint_name]) for joint_name in joint_angle_simulation.columns if joint_name.endswith(f"_{level}")]   

            error_joint_MODELvsUSER = np.linalg.norm(joint_angle_simulation - np.array([user_JOINT_joint(
                np.clip(joint_angle_simulation.index, joint_angle_user.index[0], joint_angle_user.index[-1])) for user_JOINT_joint in user_JOINT]).T, axis=1)
            RMSE_jointspace_MODELvsUSER = np.sqrt(np.square(error_joint_MODELvsUSER).mean())

            error_joint_USERvsMODEL = np.linalg.norm(joint_angle_user.loc[:, [cn for cn in joint_angle_user.columns if cn.endswith(f"_{level}")]] - np.array([simulation_JOINT_joint(
                np.clip(joint_angle_user.index, joint_angle_simulation.index[0], joint_angle_simulation.index[-1])) for simulation_JOINT_joint in simulation_JOINT]).T, axis=1)
            RMSE_jointspace_USERvsMODEL = np.sqrt(np.square(error_joint_USERvsMODEL).mean())
        elif level in ['frc']:
            user_JOINT = [interpolate.interp1d(joint_torque_user.index,
                                                joint_torque_user.loc[:, joint_name]) for joint_name in joint_torque_user.columns if joint_name.endswith(f"_{level}")]
            simulation_JOINT = [interpolate.interp1d(joint_angle_simulation.index,
                                                joint_angle_simulation.loc[:, joint_name]) for joint_name in joint_angle_simulation.columns if joint_name.endswith(f"_{level}")]   

            error_joint_MODELvsUSER = np.linalg.norm(joint_angle_simulation - np.array([user_JOINT_joint(
                np.clip(joint_angle_simulation.index, joint_torque_user.index[0], joint_torque_user.index[-1])) for user_JOINT_joint in user_JOINT]).T, axis=1)
            RMSE_jointspace_MODELvsUSER = np.sqrt(np.square(error_joint_MODELvsUSER).mean())

            error_joint_USERvsMODEL = np.linalg.norm(joint_torque_user.loc[:, [cn for cn in joint_torque_user.columns if cn.endswith(f"_{level}")]] - np.array([simulation_JOINT_joint(
                np.clip(joint_torque_user.index, joint_angle_simulation.index[0], joint_angle_simulation.index[-1])) for simulation_JOINT_joint in simulation_JOINT]).T, axis=1)
            RMSE_jointspace_USERvsMODEL = np.sqrt(np.square(error_joint_USERvsMODEL).mean())
        else:
            raise NotImplementedError

        return RMSE_jointspace_MODELvsUSER, RMSE_jointspace_USERvsMODEL


if __name__ == "__main__":

    log.debug("DEBUG TEST FOR PARAM OPTIMIZATION")
  
    po = ParamOptimizer("U1", "Virtual_Cursor_ID_ISO_15_plane", "PO_DEBUG_", "JAC", debug=True)

    # Uncomment for actual optimization
    #po = ParamOptimizer("U1", "Virtual_Cursor_ID_ISO_15_plane", "PO_", "JAC", init_params="../data/parameters/params_jac.csv")

    optx = po.optimize()

    log.debug(f"DEBUG FINISHED. SOLUTION: {optx}")
