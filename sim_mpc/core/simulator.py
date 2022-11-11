''' 
Class for simulating interaction trajectories with biomechanical model implemented in MuJoCo and model predictive control.

Authors: Markus Klar
Date: 11.2022
'''
import logging as log
import math
import time

import mujoco_py
import numpy as np
import scipy.optimize as opt

import sim_mpc.core.transferfunctions as transfer
# Local imports
import sim_mpc.core.utils as utils


class Simulator:
    def __init__(self, model_file, model_short="NA", **kwargs):
        ### General Settings
        self.param_dict = {} # Dictionary that holds all parameters
        
        # MuJoCo Model
        self.param_dict["model_file"] = model_file
        self.param_dict["model_short"] = model_short

        # Load the model
        self.model = mujoco_py.load_model_from_path(self.param_dict["model_file"]) 

        # Save model variables for quick access
        self.nq = self.model.nq
        self.nu = self.model.nu

        # Define time parameters
        self.param_dict["h"] = 0.002 # Discretization time (s)
        self.model.opt.timestep = self.param_dict["h"] # set the mujoco model timestep
        
        # Define time parameters
        self.param_dict["max_iterations"] = 30 # Max Iterations
        self.param_dict["iterations"] = 0 # Iterations until stop
        self.param_dict["total_time"] = 0 # Total calculation time of the simulation
        self.param_dict["N"] = 4 # MPC horizon
        self.param_dict["num_steps"] = 20 # How many timesteps per iteration should be skipped (run with same control)
        self.param_dict["openloop_steps"] = 1 # Number of MPC Openloop solution steps that are used in one iteration

        # Second order muscle model
        self.activation = np.zeros((self.model.nu,)) # Activation (sigma)
        self.d_activation = np.zeros((self.model.nu,)) # Change in activation (Delta sigma)
        
        # Terminal conditions
        # If distance to target < stoptol_pos && vel < stoptol_vel, the simulation will stop successfully 
        self.param_dict["stoptol_pos"] = 0.025  # Radius of the target in meter
        self.param_dict["stoptol_vel"] = 0.5 # Maximum speed of the cursor in m/s for termination
        
        self.param_dict["stay_at_target"] = 0 # Stay at the target for this long (in s). 0 equals no staying
        self.param_dict["terminate_unsucc_stay"] = True # If True, terminate simulation if target is left during stay_at_target time

        self.param_dict["continue_after_target_reach"] = False # If True, do not stop simulation if target is reached
        self.param_dict["terminate_at_timestep"] = -1 # If >0, terminate simulation at given timestep

        self.param_dict["pretermination"] = False # if True, terminate simulation if distance is not reduced at least by a threshold after first openloop
        self.param_dict["pretermination_threshold"] = 0.02 # Distance in Meter that need to be move towards the target

        # Debugmode
        self.param_dict["debugmode"] = False
        
        ### Comfortability Settings
        self.param_dict["display_steps"] = True # Display results of MPC steps
        self.param_dict["verbosity"] = True # Display general information

        ### Debug setting
        self.param_dict["debug_optimization"] = False # Stop optimization after one iteration
        
        ### Initial Values
        self.param_dict["init_qpos"] = self.model.key_qpos[0] # Load initial angle positions from mujoco model keypos
        self.param_dict["init_qvel"] = np.zeros(self.model.nv) # Initial angle velocities set to 0
        
        ### Target
        self.param_dict["target"] = np.array([.0,.0,.0]) 
        
        ### Cost Function Settings
        # Cursor distance to target cost
        self.param_dict["distance_cost"] = True  # If True, distance cost will be included
        self.param_dict["d_cost_type"] = 'lin' # Choose distance cost type: lin, quad, exp
        self.param_dict["w1"] = 1 # Position weight (distance to target)
        self.param_dict["wexp"] = 0.01**2 #additional weight for exp. distance cost (must be > 0)
        
        # Control cost
        self.param_dict["control_cost"] = True
        self.param_dict["ctrl_cost"] = "quad"  # choose control cost: quad, multgear, work   
        self.param_dict["r"] = 0.1 # Effort weight
                
        # Commanded Torque Change Cost
        self.param_dict["commanded_tc_cost"] = False
        self.param_dict["commanded_tc_cost_weight"] = 1

        # Joint Acceleration Cost
        self.param_dict["acceleration_cost"] = False
        self.param_dict["acceleration_cost_weight"] = 1
   
        # Controlbounds
        self.param_dict["ctrl_range"] = self.model.actuator_ctrlrange[:].copy()
        self.param_dict["ctrlbounds"] = [(ctrl[0],ctrl[1]) for ctrl in self.param_dict["ctrl_range"]]
        
        ### Noise Settings
        self.param_dict["noise"] = False  # If True, noise is included in the evaluation of the system
        self.param_dict["signal_dependent_noise"] = 0.103 # Input dependent noise standard deviation, multiply with activation of each joint motor
        self.param_dict["constantnoise_param"] = 0.185 # Constant motor noise
        self.param_dict["noise_seed"] = 1337
        
        ### MuJoCo model joint names and ids
        self.param_dict["physical_joints"] = ["elv_angle", "shoulder_elv", "shoulder_rot", "elbow_flexion", "pro_sup", "deviation", "flexion"]
        self.param_dict["joint_names"] = [self.model.joint_id2name(i) for i in range(self.model.njnt)]
        self.param_dict["physical_joint_ids"] = [self.model.joint_name2id(name) for name in self.param_dict["physical_joints"]]
        self.param_dict["virtual_joint_ids"] = [i for i in range(self.model.njnt) if (i not in self.param_dict["physical_joint_ids"])]
        self.param_dict["virtual_joints"] = [self.model.joint_id2name(i) for i in self.param_dict["virtual_joint_ids"]]

        self.param_dict["actuator_gears"] = np.array(self.model.actuator_gear[:, 0] ) # Gears from mujoco model
        
        ### Misc parameters
        self.param_dict["init_time"] = 0 # Initial time for visualization
        self.param_dict["iso_targets"] = False # Set True if visualizhation of ISO task targets is desired
        self.param_dict["use_torso_constraints"] = True # Can be deactivated if torso is to be moved
        
        # Time parameters for second-order muscle model
        self.param_dict["t_activation"] = 0.04
        self.param_dict["t_excitation"] = 0.03
        
        # Names
        self.param_dict["end_effector"] = "end-effector"  # End effector name in mujoco file
        self.param_dict["cursor"] = "cursor" # transferpoint name in mujoco file
        self.param_dict["filename"] = ""
        
        ### Optimization settings      
        self.param_dict["optmethod"] = 'L-BFGS-B' #'SLSQP' #'L-BFGS-B' #'One of the methods specified by scipy.optimize.minimize
        self.param_dict["opts"] = {'maxfun': 10000, 'disp': False, 'gtol': 1e-5,'ftol': 1e-6, 'eps': 1e-08, 'maxiter': 1000}

        self.param_dict["run_openloop"] = False # If True, run simulation in openloop instead of MPC (only without noise)
           
        # Options for scipy.optimize.minimize
        self.param_dict["opt_tol"] =  1e-8 # Optimization precision for constraints
   
    #### Parameter functions
    # Extend number of maximal iterations
    def extend_max_iter(self, num_steps):
        self.param_dict["max_iterations"] += num_steps
        

    # Reset model information    
    def reset_model_info(self):
        self.model = mujoco_py.load_model_from_path(self.param_dict["model_file"])

        self.model.opt.timestep = self.param_dict["h"] # set the mujoco model timestep
        
        self.param_dict["init_qpos"] = self.model.key_qpos[0]
        self.param_dict["init_qvel"] = np.zeros(self.model.nv)
        
        self.param_dict["actuator_gears"] = self.model.actuator_gear[:, 0] #gears from xml model
        
        # setup joint names and ids
        self.param_dict["joint_names"] = [self.model.joint_id2name(i) for i in range(self.model.njnt)]
        
        self.param_dict["physical_joints"] = ["elv_angle", "shoulder_elv", "shoulder_rot", "elbow_flexion", "pro_sup", "deviation", "flexion"]
            
        self.param_dict["physical_joint_ids"] = [self.model.joint_name2id(name) for name in self.param_dict["physical_joints"]]
        
        self.param_dict["virtual_joint_ids"] = [i for i in range(self.model.njnt) if (i not in self.param_dict["physical_joint_ids"])]
        self.param_dict["virtual_joints"] = [self.model.joint_id2name(i) for i in self.param_dict["virtual_joint_ids"]]    

    
    #### Simulation functions
    def init_sim(self):
        self.sim = mujoco_py.MjSim(self.model)

    def reset(self):
        # Reset simulation
        self.sim.reset()

        # Set initial state
        x0 = self.sim.get_state()
        x0.qpos[:] = self.param_dict["init_qpos"]
        x0.qvel[:] = self.param_dict["init_qvel"]
        self.sim.set_state(x0)

        # Set noise seed if chosen
        if self.param_dict["noise_seed"] is not None:
            np.random.seed(self.param_dict["noise_seed"])

        return self.sim.get_state()

    def eval_sys(self, x0, U, steps, transferfct):
        """Evaluate the system with given initial state x0 and control trajectory U.

        Args:
            x0, defines the initial condition 
            U, control trajectory used for evaluation
            steps, number of steps to be simulated
            transferfct, used transferfunction

        Returns:
            X, 
            XPOS, 
            XVEL, 
            TORQUE, 
            XXPOS, 
            XXVEL, 
            XXACC

        """       
        self.sim.set_state(x0) # Set initial state
        mujoco_py.cymj._mj_kinematics(self.model, self.sim.data) # Eun kinematics to update body_xpos

        X = [x0]
        XX = [x0]
        QACC = []
        xpos, xvel, xacc = transferfct(self.sim, self)
        XPOS = [xpos]
        XVEL = [xvel]
        TORQUE = [np.array(self.sim.data.qfrc_inverse)]
        XXACC = [xacc]
        XXPOS = [xpos]
        XXVEL = [xvel]
        ACT = []
        DACT = []

        act = self.activation.copy()
        d_act = self.d_activation.copy()
        ACT = [self.activation.copy()]
        DACT = [self.d_activation.copy()]
              
        # For each step, set control and forward simulation
        for k in range(steps-1):

            # Update transferpoint position
            utils.updatebodypos(self.sim, "cursor", XPOS[-1])

            # Set controls for next step
            control = U[self.model.nu*k:self.model.nu*(k+1)]
            
            # Forward the simulation for num_steps with the same control
            for l in range(self.param_dict["num_steps"]):
                
                # Forward second-order muscle model and set MuJoCo controls
                act, d_act = utils.muscle_activation_model_secondorder(act, d_act, control, self.param_dict["h"], self.param_dict["t_activation"], self.param_dict["t_excitation"])
                self.sim.data.ctrl[:] = act

                # Save sigma and delta sigma
                ACT.append(act.copy())
                DACT.append(d_act.copy())
                    
                # MuJoCo simulation step
                try:
                    self.sim.step()
                except mujoco_py.MujocoException:
                    log.warning("Mujoco Exception in optimization step occured. Probably due to physical instability. Terminate simulation.")

                    log.debug(self.sim.data.ctrl)
                    log.debug(act)
                    log.debug(d_act)
                    raise mujoco_py.MujocoException
                        
                # Calculate MuJoCo kinematics to update global positions and velocities (needed to get end-effector state)   
                mujoco_py.cymj._mj_kinematics(self.model, self.sim.data)
                
                # Get state of the cursor from the physical end-effector (input device/interface dynamics)
                xpos, xvel, xacc = transferfct(self.sim, self)

                # Sace MuJoCo state
                XX.append(self.sim.get_state())

                # Save transferpoint state
                XXPOS.append(xpos)
                XXVEL.append(xvel)
                XXACC.append(xacc)

                # Update global body position of the transferpoint (for visualization)
                utils.updatebodypos(self.sim, "cursor", xpos)
            #end for num_steps

            # Save MuJoCo state and transferfct pos after num_steps
            X.append(self.sim.get_state())
            QACC.append(np.array(self.sim.data.qacc[:]))
            XPOS.append(xpos)
            XVEL.append(xvel)
        #end for steps


        # Reset simulation to initial state
        self.sim.set_state(x0)

        return X, XX, QACC, XPOS, XVEL, TORQUE, XXPOS, XXVEL, XXACC, ACT, DACT


    def cost_fct_opt(self, U, x0, cur_iter, steps, transferfct, printcost=False):
        """Wrapper for costfunction that only returns the total cost.
        Args: 
        U, 
        x0, 
        
        cur_iter, 
        steps, 
        transferfct, 
        printcost=False
        
        Returns: 
        total_cost
        
        """

        total_cost,_= self.cost_fct(U, x0, cur_iter, steps, transferfct, printcost)

        return total_cost 
    

    def cost_fct(self, U, x0, cur_iter, steps, transferfct, printcost=False):
        """Calculate the cost of a given control U with inital state x0.
        Args: 
        U, 
        x0, 
        
        cur_iter, 
        steps, 
        transferfct, 
        printcost=False
        
        Returns: 
        total_cost,
        cost_dict
        
        """
        # Evaluate the system and get system and transferpoint trajectories
        X, XX, _, XPOS, _, _, _, XXVEL, _, ACT, _ = self.eval_sys(x0, U, steps, transferfct)

        total_cost = 0 # Total cost for U
        cost_dict = {} # Dictionary that contains the different cost types and their value

        # Set target
        target = self.param_dict["target"]

        # Distance cost
        dist_cost = 0
        if self.param_dict["distance_cost"]:
            # Check for valid cost type:
            if self.param_dict["d_cost_type"] not in ["lin","quad","exp","log","sqrt"]:
                log.warning("Warning: No valid distance cost type chosen [lin, quad, exp]! Using linear cost.")
                self.param_dict["d_cost_type"] = "lin"

            for k in range(1,steps):
                dist = np.linalg.norm(XPOS[k] - target)
                if self.param_dict["d_cost_type"] == "quad":
                    dist_cost +=  dist**2   
                elif self.param_dict["d_cost_type"] == "lin":
                    dist_cost += dist
                elif self.param_dict["d_cost_type"] == "exp":
                    dist_cost += np.exp(- np.square(dist) / self.param_dict["wexp"])
                elif self.param_dict["d_cost_type"] == "log":
                    dist_cost += 2*dist + np.log(1+10*dist)
                elif self.param_dict["d_cost_type"] == "sqrt":
                    dist_cost += np.sqrt(dist)
            dist_cost *= self.param_dict["w1"]

            if printcost:
                log.info("Distance cost = {}".format(dist_cost))

            total_cost += dist_cost 
            cost_dict["dist_cost"] = dist_cost
        
        # Control cost
        if self.param_dict["control_cost"]:
            ctrl_cost = 0
            for k in range(steps-1):
                if self.param_dict["ctrl_cost"] == "multgear":
                    ctrl_cost += self.param_dict["r"] * np.linalg.norm(U[self.model.nu*k:self.model.nu*(k+1)]*self.param_dict["actuator_gears"][:])**2
                elif self.param_dict["ctrl_cost"] == "quad":
                    ctrl_cost += self.param_dict["r"]  *np.square(np.linalg.norm(U[self.model.nu*k:self.model.nu*(k+1)]))
                else:
                    ctrl_cost += self.param_dict["r"] * np.linalg.norm(U[self.model.nu*k:self.model.nu*(k+1)])
                    
            total_cost += ctrl_cost
            cost_dict["ctrl_cost"] = ctrl_cost
            
            if printcost:
                log.info("Control cost = {}".format(ctrl_cost))

        # Commanded Torque Change Cost
        if self.param_dict["commanded_tc_cost"]:
            mjsteps = len(ACT)
            tc_cost = 0
            tc = np.zeros((len(self.param_dict["physical_joint_ids"]),mjsteps))

            for k in range(mjsteps):
                tc[:,k] = ACT[k]*self.param_dict["actuator_gears"]

            for joint in range(self.nu):
                tc[joint][:] = np.gradient(tc[joint][:]) / self.param_dict["h"]

            tc = tc.transpose()

            for k in range(mjsteps):
                tc_cost += np.square(np.linalg.norm(tc[k][:])) 
            
            tc_cost *= self.param_dict["commanded_tc_cost_weight"] 

            total_cost += tc_cost
            cost_dict["tc_cost"] = tc_cost    

            if printcost:
                log.info("Torque change cost = {}".format(tc_cost))  

        # Joint Acceleration Cost
        if self.param_dict["acceleration_cost"]:
            acceleration_cost_old = 0
            acceleration_cost = 0
            mjsteps = len(XXVEL)
            
            qacc_old = np.zeros((len(self.param_dict["physical_joint_ids"]),steps))
            qacc = np.zeros((len(self.param_dict["physical_joint_ids"]),mjsteps))
            jid = 0
            for joint in self.param_dict["physical_joint_ids"]:
                qacc_old[jid][:] = np.gradient(np.array([X[k].qvel[joint] for k in range(steps)])) / (self.param_dict["num_steps"] * self.param_dict["h"])
                qacc[jid][:] = np.gradient(np.array([XX[k].qvel[joint] for k in range(mjsteps)])) / self.param_dict["h"]
                jid += 1
                
            qacc_old = qacc_old.transpose()
            qacc = qacc.transpose()

            for k in range(steps):   
                acceleration_cost_old += np.square(np.linalg.norm(qacc_old[k])) 
            for k in range(mjsteps):
                acceleration_cost += np.square(np.linalg.norm(qacc[k]))

            acceleration_cost_old = self.param_dict["acceleration_cost_weight"] * acceleration_cost_old
            acceleration_cost = self.param_dict["acceleration_cost_weight"] * acceleration_cost

            if printcost:
                log.info("Acceleration Cost = {}".format(acceleration_cost))

            total_cost += acceleration_cost
            cost_dict["acceleration_cost"] = acceleration_cost  

        return total_cost, cost_dict 


    def minimize_cost(self, u_start, x0, cur_iter, steps, transferfct):
        """Optimization routine that finds the control with minimal costs.
        Args: 
        u_start, 
        x0, 
        cur_iter, 
        steps,
        transferfct
        
        Returns: 
        sol
        
        
        """
        if self.param_dict["debug_optimization"]:
            log.debug("Debugging Optimization...")
            log.debug(f"optmethod: {self.param_dict['optmethod']}; options: {self.param_dict['opts']}")
        
        constraints = ()

        ctrl_range = self.model.actuator_ctrlrange[:]
        
        # Control bounds
        bounds = [(ctrl[0],ctrl[1]) for ctrl in ctrl_range]
        bounds *= (self.param_dict["N"]-1)
   
        sol = opt.minimize(self.cost_fct_opt, u_start, args=(x0, cur_iter, steps, transferfct, False), method=self.param_dict["optmethod"], options=self.param_dict["opts"], constraints=constraints, bounds=bounds)

        return sol  


    def mpc(self, x0, transferfct):
        """Main model predictive control routine to simulate a movement.
        Args:
        x0,
        transferfct
        
        Returns:
        X, 
        U, 
        XPOS, 
        XVEL, 
        XACC, 
        TORQUE, 
        JN, 
        NFEV, 
        COST, 
        total_time, 
        target_reached
        """
        # Initialize MPC
        u = np.array([(ctrl[1]-ctrl[0])/2 + ctrl[0] for ctrl in self.model.actuator_ctrlrange[:]] * (self.param_dict["N"]-1)) # Init with 50% control

        #sim = mujoco_py.MjSim(self.model) # Create closed-loop MuJoCo simulation

        self.sim.set_state(x0) # Set initial state
        mujoco_py.cymj._mj_kinematics(self.sim.model, self.sim.data)

        # Update target position (for visualization)
        utils.updatebodypos(self.sim, "target", self.param_dict["target"])

        # Make one forward step (do not proceed state) to get initial end effector velocity
        self.sim.data.ctrl[:] = self.activation
        self.sim.forward()

        # Initialize output
        U = []
        X = [x0]
        QACC = []
        xpos, xvel, xacc = transferfct(self.sim, self)
        XPOS = [xpos]
        XVEL = [xvel]
        XACC = [xacc]
        JN = []
        NFEV =[]
        COST = []
        TORQUE = []
        ACT = [self.activation]
        DACT = [self.d_activation]

        # save initial end effector pos in param
        self.param_dict["initial_pos"] = xpos

        target_reached = False
        stay_at_target_time = self.param_dict["stay_at_target"]
        stay_at_target_iter = 0

        # Pretermination startdistance
        distpreterm = np.linalg.norm(XPOS[-1]-self.param_dict["target"])

        if self.param_dict["verbosity"]:
            log.info("-------------------- Starting MPC --------------------")
            log.info("init_pos: {0} - init_vel: {1} - first target: {2} - start distance: {3}".format(XPOS[-1], XVEL[-1], self.param_dict["target"], distpreterm))

        start_time = time.time()
        cur_iter = 0

        ## Main loop. Simulate until target or max_iterations is reached
        while cur_iter < self.param_dict["max_iterations"]:

            # Get current system state
            x0 = self.sim.get_state()

            # Update transferpoint position
            utils.updatebodypos(self.sim, "cursor", XPOS[-1])

            # Start time keeping
            stepstart = time.time()

            # Main part: Find optimal control u for inital state x and next N steps
            if cur_iter == 0 or not self.param_dict["run_openloop"]:
                sol = self.minimize_cost(u, x0, cur_iter, self.param_dict["N"], transferfct)

            # Stop time keeping
            stepend = time.time()

            # Save output
            JN.append(sol.fun)
            NFEV.append(sol.nfev)

            # set u*
            uopt = sol.x

            # Display step information
            if self.param_dict["display_steps"]:
                log.info(f"------- Results step {cur_iter} -------")
                if cur_iter==0 or not self.param_dict["run_openloop"]:
                    Jprint, cost_dict = self.cost_fct(uopt, x0, cur_iter, self.param_dict["N"], transferfct, True)
                    log.info("J_N = {}".format(Jprint))
            else:
                _, cost_dict = self.cost_fct(uopt, x0, cur_iter, self.param_dict["N"], transferfct, False)

            if not self.param_dict["run_openloop"]:
                # Use first part of optimal u as control
                u_star = uopt[:self.model.nu*self.param_dict["openloop_steps"]]
            else:
                # Use corresponding part of optimal u as control
                u_star = uopt[self.model.nu*cur_iter*self.param_dict["openloop_steps"]:(cur_iter+1)*self.model.nu*self.param_dict["openloop_steps"]]

            # Save cost
            COST.append(cost_dict)

            # Get openloop for pretermination
            if self.param_dict["pretermination"] and cur_iter == 0:
                _, _, _, XPOSol, _, _, _, _, _, _, _ = self.eval_sys(x0, uopt, self.param_dict["N"], transferfct)

            # Proceed simulation with optimal control
            for l in range(self.param_dict["openloop_steps"]):

                applied_control = u_star[l*self.model.nu:(l+1)*self.model.nu]
                
                if self.param_dict["noise"]:
                    # Obtain control noise
                    current_noise = np.zeros(applied_control.shape)
                    current_noise = utils.get_motor_noise(applied_control, self)
                    if self.param_dict["display_steps"]: 
                        log.info("Curnoise = {}".format(current_noise))

                    # Apply noise and clip control + noise to stay in ctrl_range
                    applied_control = np.clip(u_star[l*self.model.nu:(l+1)*self.model.nu] + current_noise, self.param_dict["ctrl_range"][:,0], self.param_dict["ctrl_range"][:,1])
                

                # Forward simulation with same control for num_steps 
                for k in range(self.param_dict["num_steps"]):
                    # Update second order muscle model and set MuJoCo control
                    self.activation, self.d_activation = utils.muscle_activation_model_secondorder(self.activation, self.d_activation,  applied_control, self.param_dict["h"], self.param_dict["t_activation"], self.param_dict["t_excitation"])
                    self.sim.data.ctrl[:] = self.activation
                    ACT.append(self.activation[:])
                    DACT.append(self.d_activation[:])
                    
                    # Forward MuJoCo simulation
                    try:
                        self.sim.step()
                    except mujoco_py.MujocoException:
                        log.info("Mujoco Exception in closedloop step occured. Probably due to physical instability. Terminate simulation.")
                        if self.param_dict["debugmode"]:
                            log.debug(applied_control)
                            log.debug(self.activation)
                            log.debug(self.d_activation)
                        raise mujoco_py.MujocoException
                    
                    # Forward kinematics to obtain new cursor state
                    mujoco_py.cymj._mj_kinematics(self.model, self.sim.data)
                    xpos, xvel, xacc = transferfct(self.sim, self)

                    # Update transferpoint position
                    utils.updatebodypos(self.sim, "cursor", xpos)

                    XPOS.append(xpos)
                    XVEL.append(xvel)
                    XACC.append(xacc)

                    # Calculate invervse dynamics to obtain total torque
                    mujoco_py.cymj._mj_inverse(self.model, self.sim.data)
                    TORQUE.append(np.array(self.sim.data.qfrc_inverse))

                    # Save trajectory
                    U.append(u_star[l*self.model.nu:(l+1)*self.model.nu])
                    X.append(self.sim.get_state())
                    QACC.append(np.array(self.sim.data.qacc[:]))

            # Warmstart: Use shifted optimal control as initial control for next step
            u = np.hstack([uopt[self.model.nu*self.param_dict["openloop_steps"]:], uopt[self.model.nu*self.param_dict["openloop_steps"]*(self.param_dict["N"]-2):]])

            # Get current cursor state
            xpos, xvel, xacc = transferfct(self.sim, self)

            dist = np.linalg.norm(XPOS[-1]-self.param_dict["target"])
            vel = np.linalg.norm(XVEL[-1])

            if self.param_dict["display_steps"]: 
                log.info("MPC Iteration {0}: J_N = {1} --- nfev = {2}".format(cur_iter, sol.fun, sol.nfev))
                log.info("Cursor position: {0} --- Target: {1}".format(XPOS[-1],self.param_dict["target"]))
                log.info("Remaining distance: {0} --- Current cursor velocity: {1}".format(dist, vel))
                log.info("Solver message: {0}".format(sol.message))
                log.info('Step time: {0:5.3f}s  --- Elapsed time: {1:5.3f}s --- Remaining time (est.): {2:5.3f}s'.format(stepend - stepstart, stepend - start_time, ((stepend - start_time) / (cur_iter+1))* (self.param_dict["max_iterations"]-cur_iter-1)))

            # Terminate if target or maximum steps are reached
            if dist < self.param_dict["stoptol_pos"] and vel < self.param_dict["stoptol_vel"]:
                # If staying at target is active, do so as long as desired
                if self.param_dict["stay_at_target"] > 1e-6:
                    if not target_reached:
                        if self.param_dict["display_steps"]:
                            log.info("Target reached within tolerance at iteration {0}. Remaining distance: {1} - Velocity: {2}. Stay at target is active. Simulations has to stay at target for {3}s.".format(cur_iter, dist, vel, self.param_dict["stay_at_target"]))
                        target_reached = True
                        # Extend maxiteration such that staying at target is possible long enough
                        stay_at_target_iter = math.ceil((self.param_dict["stay_at_target"]/self.param_dict["h"])/self.param_dict["num_steps"])
                        self.extendmaxiter(stay_at_target_iter)
                        if self.param_dict["display_steps"]:
                            log.info("New max_iterations: {}".format(self.param_dict["max_iterations"]))

                    else:
                        stay_at_target_time -= self.param_dict["h"]*self.param_dict["num_steps"]*self.param_dict["openloop_steps"]
                        stay_at_target_iter -= 1
                        if stay_at_target_iter > 1e-6:
                            if self.param_dict["display_steps"]:
                                log.info("Simulation has to stay at target for {}s - iterations: {}".format(stay_at_target_time, stay_at_target_iter))
                                log.info("Control to stay_at_target = {} \nSq Norm of that control = {}".format(U[-1], np.square(np.linalg.norm(U[-1])) ))
                            
                        else:
                            if self.param_dict["verbosity"]:
                                log.info("Success. Simulation stayed at target long enough. Simulation stops at iteration {0}. Remaining distance: {1} - Velocity: {2}.".format(cur_iter, dist, vel))
                            target_reached = True
                            cur_iter += 1
                            break
                else:
                    # Continue simulation although target is already reached
                    if self.param_dict["continue_after_target_reach"]:
                        if self.param_dict["display_steps"]:
                            log.info("Success. Target reached within tolerance. Simulation CONTINUES with iteration {0}. Remaining distance: {1} - Velocity: {2}".format(cur_iter, dist, vel))
                        target_reached = True
                    # Terminate simulation otherwise
                    else:
                        if self.param_dict["verbosity"]:
                            log.info("Success. Target reached within tolerance. Simulation stops at iteration {0}. Remaining distance: {1} - Velocity: {2}".format(cur_iter, dist, vel))
                        cur_iter += 1
                        target_reached = True
                        break
            # If target was already reached, staying at target was unsuccesful
            elif target_reached and self.param_dict["terminate_unsucc_stay"]:
                if self.param_dict["verbosity"]:
                    log.info("Failure. Simulation did not stay at target long enough. Simulation stops at iteration {0}. Distance: {1} - Velocity: {2}.".format(cur_iter, dist, vel, self.param_dict["stay_at_target"]))
                cur_iter += 1
                target_reached = False
                break
            # Check if pretermination conditions hold in first MPC step
            elif self.param_dict["pretermination"] and cur_iter == 0:
                distol = np.linalg.norm(XPOSol[-1]-self.param_dict["target"][0])
                distchange = distpreterm - distol
                if distchange < self.param_dict["pretermination_threshold"]:
                    if self.param_dict["verbosity"]:
                        log.info("Failure - Pretermination. Simulation did not proceed far enough in first openloop, distance change = {3}, threshold = {4}. Simulation stops at iteration {0}. Distance: {1} - Velocity: {2}.".format(cur_iter, dist, vel, distchange, self.param_dict["pretermination_threshold"]))
                        log.info(f"DEBUG: pretermination distchange = {distchange}, distol = {distol}")
                    cur_iter += 1
                    target_reached = False
                    break

            # Terminate at given timestep
            if self.param_dict["terminate_at_timestep"] - 1  == cur_iter:
                if self.param_dict["verbosity"]:
                    log.info(f"Terminating simulation at timestep {cur_iter}. Simulations stops. Distance: {dist} - Velocity: {vel}")
                target_reached = (dist < self.param_dict["stoptol_pos"] and vel < self.param_dict["stoptol_vel"])
                cur_iter += 1
                break

            # Increase iteration number
            cur_iter += 1

        # end while
        ## End of main loop

        # Update total iteration number
        self.param_dict["iterations"] = cur_iter

        end_time = time.time()

        total_time = end_time - start_time

        self.param_dict["total_time"] = total_time

        if self.param_dict["verbosity"]:
            log.info('MPC finished.')
            log.info('Total Time{:5.3f}s'.format(total_time))
            log.info("-------------------- End MPC --------------------")

        return X, QACC, U, XPOS, XVEL, XACC, TORQUE, JN, NFEV, COST, ACT, DACT, total_time, target_reached


    def run_movement(self, transfer_fct=transfer.transfer_dflt):
        """Main function. Runs simulation with MPC.


        """
        # Initialise simulation
        self.init_sim()

        # Reset simulation and create initial state
        x0 = self.reset()

        return self.mpc(x0, transfer_fct)
       