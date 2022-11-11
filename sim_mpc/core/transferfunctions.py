''' 
A collection of transferfunctions that can be used to obtain cursor positions and velocities.

Note: Cursor accelerations are not yet implemented.

Authors: Markus Klar
Date: 11.2022
'''
import numpy as np
  
# default transfer (no transfer)
def transfer_dflt(sim, simulator):
    xpos = np.array(sim.data.get_body_xpos(simulator.param_dict["end_effector"]))
    xvel = np.array(sim.data.get_body_xvelp(simulator.param_dict["end_effector"]))
    
    
    if "ee_acc" in [sim.model.sensor_id2name(i) for i in range(sim.model.nsensor)]:
        xacc = np.array(sim.data.sensordata[sim.model.sensor_name2id("ee_acc")*3:sim.model.sensor_name2id("ee_acc")*3+3])
    elif "accsensor_end-effector" in [sim.model.sensor_id2name(i) for i in range(sim.model.nsensor)]:
        xacc = np.array(sim.data.sensordata[sim.model.sensor_name2id("accsensor_end-effector")*3:sim.model.sensor_name2id("accsensor_end-effector")*3+3])
    else:
        #log.warning("Warning: Acceleration Sensor not found in mujoco model. Sensor name should be 'ee_acc' or 'accsite_end-effector'. Setting acc to 0.")
        xacc = np.zeros(sim.model.nv)

    return xpos, xvel, xacc

# linear transferfunction F = k
def lin_transfer(F, sim, simulator):
    xpos = np.array(sim.data.get_body_xpos(simulator.param_dict["end_effector"]))
    xvel = np.array(sim.data.get_body_xvelp(simulator.param_dict["end_effector"]))
    
    return F*xpos, F*xvel, np.zeros((3,))

# affine transferfunction f(x) = F+x
def affine_transfer(F, sim, simulator):
    xpos = np.array(sim.data.get_body_xpos(simulator.param_dict["end_effector"]))
    xvel = np.array(sim.data.get_body_xvelp(simulator.param_dict["end_effector"]))
    
    return xpos+F, xvel, np.zeros((3,))

# nonlinear transferfunction F = k
def nonlin_transfer(F, sim, simulator):
    xpos = np.array(sim.data.get_body_xpos(simulator.param_dict["end_effector"]))
    xvel = np.array(sim.data.get_body_xvelp(simulator.param_dict["end_effector"]))
    
    return xpos + sim.model.opt.timestep * F[0] * xvel**F[1],  F[0] * xvel**F[1]

# Outputspace transfer (no rotation)
def outputspace_transfer(F, sim, simulator):
    # Global position/velocity of the end_effector
    xpos_g = np.array(sim.data.get_body_xpos(simulator.param_dict["end_effector"]))
    xvel_g = np.array(sim.data.get_body_xvelp(simulator.param_dict["end_effector"]))
    
    origin_i = F[0] # Inputspace origin
    origin_o = F[1] # Outputspace origin
    transferdilation = F[2] # Dilation applied on the inputspace vector before transferation to outputspace
    
    # Inputspace position/velocity of the end_effector
    xpos_i = xpos_g - origin_i
    xvel_i = xvel_g
    
    xpos_i_d = transferdilation*xpos_i
    
    # Outputspace position/velocity of the end_effector
    xpos_o = xpos_i_d + origin_o
    xvel_o = transferdilation*xvel_i
    
    return xpos_o, xvel_o, np.zeros((3,))

# Projects x onto plane given by point p and normal n
def proj(x, p, n):
    v = x-p
    n = n/np.linalg.norm(n)
    dist = np.dot(v,n)
    projected_point = x - dist*n
    return projected_point

# Rotates and transfers point on plane (p,n) onto plane (p_o,n_o)
def rot_trans(point, p, n, p_o, n_o):
    transpoint = point - p 
    rotpoint = rot(transpoint, n, n_o)
    return rotpoint + p_o
 
# rotates vector from plane with normal n into rotation of plane with normal n_o
def rot(vec, n, n_o):
    # check whether vector already has correct rotation
    if np.linalg.norm(n - n_o) > 1e-12:
        # rotate
        c = np.dot(n,n_o)/((np.linalg.norm(n)*np.linalg.norm(n_o)))

        cross = np.cross(n, n_o)
        axis = cross/ np.linalg.norm(cross)

        s = np.sqrt(1-c*c)
        C = 1-c
        x = axis[0]
        y = axis[1]
        z = axis[2]
        rmat = np.array([[ x*x*C+c,    x*y*C-z*s,  x*z*C+y*s ],
                      [ y*x*C+z*s,  y*y*C+c,    y*z*C-x*s ],
                      [ z*x*C-y*s,  z*y*C+x*s,  z*z*C
                       
                       +c   ]])

        return rmat@vec

    else:
        # don't rotate
        return vec

# Plane transfer
def plane_transfer(F, sim, simulator):

    # Global position/velocity of the end_effector
    xpos_g = np.array(sim.data.get_body_xpos(simulator.param_dict["end_effector"]))
    xvel_g = np.array(sim.data.get_body_xvelp(simulator.param_dict["end_effector"]))
    
    p = F[0] # center projection plane
    n = F[1] # normal vector projection plane
    p_o = F[2] # center output plane
    n_o = F[3] # normal vector output plane
    
    n = n/np.linalg.norm(n)
    n_o = n_o/np.linalg.norm(n_o)
    
    # Project end_effector pos on plane
    y = proj(xpos_g, p, n)
    y_vel = xvel_g - n * np.dot(xvel_g, n)
    
    # Rotate and translate point to output plane
    xpos_o = rot_trans(y, p, n, p_o, n_o)
    xvel_o = rot(y_vel, n, n_o)

    return xpos_o, xvel_o, np.zeros((3,))
    #xpos_o = rot(y, n, n_o)
    
# Projection transfer
def proj_transfer(F, sim, simulator):

     # Global position/velocity of the end_effector
    xpos_g = np.array(sim.data.get_body_xpos(simulator.param_dict["end_effector"]))
    xvel_g = np.array(sim.data.get_body_xvelp(simulator.param_dict["end_effector"]))
    
    p = F[0] # center projection plane
    n = F[1] # normal vector projection plane
    
    n = n/np.linalg.norm(n)
    
    # Project end_effector pos on plane
    y = proj(xpos_g, p, n)
    y_vel = xvel_g - n * np.dot(xvel_g, n)
    
    return y, y_vel, np.zeros((3,))
    #xpos_o = rot(y, n, n_o)