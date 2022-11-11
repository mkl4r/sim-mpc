'''
Basic visualization functions to create a simulation video.

Authors: Markus Klar
Date: 11.2022
'''
import logging as log
import os
from datetime import datetime

import imageio
import mujoco_py as mj
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import sim_mpc.core.utils as utils


def add_text_to_image(image, text, font="dejavu/DejaVuSans.ttf", pos=(400, 300), color=(255, 0, 0), fontsize=24):
    draw = ImageDraw.Draw(image)
    draw.text(pos, text, fill=color, font=ImageFont.truetype("/usr/share/fonts/truetype/" + font, fontsize))
    return image

def find_contacts(sim):
    return [(sim.data.contact[i].dim, sim.data.contact[i].geom1, sim.data.contact[i].geom2, sim.model.geom_rgba[sim.data.contact[i].geom1,:], sim.model.geom_rgba[sim.data.contact[i].geom2,:]) for i in range(sim.data.ncon)]

def placeTargets(targets, param):
    for k in range(len(targets)):
        utils.updatebodypos(param, "target{}".format(k+1), targets[k])

def visualize_run(inputfolder, use_diff_model=None, changefilename=None, vis_contacts=True,vis_targets=True,vis_cursor=True,vis_ee=True):
    param_dict = np.load(inputfolder + "/param.npy",allow_pickle='TRUE').item()
  
    if changefilename is not None:
        video_filename = changefilename
    else:
        video_filename = inputfolder + "/video.mp4"
        
    visualize_run_from_csv(inputfolder, param_dict, video_filename, use_diff_model,vis_contacts=vis_contacts,vis_targets=vis_targets,vis_cursor=vis_cursor,vis_ee=vis_ee)
        
def visualize_run_from_csv(inputfolder, param_dict, video_filename, use_diff_model=None, vis_contacts=True,vis_time=True, vis_targets=True,vis_cursor=True,vis_ee=True):

    log.info(f"Visualizing...")

    # Load data     
    joint_cols = [joint + "_pos" for joint in param_dict["joint_names"]]

    complete =  pd.read_csv(inputfolder + '/complete.csv')
    QPOS =  complete[joint_cols].to_numpy()
    XPOS = np.array(complete.loc[:, ['end-effector_xpos_x', 'end-effector_xpos_y', 'end-effector_xpos_z']].to_numpy())

    num_iterations = len(QPOS)

    if use_diff_model is not None:
        model = mj.load_model_from_path(use_diff_model)
    else:
        model = mj.load_model_from_path(param_dict["model_file"])

    model.opt.timestep = param_dict["h"]

    if vis_targets:
        # Update target size
        model.geom_size[model.geom_name2id("target")][0] = param_dict["stoptol_pos"]
    else:
        # Hide target
        model.geom_rgba[model.geom_name2id("target")] = np.array([0,0,0,0])
        
    if not vis_cursor:
        # Hide cursor 
        model.geom_rgba[model.geom_name2id("cursor")] = np.array([0,0,0,0])
    
    if not vis_ee:
        # Hide end-effector 
        model.geom_rgba[model.geom_name2id("end-effector")] = np.array([0,0,0,0])
    
    sim = mj.MjSim(model)
                         
    x = sim.get_state()

    utils.updatebodypos(sim, "target", param_dict["target"])

    # Resolution
    resx = 16*16*7
    resy = 9*16*7

    FPS = 60 # Frames per second
    slowmow = 1 # Slowmotion factor

    framerate = 1/param_dict["h"] # Simulation framerate
    per_frame = framerate//FPS # Number of simulation frames per video frame
    per_frame = per_frame // slowmow

    if per_frame < 1:
        per_frame = 1

    repeatframe = 1

    # Generate frames and create video
    with imageio.get_writer(video_filename, fps=FPS) as video:
        cur_step = 0
        num_frames = 0
        for k in range(num_iterations):

            utils.updatebodypos(sim, "cursor", XPOS[k])

            x.qpos[:] = QPOS[k]
            sim.set_state(x)
            mj.cymj._mj_kinematics(sim.model, sim.data)

            current_time = cur_step * param_dict["h"] + param_dict["init_time"]
            
            if cur_step % per_frame == 0:
                # Change color of geoms with contact to red
                if vis_contacts:
                    contacts = find_contacts(sim)
                    if len(contacts) > 0:
                        log.debug("Contact!")
                        for c in contacts:
                            log.debug(sim.model.geom_id2name(c[1]), sim.model.geom_id2name(c[2]))
                            
                            sim.model.geom_rgba[c[1],:] = [1, 0, 0, 1]
                            sim.model.geom_rgba[c[2],:] = [1, 0, 0, 1]

                # Render image from MuJoCo simulation
                cur_img = sim.render(width=resx, height=resy, camera_name="mycamera")
                image_obj = Image.frombytes('RGB', (resx,resy), cur_img, 'raw')

                # Reset geom colors
                if vis_contacts:
                    if len(contacts) > 0:
                        for c in contacts:
                            sim.model.geom_rgba[c[1],:] = c[3]
                            sim.model.geom_rgba[c[2],:] = c[4]

                rotated_image = image_obj.transpose(Image.FLIP_TOP_BOTTOM)

                if vis_time:
                    # Add time
                    image_final = add_text_to_image(rotated_image, "{:%M:%S.%f}".format(datetime.fromtimestamp(current_time))[:-3], pos=(10, 10), color=(8, 131, 208)) 
                else:
                    image_final = rotated_image
                
                for _ in range(repeatframe):
                    video.append_data(np.array(image_final))
                    num_frames += 1
            
            cur_step += 1

    log.info(f"Video generated and saved to {os.path.abspath(video_filename)}.")