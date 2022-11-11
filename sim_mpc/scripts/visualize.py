''' 
Script to visualize a single simulation.

Authors: Markus Klar
Date: 11.2022
'''
from sim_mpc.core.visualize_backend import visualize_run
import argparse
import sys   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder")
    parser.add_argument("--modelname", required=False)    
    parser.add_argument("--vis_contacts",default=False)
    parser.add_argument("--vis_targets",default=True)
    args = parser.parse_args()

    folder = args.folder
    modelname = args.modelname
    vis_contacts = bool(args.vis_contacts)
    vis_targets = bool(args.vis_targets)

    if len(sys.argv) < 2:
        print("Please enter the name of a simulation folder as argument --folder")
    else:
        if modelname is not None:
            visualize_run(folder, use_diff_model=modelname, vis_contacts=vis_contacts,vis_targets=vis_targets)
        else:
            visualize_run(folder, vis_contacts=vis_contacts,vis_targets=vis_targets)