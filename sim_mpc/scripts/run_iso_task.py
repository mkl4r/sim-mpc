''' 
Script to run an example ISO type pointing task for one user model and one condition.

Authors: Markus Klar
Date: 11.2022
'''
import sim_mpc.scripts.iso_task as iso

if __name__ == "__main__":
    # Available conditions, participants and costfunctions
    CONDITIONS = ['Virtual_Cursor_ID_ISO_15_plane', 'Virtual_Cursor_Ergonomic_ISO_15_plane', 'Virtual_Pad_ID_ISO_15_plane', 'Virtual_Pad_Ergonomic_ISO_15_plane']
    PARTICIPANTS = ['U1', 'U2', 'U3', 'U4', 'U5', 'U6']
    COSTFCTS = ["DC","CTC","JAC"]

    participant = PARTICIPANTS[0]
    condition = CONDITIONS[0]
    costfct = COSTFCTS[2]
        
    iso.run_iso_task(participant, condition, costfct, paramfolder="../data/parameters", outputprefix='fitts', createvideos=True, verbosity=True)           
        