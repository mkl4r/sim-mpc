
''' 
Script to calculate the initial activations on a slurm cluster.

Authors: Florian Fischer
Date: 11.2022
'''
import os
from pathlib import Path
import logging
import cfat
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

    job_directory = f"{os.getcwd()}/../_slurm/jobs"
    if not os.path.exists(job_directory):
        os.makedirs(job_directory, exist_ok=True)
    output_directory = f"{os.getcwd()}/../_slurm/out"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)
    error_directory = f"{os.getcwd()}/../_slurm/err"
    if not os.path.exists(error_directory):
        os.makedirs(error_directory, exist_ok=True)

    use_mujoco_py = True  # whether to use mujoco-py or mujoco

    for trial_id, (username, table_filename) in enumerate(filelist):
        print(f'\nCOMPUTING FEASIBLE CONTROLS for {table_filename}.')

        job_file = os.path.join(job_directory, f"{trial_id}_initacts.job")
        with open(job_file, 'w+') as fh:
            fh.writelines("#!/bin/bash\n")
            fh.writelines(f"#SBATCH -J {table_filename.split('/')[-1]}\n")
            fh.writelines(f"#SBATCH -N 1 # (Number of requested nodes)\n")
            # fh.writelines(f"#SBATCH --ntasks-per-node 20 # (Number of requested cores per node)\n")
            fh.writelines(f"#SBATCH -t 1:00:00 # (Requested wall time)\n")
            fh.writelines(f"#SBATCH --output={output_directory}/{trial_id}_initacts.out\n")
            fh.writelines(f"#SBATCH --error={error_directory}/{trial_id}_initacts.err\n")
            fh.writelines(
                f"srun python compute_initacts_example.py --username={username} --table_filename={table_filename} {'--mujoco-py' if use_mujoco_py else '--mujoco'}")

        os.system(f"sbatch {job_file}")
        