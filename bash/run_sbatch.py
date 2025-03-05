import os
import subprocess
import re
import time


LIST_OF_JOBS_NAME = ["run_exp_evolve_150k.sh", "run_exp_evolve_120k.sh", "run_exp_evolve_90k.sh", "run_exp_evolve_60k.sh", "run_exp_evolve_150k_tdg.sh", "run_exp_evolve_120k_tdg.sh", "run_exp_evolve_90k_tdg.sh", "run_exp_evolve_60k_tdg.sh"]

if __name__ == '__main__':
    # Poll the job status using squeue
    while True:
        result = subprocess.run(['squeue', "--format='%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R' --me"], capture_output=True, text=True)
        # print(result.stdout)
        number_of_running_jobs = 0
        for job_name in LIST_OF_JOBS_NAME:
            if job_name in result.stdout:
                number_of_running_jobs += 1
        print(f"Number of running jobs: {number_of_running_jobs}")
        
        if number_of_running_jobs == 0:
            # run run_sbatch.sh
            results = subprocess.run(['bash', 'run_sbatch.sh'], capture_output=True, text=True)
            print("Submitted new jobs.")

        time.sleep(10)  # Wait for 10 seconds before polling again