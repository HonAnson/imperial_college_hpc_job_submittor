import os
from pathlib import Path
import subprocess
import time
from datetime import datetime
from types import SimpleNamespace
import toml

def dict_to_namespace(d):
    '''Recursively convert the dictionary to a SimpleNamespace'''
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    else:
        return d

date = datetime.today().strftime("%Y%m%d")

# List of configurations file to be used
config_directory = Path("./configs/staging")
config_paths = [file.resolve() for file in config_directory.glob("*.toml")]

# Template for the PBS script
pbs_template_cx3_gpu = """#PBS -l select=1:ncpus=4:mem=64gb:ngpus=1
#PBS -l walltime=12:00:00

# Navigate to the project directory
cd $HOME/neural_lidar

eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate neural_lidar
module load CUDA/12.1

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "PATH: $PATH"
echo "which nvcc: $(which nvcc)"
echo "nvcc version: $(nvcc --version)"

python ./src/train.py -i {config_path} > {pbs_directory}/{log}
"""

pbs_template_cx3_cpu = """#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=16:mem=24gb

# Navigate to the project directory
cd $HOME/neural_lidar

eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate neural_lidar

echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "PATH: $PATH"

python ./src/train.py -i {config_path} > {pbs_directory}/{log}
"""

# Directory to store generated PBS scripts


for config_path in config_paths:
    config = dict_to_namespace(toml.load(config_path))
    job_name = config.data_io.job_name

    # Define the PBS directory using Pathlib
    pbs_directory = Path(f"./jobs/{date}/{job_name}")
    output_directory = Path(config.data_io.output_dir)/ job_name
    pbs_directory.mkdir(parents=True, exist_ok=True)  # Create directories if not exist
    output_directory.mkdir(parents=True, exist_ok=True)  # Create directories if not exist

    # Move configuration file from staging to processed
    subprocess.run(["mv", config_path, output_directory], check=True)  # Ensure paths are strings

    # the new configuration path is:
    new_config_path = output_directory / config_path.name

    # Define PBS script and log filenames
    pbs_filename = pbs_directory / f"{job_name}.pbs"
    log_filename = f"log_{job_name}.txt"

    # Write the PBS script
    with open(pbs_filename, 'w') as pbs_file:
        pbs_file.write(pbs_template_cx3_gpu.format(
            config_path=str(new_config_path),  # Convert to string for safe formatting
            log=log_filename,
            pbs_directory=str(pbs_directory)  # Convert to string for safe formatting
        ))

    # Submit the PBS script from the correct directory
    subprocess.run(["qsub", pbs_filename.name], cwd=pbs_directory, check=True)  # Set working directory
    time.sleep(1)


print("All PBS scripts generated in this batch are submitted.")


