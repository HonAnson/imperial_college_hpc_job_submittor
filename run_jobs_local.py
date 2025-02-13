import toml
from pathlib import Path
import subprocess
from datetime import datetime
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor, as_completed

def dict_to_namespace(d):
    """Recursively convert the dictionary to a SimpleNamespace."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    else:
        return d

def run_job(config_path, output_directory):
    """Function to run a single job."""
    config = dict_to_namespace(toml.load(config_path))
    job_name = config.data_io.job_name

    # Define log file
    log_filename = output_directory / f"log_{job_name}.txt"

    # Construct command to run train.py
    command = [
        "python", "./src/train.py", "-i", str(config_path)
    ]

    print(f"Starting job: {job_name} (Logging to {log_filename})")

    # Run the command and redirect output to log file
    with open(log_filename, "w") as log_file:
        process = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, check=True)

    print(f"Completed job: {job_name}")
    return job_name

def main(max_jobs=1):
    date = datetime.today().strftime("%Y%m%d")

    # List all configuration files
    config_directory = Path("./configs/staging")
    config_paths = [file.resolve() for file in config_directory.glob("*.toml")]

    jobs = []
    for config_path in config_paths:
        config = dict_to_namespace(toml.load(config_path))
        job_name = config.data_io.job_name

        # Define output directory
        output_directory = Path(config.data_io.output_dir) / job_name
        output_directory.mkdir(parents=True, exist_ok=True)

        # Move configuration file to the processed directory
        new_config_path = output_directory / config_path.name
        subprocess.run(["mv", config_path, output_directory], check=True)
        print(f"Moved configuration file to {output_directory}")

        # Store job details
        jobs.append((new_config_path, output_directory))

    # Run jobs with limited concurrency
    with ProcessPoolExecutor(max_workers=max_jobs) as executor:
        future_to_job = {executor.submit(run_job, cfg_path, out_dir): (cfg_path, out_dir) for cfg_path, out_dir in jobs}

        for future in as_completed(future_to_job):
            cfg_path, out_dir = future_to_job[future]
            try:
                result = future.result()  # Get result to catch exceptions
                print(f"Job {result} completed successfully.")
            except Exception as exc:
                print(f"Job {cfg_path.name} failed with error: {exc}")

    print("All jobs completed.")

if __name__ == "__main__":
    max_jobs = 6  # Adjust this number to control parallel jobs
    main(max_jobs=max_jobs)
