from datetime import datetime
import os
import toml
import itertools
import numpy as np
from dataclasses import dataclass
from pathlib import Path

@dataclass
class JobArguments:
    scaling_factor: float 
    max_scalar: float 

    ### Data io ###
    input_dir_raw: str
    input_dir_pseudo: str
    output_dir: str
    batch_name: str

    ### Preparation parameters ###
    nerf_model_type: str = "density"
    ray_sampling_strategy: str = "ray_shuffle"  # Ray sampling strategy
    num_ray_sample_per_frame: int = 16  # Number of rays sampled per frame
    num_frame_sample_per_iter: int = 4  # Number of frames sampled per iteration
    batch_size: int = 64  # Number of rays sampled per batch

    # Perturbations
    rotation_perturbation_variance: float = 0.02  # Variance of sensor rotation perturbation
    translation_perturbation_variance: float = 0.02  # Variance of sensor translation perturbation

    # Learning rates
    pose_learning_rate: float = 0.00025  # Learning rate of pose layer
    nerf_learning_rate: float = 5e-6  # Learning rate of NeRF model
    nerf_learning_rate_decay: float = 0.7071
    pose_learning_rate_decay: float = 0.7071

    # Positional encoding
    pos_emb_dim: int = 10  # Positional encoding dimension for positions
    dir_emb_dim: int = 4  # Positional encoding dimension for directions
    hidden_dim: int = 512  # Positional encoding dimension

    # BARF parameters
    start_barf_iter: int = 0  # Epoch to start using higher frequency positional encoding
    end_barf_iter: int = 40000  # Epoch to fully use all positional encoding frequencies
    start_learn_pose_L: int = 8  # Starting positional encoding dimension for pose learning

    ### Training settings ###
    num_epoch: int = 4  # Number of epochs for training

    # Pose learning
    learn_rot: bool = True  # Whether to learn rotation
    learn_trans: bool = True  # Whether to learn translation

    # Loss weights
    lambda_T: float = 1.0  # Weight for translation loss
    lambda_h: float = 1.0  # Weight for rotation loss
    lambda_d: float = 1.0  # Weight for depth loss
    lambda_r: float = 1.0  # Weight for additional loss

    # Volume sampling
    num_bins: int = 100  # Number of bins along a ray
    near_plane: float = 0.0  # Near plane distance
    far_plane: float = 3.73  # Far plane distance

    # Targets
    prediction_variance: float = 0.001  # Variance of target ray termination distribution

def generate_toml_config(file_name, file_path, args):
    # Define the configuration data
    config_data = {
        "scaling_factor": args.scaling_factor,
        "max_scalar": args.max_scalar,
        "data_io": {
            "input_dir_raw": args.input_dir_raw,
            "input_dir_pseudo": args.input_dir_pseudo,
            "output_dir": args.output_dir,
            "job_name": file_name,
        },
        "preperation": {
            "nerf_model_type": args.nerf_model_type,
            "ray_sampling_strategy": args.ray_sampling_strategy,
            "num_ray_sample_per_frame": args.num_ray_sample_per_frame,
            "num_frame_sample_per_iter": args.num_frame_sample_per_iter,
            "batch_size": args.batch_size,
            "rotation_perturbation_variance": args.rotation_perturbation_variance,
            "translation_perturbation_variance": args.translation_perturbation_variance,
            "nerf_learning_rate": args.nerf_learning_rate,
            "pose_learning_rate": args.pose_learning_rate,
            "nerf_learning_rate_decay": args.nerf_learning_rate_decay,
            "pose_learning_rate_decay": args.pose_learning_rate_decay,
            "pos_emb_dim": args.pos_emb_dim,
            "dir_emb_dim": args.dir_emb_dim,
            "hidden_dim": args.hidden_dim,
            "start_barf_iter": args.start_barf_iter,
            "end_barf_iter": args.end_barf_iter,
            "start_learn_pose_L": args.start_learn_pose_L,
        },
        "training": {
            "num_epoch": args.num_epoch,
            "learn_rot": args.learn_rot,
            "learn_trans": args.learn_trans,
            "lambda_T": args.lambda_T,
            "lambda_h": args.lambda_h,
            "lambda_d": args.lambda_d,
            "lambda_r": args.lambda_r,
            "num_bins": args.num_bins,
            "near_plane": args.near_plane,
            "far_plane": args.far_plane,
            "prediction_variance": args.prediction_variance,
        },
    }

    # Write the configuration to a TOML file
    if os.path.exists(file_path):
        raise FileExistsError(f"The file '{file_path}' already exists. Please use a different batch name.")

    with open(file_path, 'w') as toml_file:
        date = datetime.today().strftime("%Y%m%d")
        toml_file.write(f"# This configuration is autometically generated on date {date} (YYYYMMDD) \n")
        toml.dump(config_data, toml_file)
        toml_file.write("\n# End of configuration\n")

def generate_multiple_config(file_io, hyperparameters, loss_weights):    
    # load model scale and scalar scale for writing into config file
    input_dir_raw = Path(file_io["input_dir_raw"])
    npz_file = sorted(input_dir_raw.glob("*.npz"))[0]
    data = np.load(npz_file)
    meta_data = {
        "scaling_factor": data['scaling_factor'].item(),
        "max_scalar": data['max_scalar'].item()
    }

    # Generate all combinations of hyperparameters
    hyperparameter_combinations = list(itertools.product(*hyperparameters.values()))    # * is unpacking operator
    loss_weights_combinations = list(itertools.product(*loss_weights.values()))

    # filter out linearly dependent combinations on loss function weights and case where all weights = 0
    linear_independent_set = []
    filtered_loss_weights_combinations = []
    for combo in loss_weights_combinations:
        # skip if all weights are = 0
        if sum(combo) == 0:
            continue
            
        # normalize loss combinations 
        loss_arr = np.array(combo)
        normal_loss = loss_arr/ np.linalg.norm(loss_arr)

        if not any(np.allclose(normal_loss, arr) for arr in linear_independent_set):
            linear_independent_set.append(normal_loss)
            filtered_loss_weights_combinations.append(combo)

    # create combinations of other hyperparameters and loss weights
    all_combinations = [
        tuple(itertools.chain(*combo))  # Flatten each combined tuple
        for combo in itertools.product(hyperparameter_combinations, filtered_loss_weights_combinations)
    ]

    if len(filtered_loss_weights_combinations) == 0:
        all_combinations = hyperparameter_combinations

    # put hyperparameter names and values into a dict
    parameter_names = [key for key in hyperparameters.keys()] + [key for key in loss_weights.keys()]
    named_combinations = [dict(zip(parameter_names, combo)) | file_io | meta_data for combo in all_combinations]

    # use replace values in JobArguments dataclass, while unspecified hyperparameter is kept default
    combination_dataclass_list = [JobArguments(**parameters) for parameters in named_combinations]

    # convert dataclass into .toml file
    for index, args in enumerate(combination_dataclass_list):
        file_name = f'{args.batch_name}_{index:04}'
        file_path = f'./configs/staging/{file_name}.toml'
        generate_toml_config(file_name, file_path, args)
    print(f"Generated {len(combination_dataclass_list)} configurations files.")
    return

if __name__ == "__main__":
    # "11-35-14_quad-hard"
    # "10-37-38_quad-easy"
    # "11-31-35_quad-medium"

    ### Choose data here ###
    dataset = "nuscenes"
    sequence_name = "mini_sample"
    batch_name = "nuscene_initial"

    ### Choose parameters here ###
    hyperparameters = {
        "end_barf_iter": [40000],
        "start_learn_pose_L" : [8],
        "pos_emb_dim": [10],
        "rotation_perturbation_variance": [0.02],
        "translation_perturbation_variance": [0.02],
        "pose_learning_rate": [0.00025]
        
    }

    loss_weights = {
        "lambda_T" : [1],
        "lambda_h" : [1],
        "lambda_d" : [1],
        "lambda_r" : [1],
    }

    #################################
    ########## DO NOT EDIT ##########
    #################################
    date = datetime.today().strftime("%Y%m%d")
    file_io = {
        "input_dir_raw": f"./data/training/{dataset}/{sequence_name}/raw",
        "input_dir_pseudo": f"./data/training/{dataset}/{sequence_name}/pseudo128",
        "output_dir": f"./output/{date}",
        "batch_name": batch_name,
    }
    generate_multiple_config(file_io, hyperparameters, loss_weights)
    ####### END DO NOT EDIT #########
