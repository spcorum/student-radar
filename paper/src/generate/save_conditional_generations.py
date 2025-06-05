from imageio import save
import tensorflow as tf
import numpy as np
import sys
import os 
import time
import argparse
import glob
from utils import load_weights

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../../'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src/models/hyperparemeters'))

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.random.set_seed(1729)


def reshape_generations(generated):
    return tf.transpose(generated, [0, 2, 1])


def denormalize(generated, data_min, data_max, a, b):
    return data_min + (generated - a) * (data_max - data_min) / (b - a)

def get_checkpoint_path(working_dir, model_type, model_id=None, epoch=-1):
    base_dir = os.path.join(working_dir, "student-radar", "paper", "checkpoints", model_type)

    # If model_id is not provided, find the most recently modified model directory
    if model_id is None:
        subdirs = [d for d in glob.glob(os.path.join(base_dir, "model-*")) if os.path.isdir(d)]
        if not subdirs:
            raise FileNotFoundError(f"No model directories found in: {base_dir}")
        latest_dir = max(subdirs, key=os.path.getmtime)
        print(f"[INFO] Using most recent model directory: {latest_dir}")
    else:
        latest_dir = os.path.join(base_dir, f"model-{model_id}")
        if not os.path.exists(latest_dir):
            raise FileNotFoundError(f"Model directory not found: {latest_dir}")

    model_id = os.path.basename(latest_dir).replace("model-", "")

    if epoch == -1:
        # Automatically get the latest checkpoint
        pattern = os.path.join(latest_dir, f"model-{model_id}-epoch-*.weights.h5")
        weight_files = sorted(
            glob.glob(pattern),
            key=lambda x: int(x.split("-epoch-")[-1].split(".")[0])
        )
        if not weight_files:
            raise FileNotFoundError(f"No checkpoints found in: {latest_dir}")
        latest_path = weight_files[-1]
        print(f"[INFO] Using latest checkpoint: {latest_path}")
        return latest_path, model_id
    else:
        # Construct the exact path for the given epoch
        weight_path = os.path.join(latest_dir, f"model-{model_id}-epoch-{epoch}.weights.h5")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Checkpoint not found: {weight_path}")
        print(f"[INFO] Using checkpoint: {weight_path}")
        return weight_path, model_id


def save_conditional_generations(working_dir, model_type, model, epoch, name, num_generations=6000):

    if name is None:
        raise ValueError("You must specify an --output_name to save the output.")

    # Add model paths
    hyperparam_path = os.path.join(working_dir, 'src/models/hyperparemeters')
    sys.path.append(hyperparam_path)

    # Import correct model
    if model_type == "student":
        from model_cwgangp import build_gan
    elif model_type == "original":
        from model_cwgangp_original import build_gan
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Build and load weights
    gan, _, _, _ = build_gan()

    #checkpoint_dir = os.path.join(working_dir, f'checkpoints/{model_type}/model-{model}')
    
    # weights_path = get_checkpoint_path(working_dir, model_type, model, epoch)
    # gan = load_weights(gan, weights_path)
    # generator = gan.generator

    # Load weights and resolved model_id
    weights_path, model_id = get_checkpoint_path(working_dir, model_type, model, epoch)
    gan = load_weights(gan, weights_path)
    generator = gan.generator

    print(f"[INFO] Loading weights from: {weights_path}")

    # Create input noise + conditional labels
    # distances = np.linspace(25.0, 0.0, num_generations).reshape((num_generations, 1))
    distances = np.random.uniform(0.0, 25.0, num_generations).reshape((num_generations, 1))
    distances_scaled = (distances - 10.981254577636719) / 7.1911773681640625
    noise = tf.random.normal(shape=[num_generations, 100])

    # Generate
    start_time = time.time()
    if model_type == "student":
        generated = generator.predict([noise, distances_scaled], verbose=0)
    else: # original
        gen_input = tf.concat([noise, distances_scaled], axis=1)
        generated = generator.predict(gen_input, verbose=0)
    generated = reshape_generations(generated)
    #generated = denormalize(generated, -3884.0, 4772.0, -1, 1)
    #generated = np.round(generated, 0)
    print(f"[INFO] Generated data shape: {generated.shape}")
    print("Generation time: %.2f seconds" % (time.time() - start_time))

    # Save output to working_dir/data/generated/<model_type>/model-<model_id>/
    output_dir = os.path.join(working_dir, 'student-radar', 'paper', 'data', 'generated', model_type, f'model-{model_id}', name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[Output directory] {output_dir}")
    # output_path = os.path.join(output_dir, f'{name}.npy')
    # np.save(output_path, generated)
    # print(f"[Saved] {output_path}")
      
    # return output_path

    output_path_generations = os.path.join(output_dir, 'generator_data.npy')
    output_path_labels = os.path.join(output_dir, 'generator_labels.npy')
    print(f"[Output generations file] {output_path_generations}")
    print(f"[Output labels file] {output_path_generations}")
    np.save(output_path_generations, generated, allow_pickle=True)
    np.save(output_path_labels, distances_scaled, allow_pickle=True)
    print(f"[Saved] {output_path_generations}")
    print(f"[Saved] {output_path_labels}")

    return [output_path_generations, output_path_labels]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Model identifier (e.g., wandb run ID)", default = None)
    parser.add_argument("--output_name", help="Output filename (without extension)")
    parser.add_argument("--num_generations", type=int, default=6000, help="Number of samples to generate")
    parser.add_argument('--model_type', choices=['student', 'original'], default='student')
    parser.add_argument("--epoch", type=int, help="Epoch number to load weights from", default = -1)
    parser.add_argument('--working_dir', type=str, default='/content/drive/MyDrive/cs231n/', help='Base working directory')
    args = parser.parse_args()

    save_conditional_generations(
        working_dir=args.working_dir,
        model_type=args.model_type,
        model=args.model,
        epoch=args.epoch,
        name=args.output_name,
        num_generations=args.num_generations,
    )


if __name__ == "__main__":
    main()