import os
import sys
import gc
import numpy as np
import tensorflow as tf
import wandb
import argparse
from tensorflow.keras.callbacks import ModelCheckpoint
import glob
import time
from wandb.errors import CommError, Error as WandbError

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CALLBACKS_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'callbacks'))
TRAINERS_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'trainers'))
HYPERPARAMS_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'hyperparemeters'))

sys.path.insert(0, CALLBACKS_PATH)
sys.path.insert(0, TRAINERS_PATH)
sys.path.insert(0, HYPERPARAMS_PATH)

from callback_conditional import WandbCallbackGANConditional
# # Append custom module paths
# sys.path.append('models/callbacks/')
# sys.path.append('models/hyperparemeters/')

# Modern GPU setup
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)  # still needed
    except:
        pass
else:
    print("No GPU device found. Using CPU.")

def load_weights(gan, weights_path):
    gan.built = True
    gan.load_weights(weights_path)
    return gan

# def load_dataset_labaled(path_data, path_label, batch_size):
#     print(path_data) 
#     data = np.load(path_data)
#     labels_generator, labels_discriminator = np.load(path_label, allow_pickle=True)

#     dataset = tf.data.Dataset.from_tensor_slices((
#         tf.convert_to_tensor(data),
#         tf.convert_to_tensor(labels_generator),
#         tf.convert_to_tensor(labels_discriminator)
#     ))

#     del data, labels_generator, labels_discriminator
#     gc.collect()

#     dataset = dataset.shuffle(10_000)
#     dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
#     print(f'\nNumber of batches: {len(dataset)}')

#     return dataset

def load_dataset_labeled(path_data, path_label, batch_size, validation_split=0.2, seed = 1729):
    print(path_data)

    data = np.load(path_data)
    labels_generator, labels_discriminator = np.load(path_label, allow_pickle=True)

    assert len(data) == len(labels_generator) == len(labels_discriminator), \
    f"Mismatch in lengths: data={len(data)}, labels_generator={len(labels_generator)}, labels_discriminator={len(labels_discriminator)}"

    num_samples = len(data)
    split_idx = int(num_samples * (1 - validation_split))

    # Shuffle before splitting
    np.random.seed(seed)
    indices = np.random.permutation(num_samples)
    data = data[indices]
    labels_generator = labels_generator[indices]
    labels_discriminator = labels_discriminator[indices]

    # Train split
    train_dataset = tf.data.Dataset.from_tensor_slices((
        tf.convert_to_tensor(data[:split_idx]),
        tf.convert_to_tensor(labels_generator[:split_idx]),
        tf.convert_to_tensor(labels_discriminator[:split_idx])
    ))

    # Validation split
    val_dataset = tf.data.Dataset.from_tensor_slices((
        tf.convert_to_tensor(data[split_idx:]),
        tf.convert_to_tensor(labels_generator[split_idx:]),
        tf.convert_to_tensor(labels_discriminator[split_idx:])
    ))

    # Free up memory
    del data, labels_generator, labels_discriminator
    gc.collect()

    train_dataset = train_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    print(f'\nTrain batches: {len(train_dataset)} | Validation batches: {len(val_dataset)}')

    return train_dataset, val_dataset

def safe_wandb_init(project, entity, config, model_type, run_id=None):
    """
    Safely initialize a wandb run, resuming if run_id exists, or starting fresh otherwise.
    """
    resume = 'allow'  # default to allow new run
    name = f'conditional-radar-gan-{model_type}'

    if run_id:
        try:
            wandb.Api().run(f"{entity}/{project}/{run_id}")
            resume = 'must'
            name = None  # don't override name if resuming
            print(f"[INFO] Resuming existing run: {run_id}")
        except WandbError:
            print(f"[WARN] Run ID '{run_id}' not found on WandB. Starting a new run.")
            run_id = None
            resume = 'allow'
            name = f'conditional-radar-gan-{model_type}'

    return wandb.init(
        project=project,
        entity=entity,
        id=run_id,
        resume=resume,
        name=name,
        config={
            **config,
            "model_type": model_type
        },
        settings=wandb.Settings(init_timeout=180)
    )

def main(model_type="student", num_epochs = 10, batch_size = 32, working_dir="/content/drive/MyDrive/cs231n/", model_name=None):

    start_time = time.time()

    if model_type == "student":
        from model_cwgangp import build_gan
    elif model_type == "original":
        from model_cwgangp_original import build_gan
    else:
        raise ValueError("model_type must be 'student' or 'original'")
    
    # Build the model
    gan, batch_size, epochs, gan_config = build_gan(num_epochs = num_epochs, batch_size = batch_size)
    print(f"[INFO] GAN model successfully built")

    # Connect to WandB
    project = 'student-generative-radar'
    entity = 'spcorum-aerostar-international'
    config={
            **gan_config,
            "model_type": model_type
    }
    safe_wandb_init(project,entity, config, model_type, run_id=model_name)

    # Define absolute checkpoint directory without changing working directory
    checkpoint_dir = os.path.join(working_dir, "student-radar", "paper", "checkpoints", model_type)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, f"model-{wandb.run.id}.weights.h5")

    # Load from checkpoint if resuming
    initial_epoch = 0
    if wandb.run.resumed:
        print('[INFO] Attempting to resume from last checkpoint...')
        pattern = os.path.join(checkpoint_dir, f"model-{wandb.run.id}", f"model-{wandb.run.id}-epoch-*.weights.h5")
        # Find the weight from the youngest epoch
        weight_files = sorted(
            glob.glob(pattern),
            key=lambda x: int(x.split("-epoch-")[-1].split(".")[0])
        )
        if weight_files:
            latest_weights = weight_files[-1]
            print(f"[INFO] WandB Run URL: https://wandb.ai/{entity}/{project}/runs/{wandb.run.id}")
            print(f"[INFO] Resuming from: {latest_weights}")
            gan = load_weights(gan, latest_weights)
            print(f"[INFO] GAN successfully built from from: {latest_weights}")
            try:
                initial_epoch = int(latest_weights.split("-epoch-")[-1].split(".")[0])
            except ValueError:
                print("[WARN] Could not parse epoch from filename, starting from scratch.")
        else:
            print("[INFO] No previous checkpoints found, starting from scratch.")

    data_path = os.path.join(working_dir, "student-radar", "paper", "data", "preprocessed", "EXP_17_M_chirps_scaled_train.npy")
    label_path = os.path.join(working_dir, "student-radar", "paper", "data", "preprocessed", "EXP_17_M_chirps_labels_train.npy")
    train_dataset, val_dataset = load_dataset_labeled(data_path,label_path,batch_size)

    print(f'\n\n--------------------- Run: {wandb.run.name} ---------------------------\n\n')

    # Force building the gan model before training to avoid save_weights error
    try:
        dummy_noise = tf.random.normal((1, gan.latent_dim))
        dummy_label = tf.zeros((1, 1))
        gan([dummy_noise, dummy_label], training=False)  # Force model build
        if model_type == 'student':
            print("[INFO] StudentGAN model successfully built before training.")
        else: # orignal
            print("[INFO] OriginalGAN model successfully built before training.")
    except Exception as e:
        print(f"[ERROR] Failed to build model before training: {e}")

    # extract a sample of real data for passing into logging metrics
    for batch in train_dataset.take(1):
        real_sample, _, _ = batch  # (data, gen_labels, disc_labels) f

    # Define WandB call back and save model to it
    wandb_callback = WandbCallbackGANConditional(
        wandb_module=wandb,
        real_sample=real_sample,
        model_type=model_type,
        save_interval=10
    )
    wandb_callback.set_model(gan)

    print(f"[INFO] Starting training from epoch {initial_epoch}...")
    print(f"[INFO] WandB run ID: {wandb.run.id}")
    if wandb.run.resumed and weight_files:
        print(f"[INFO] Loaded checkpoint from: {latest_weights}")
    else:
        print("[INFO] No checkpoint loaded; training will start from scratch.")

    gan.fit(
        train_dataset,
        validation_data=val_dataset,
        initial_epoch=initial_epoch,
        epochs=epochs,
        batch_size=batch_size,
        callbacks = [wandb_callback]
    )

    total_time = time.time() - start_time
    print(f"\n[INFO] Total training time: {total_time:.2f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', choices=['student', 'original'], default='student')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--working_dir', type=str, default='/content/drive/MyDrive/cs231n/', help='Base working directory')
    parser.add_argument('--model_name', type=str, default=None, help='WandB run ID to resume from')
    args = parser.parse_args()
    
    main(model_type=args.model_type, num_epochs=args.epochs, batch_size=args.batch_size, working_dir = args.working_dir, model_name = args.model_name)
