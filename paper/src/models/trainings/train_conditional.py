import os
import sys
import gc
import numpy as np
import tensorflow as tf
import wandb

# Append custom module paths
sys.path.append('models/callbacks/')
sys.path.append('models/hyperparemeters/')

from callback_conditional import WandbCallbackGANConditional
from model_cwgangp import build_gan

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

def main():
    gan, batch_size, epochs, gan_config = build_gan()

    wandb.init(
        project='radar-synth',
        entity='vimalashekar04',
        name='conditional-nooutiliers',
        config=gan_config,
        # resume='allow',  # enable if resuming run
        # id="19fh3wpc"
    )

    if wandb.run.resumed:
        print('Resuming Run...')
        model_file = wandb.restore(f'model-{wandb.run.id}.h5').name
        gan = load_weights(gan, model_file)
    
    os.chdir('/content/drive/MyDrive/student-radar/paper')

    dataset = load_dataset_labaled(
        './data/preprocessed/EXP_17_M_chirps_scaled.npy',
        './data/preprocessed/EXP_17_M_chirps_labels.npy',
        batch_size
    )

    print(f'\n\n--------------------- Run: {wandb.run.name} ---------------------------\n\n')

    gan.fit(
        dataset,
        initial_epoch=wandb.run.step,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[]
    )

if __name__ == '__main__':
    main()
