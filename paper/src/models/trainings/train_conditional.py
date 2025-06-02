import os
import sys
import gc
import numpy as np
import tensorflow as tf
import wandb
from tensorflow.keras.callbacks import ModelCheckpoint

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

def load_dataset_labeled(path_data, path_label, batch_size, validation_split=0.2, seed = 1729):
    print(path_data)

    data = np.load(path_data)
    labels_generator, labels_discriminator = np.load(path_label, allow_pickle=True)

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

def main():
    gan, batch_size, epochs, gan_config = build_gan()

    wandb.init(
        project='student-generative-radar',
        entity='spcorum',
        name='conditional-radar-gan-student',
        config=gan_config,
        resume='allow',  # enable if resuming run
        # id="19fh3wpc"
    )

    if wandb.run.resumed:
        print('Resuming Run...')
        model_file = wandb.restore(f'model-{wandb.run.id}.h5').name
        gan = load_weights(gan, model_file)
    
    os.chdir('/content/drive/MyDrive/student-radar/paper')

    checkpoint_file = f'model-{wandb.run.id}.h5'

    initial_epoch = 0
    if wandb.run.resumed:
        print('Resuming Run...')
        restored = wandb.restore(checkpoint_file)
        if restored:
            gan = load_weights(gan, restored.name)
            initial_epoch = wandb.config.get("last_epoch", 0)

    #dataset = load_dataset_labaled(
    #    './data/preprocessed/EXP_17_M_chirps_scaled.npy',
    #    './data/preprocessed/EXP_17_M_chirps_labels.npy',
    #    batch_size
    #)

    train_dataset, val_dataset = load_dataset_labeled(
        './data/preprocessed/EXP_17_M_chirps_scaled.npy',
        './data/preprocessed/EXP_17_M_chirps_labels.npy',
        batch_size)

    print(f'\n\n--------------------- Run: {wandb.run.name} ---------------------------\n\n')

    gen_loss_cb_train = ModelCheckpoint(
        filepath=f'model-{wandb.run.id}.h5',
        save_weights_only=True,
        save_best_only=True,
        monitor='generator_loss',
        mode='min'
    )

    disc_loss_cb_train = ModelCheckpoint(
        filepath=f'model-{wandb.run.id}.h5',
        save_weights_only=True,
        save_best_only=True,
        monitor='discriminator_loss',
        mode='min'
    )

    gen_loss_cb = ModelCheckpoint(
        filepath=f'model-{wandb.run.id}.h5',
        save_weights_only=True,
        save_best_only=True,
        monitor='val_generator_loss',
        mode='min'
    )

    disc_loss_cb = ModelCheckpoint(
        filepath=f'model-{wandb.run.id}.h5',
        save_weights_only=True,
        save_best_only=True,
        monitor='val_discriminator_loss',
        mode='min'
    )

    # extract a sample of real data for passing into logging metrix
    for batch in train_dataset.take(1):
        real_sample, _, _ = batch  # (data, gen_labels, disc_labels)

    gan.fit(
        train_dataset,
        validation_data=val_dataset,
        initial_epoch=initial_epoch,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[gen_loss_cb_train, disc_loss_cb_train, gen_loss_cb, disc_loss_cb, WandbCallbackGANConditional(wandb_module=wandb, real_sample=real_sample)]
    )

if __name__ == '__main__':
    main()
