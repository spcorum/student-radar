import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import re
import os
import gc



#def load_weights(gan, weights_path):
#    gan.built = True
#    gan.load_weights(weights_path)
#
#    timestamp = re.findall('\d+\.\d+', weights_path)[0]
#    x = re.findall('epoch-\d.', weights_path)[0]
#    initial_epoch = int(re.findall('\d+', x)[0])
#
#    return gan, initial_epoch, timestamp


def load_weights(gan, weights_path):
    gan.built = True
    gan.load_weights(weights_path)

    return gan
    

#def load_dataset(path, batch_size):
#    data = np.load(path)
#    tensor = tf.convert_to_tensor(data)
#    dataset = tf.data.Dataset.from_tensor_slices(tensor)
#    del data
#    del tensor
#    gc.collect()
#
#    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)
#    print(f'\nNumber of batches: {len(dataset)}\n')
#
#    return dataset

def load_dataset(path, batch_size, val_split=0.2, shuffle=False):
    data = np.load(path)
    n = data.shape[0]
    split = int(n * val_split)

    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)

    val_idx = idx[:split]
    train_idx = idx[split:]

    train_data = data[train_idx]
    val_data = data[val_idx]

    train_tensor = tf.convert_to_tensor(train_data)
    val_tensor = tf.convert_to_tensor(val_data)

    train_dataset = tf.data.Dataset.from_tensor_slices(train_tensor)
    val_dataset = tf.data.Dataset.from_tensor_slices(val_tensor)

    del data, train_data, val_data, train_tensor, val_tensor
    gc.collect()

    train_dataset = train_dataset.batch(batch_size, drop_remainder=True).prefetch(1)
    val_dataset = val_dataset.batch(batch_size, drop_remainder=True).prefetch(1)

    print(f'\nTrain batches: {len(train_dataset)} | Val batches: {len(val_dataset)}\n')

    return train_dataset, val_dataset


# def load_dataset_labaled(path_data, path_label, batch_size):
#     data = np.load(path_data)
#     labels = np.load(path_label)
#     labels_generator = np.asmatrix(labels[0])
#     labels_discriminator = np.asmatrix(labels[1])

#     data_tensor = tf.convert_to_tensor(data)
#     labels_generator_tensor = tf.convert_to_tensor(labels_generator)
#     labels_discriminator_tensor = tf.convert_to_tensor(labels_discriminator)
#     del data
#     del labels
#     del labels_discriminator
#     del labels_generator
#     gc.collect()

#     dataset = tf.data.Dataset.from_tensor_slices((data_tensor, labels_generator_tensor, labels_discriminator_tensor))
#     del data_tensor
#     del labels_generator_tensor
#     del labels_discriminator_tensor
#     gc.collect()

#     dataset = dataset.shuffle(10_000)
#     dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)
#     print(f'\nNumber of batches: {len(dataset)}')

#     return dataset

def load_dataset_labeled(path_data, path_label, batch_size, val_split=0.2, shuffle=False):
    data = np.load(path_data)
    labels = np.load(path_label)

    labels_generator = np.asmatrix(labels[0])
    labels_discriminator = np.asmatrix(labels[1])
    del labels
    gc.collect()

    # Split indices
    num_samples = data.shape[0]
    split = int(num_samples * val_split)
    idx = np.arange(n)
    
    if shuffle:
        np.random.shuffle(idx)

    val_idx = idx[:split]
    train_idx = idx[split:]

    # Split data and labels
    train_data = data[train_idx]
    val_data = data[val_idx]

    train_labels_generator = labels_generator[train_idx]
    val_labels_generator = labels_generator[val_idx]

    train_labels_discriminator = labels_discriminator[train_idx]
    val_labels_discriminator = labels_discriminator[val_idx]

    # Convert to tensors
    train_dataset = tf.data.Dataset.from_tensor_slices((
        tf.convert_to_tensor(train_data),
        tf.convert_to_tensor(train_labels_generator),
        tf.convert_to_tensor(train_labels_discriminator),
    ))

    val_dataset = tf.data.Dataset.from_tensor_slices((
        tf.convert_to_tensor(val_data),
        tf.convert_to_tensor(val_labels_generator),
        tf.convert_to_tensor(val_labels_discriminator),
    ))

    # Cleanup
    del data
    del labels_generator
    del labels_discriminator
    del train_data, val_data
    del train_labels_generator, val_labels_generator
    del train_labels_discriminator, val_labels_discriminator
    gc.collect()

    train_dataset = train_dataset.shuffle(10_000)
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True).prefetch(1)
    val_dataset = val_dataset.batch(batch_size, drop_remainder=True).prefetch(1)

    print(f'\nTrain batches: {len(train_dataset)} | Val batches: {len(val_dataset)}')

    return train_dataset, val_dataset