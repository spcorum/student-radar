from imageio import save
import tensorflow as tf
import numpy as np
import sys
import os 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../../'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src/models/hyperparemeters'))

from model_cwgangp import build_gan
from utils import load_weights

import os

import time

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tf.random.set_seed(34643)



def reshape_generations(generated):
    return tf.transpose(generated, [0, 2, 1])


def denormalize(generated, data_min, data_max, a, b):
    return data_min + (generated - a) * (data_max - data_min) / (b - a)


def save_conditional_generations(model, epoch, name):
    gan, _, _, _ = build_gan()
    weights_path = os.path.join(PROJECT_ROOT, f'checkpoints/model-{model}/model-{model}-epoch-{epoch}.weights.h5')

    gan = load_weights(gan, weights_path)
    generator = gan.generator
    start_time = time.time()
    distances = np.linspace(25.0, 0.0, 6000).reshape((6000, 1))
    distances_scaled = (distances - 10.981254577636719) / 7.1911773681640625
    
    noise = tf.random.normal(shape=[len(distances), 100])

    noise_and_distances = tf.concat(
        [noise, distances_scaled], axis=1
    )

    generated = reshape_generations(generator.predict([noise, distances_scaled]))
    generated_denormalized = denormalize(generated, -3884.0, 4772.0, -1, 1)
    generated_denormalized = np.round(generated_denormalized, 0)
    print("%s" % (time.time() - start_time))
    
    output_dir = os.path.join(PROJECT_ROOT, 'data/generated')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{name}.npy')
    np.save(output_path, generated_denormalized)




model = sys.argv[1]
epoch = sys.argv[2]
name = sys.argv[3]

save_conditional_generations(model, epoch, name)
