import sys
import os

# Add the correct path to trainers folder
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'trainers')))

#sys.path.append('./src/models/trainers')

from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv1DTranspose, ReLU, Conv1D, LeakyReLU, InputLayer, Concatenate, LayerNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input, Model
import tensorflow as tf 
from tensorflow_addons.layers import SpectralNormalization

from cwgangp import CWGANGP

#def call(self, inputs):
#    noise, labels = inputs
#    return self.generator(tf.concat([noise, labels], axis=1))

def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)

    return fake_loss - real_loss


def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)


def build_generator(codings_size, label_dim):
    noise_input = Input(shape=(codings_size,), name="noise_input")
    label_input = Input(shape=(label_dim,), name="label_input")

    # Embed real-valued label into a dense vector
    label_embedding = Dense(16, activation='relu', name='label_embedding')(label_input)
    combined_input = Concatenate(name="concat_noise_label")([noise_input, label_embedding])

    x = Dense(4 * 256)(combined_input)
    x = Reshape([4, 256])(x)
    x = LayerNormalization()(x)
    x = ReLU()(x)

    x = Conv1DTranspose(128, 25, strides=4, padding='same')(x)
    x = LayerNormalization()(x)
    x = ReLU()(x)

    x = Conv1DTranspose(64, 25, strides=4, padding='same')(x)
    x = LayerNormalization()(x)
    x = ReLU()(x)

    x = Conv1DTranspose(32, 25, strides=4, padding='same')(x)
    x = LayerNormalization()(x)
    x = ReLU()(x)

    output = Conv1DTranspose(16, 25, strides=4, padding='same', activation='tanh')(x)

    return Model([noise_input, label_input], output, name="generator")


def build_discriminator(input_shape, label_dim=1):
    # Input: radar signal
    signal_input = Input(shape=input_shape, name="signal_input")
    # Input: label broadcasted across time
    label_input = Input(shape=(input_shape[0], label_dim), name="label_input")

    # Concatenate along channel dimension
    x = Concatenate(axis=-1)([signal_input, label_input])  # (1024, 17)

    # x = Conv1D(32, kernel_size=25, strides=4, padding='same')(x)
    x = SpectralNormalization(Conv1D(32, kernel_size=25, strides=4, padding='same'))(x)
    x = LeakyReLU(0.2)(x)
    # x = Conv1D(64, kernel_size=25, strides=4, padding='same')(x)
    x = SpectralNormalization(Conv1D(64, kernel_size=25, strides=4, padding='same'))(x)
    x = LeakyReLU(0.2)(x)
    # x = Conv1D(128, kernel_size=25, strides=4, padding='same')(x)
    x = SpectralNormalization(Conv1D(128, kernel_size=25, strides=4, padding='same'))(x)
    x = LeakyReLU(0.2)(x)
    # x = Conv1D(256, kernel_size=25, strides=4, padding='same')(x)
    x = SpectralNormalization(Conv1D(256, kernel_size=25, strides=4, padding='same'))(x)
    x = LeakyReLU(0.2)(x)

    x = Flatten()(x)
    # output = Dense(1, activation='linear')(x)
    output = SpectralNormalization(Dense(1, activation='linear'))(x)

    return Model([signal_input, label_input], output, name='discriminator')

def build_discriminator(input_shape, label_dim=1):
    # Input: radar signal
    signal_input = Input(shape=input_shape, name="signal_input")
    # Input: label broadcasted across time (e.g., (1024, 1))
    label_input = Input(shape=(input_shape[0], label_dim), name="label_input")

    # Concatenate signal and label along the channel axis → (1024, 16 + 1)
    x = Concatenate(axis=-1)([signal_input, label_input])  # (1024, 17)

    x = SpectralNormalization(Conv1D(32, kernel_size=25, strides=4, padding='same'))(x)
    x = LeakyReLU(0.2)(x)

    x = SpectralNormalization(Conv1D(64, kernel_size=25, strides=4, padding='same'))(x)
    x = LeakyReLU(0.2)(x)

    x = SpectralNormalization(Conv1D(128, kernel_size=25, strides=4, padding='same'))(x)
    x = LeakyReLU(0.2)(x)

    x = SpectralNormalization(Conv1D(256, kernel_size=25, strides=4, padding='same'))(x)
    x = LeakyReLU(0.2)(x)

    x = Flatten()(x)
    output = SpectralNormalization(Dense(1, activation='linear'))(x)

    return Model([signal_input, label_input], output, name='discriminator')

def build_gan():
    BATCH_SIZE = 32
    EPOCHS = 1000
    CODINGS_SIZE = 100
    SIGNAL_LENGTH = 1024
    NUM_CHANNELS = 16
    LABEL_DIM = 1

    generator = build_generator(codings_size = CODINGS_SIZE, label_dim = LABEL_DIM)

    discriminator = build_discriminator(input_shape=(SIGNAL_LENGTH, NUM_CHANNELS), label_dim=LABEL_DIM)

    gan = CWGANGP(discriminator, generator, CODINGS_SIZE, discriminator_extra_steps=5)

    gan.compile(
        discriminator_optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
        generator_optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
        discriminator_loss_fn=discriminator_loss,
        generator_loss_fn=generator_loss
    )

    try:
        dummy_noise = tf.random.normal((1, CODINGS_SIZE))
        dummy_label = tf.zeros((1, 1))  # adjust if using multi-class
        #gen_input = tf.concat([dummy_noise, dummy_label], axis=1)
        #fake_signal = generator(gen_input)
        fake_signal = generator([dummy_noise, dummy_label])

        #dummy_cond = tf.zeros((1, 1024, 1))
        #disc_input = tf.concat([fake_signal, dummy_cond], axis=2)
        #_ = discriminator(disc_input)
        label_broadcast = tf.repeat(dummy_label[:, tf.newaxis, :], repeats=SIGNAL_LENGTH, axis=1)
        _ = discriminator([fake_signal, label_broadcast])

        print("[INFO] Generator and discriminator successfully built.")
    except Exception as e:
        print(f"[ERROR] Failed to build models during build_gan: {e}")

    gan_config = {
        'learning_rate': 0.0001,
        'batch_size': BATCH_SIZE,
        'codings_size': CODINGS_SIZE,
        'architecture': {
            'generator': generator.get_config(), 'discriminator': discriminator.get_config()
        },
    }

    gan.build(input_shape=[(None, CODINGS_SIZE), (None, LABEL_DIM)])

    return gan, BATCH_SIZE, EPOCHS, gan_config