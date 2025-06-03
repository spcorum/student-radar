import sys
import os

# Add the correct path to trainers folder
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'trainers')))

#sys.path.append('./src/models/trainers')

from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv1DTranspose, ReLU, Conv1D, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from cwgangp_original import CWGANGPOriginal

# def call(self, inputs):
#     noise, labels = inputs
#     return self.generator(tf.concat([noise, labels], axis=1))

def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)

    return fake_loss - real_loss


def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)


def build_gan():
    BATCH_SIZE = 32
    EPOCHS = 1

    CODINGS_SIZE = 100

    generator = Sequential([
        Dense(4 * 256, input_shape=(CODINGS_SIZE+1,)),  # Added 1 here to have correct shape
        Reshape([4, 256]),
        ReLU(),
        Conv1DTranspose(128, kernel_size=25, strides=4, padding='same', activation='relu'),
        Conv1DTranspose(64, kernel_size=25, strides=4, padding='same', activation='relu', ),
        Conv1DTranspose(32, kernel_size=25, strides=4, padding='same', activation='relu', ),
        Conv1DTranspose(16, kernel_size=25, strides=4, padding='same', activation='tanh', )
    ], name='generator')

    discriminator = Sequential([
        # InputLayer((1024, 17)),
        # Conv1D(32, kernel_size=25, strides=4, padding='same',  input_shape=[1024, 16]),  # Is there a shape mismatch here? Original model in repo had it this way
        Conv1D(32, kernel_size=25, strides=4, padding='same',  input_shape=[1024, 17]),  # Is there a shape mismatch here? Original model in repo had it this way
        LeakyReLU(0.2),
        Conv1D(64, kernel_size=25, strides=4, padding='same'),
        LeakyReLU(0.2),
        Conv1D(128, kernel_size=25, strides=4, padding='same'),
        LeakyReLU(0.2),
        Conv1D(256, kernel_size=25, strides=4, padding='same'),
        LeakyReLU(0.2),
        Flatten(),
        Dense(1, activation='linear'),
    ], name='discriminator')

    gan = CWGANGPOriginal(discriminator, generator, CODINGS_SIZE, discriminator_extra_steps=5)

    gan.compile(
        discriminator_optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
        generator_optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
        discriminator_loss_fn=discriminator_loss,
        generator_loss_fn=generator_loss
    )

    try:
        dummy_noise = tf.random.normal((1, CODINGS_SIZE))
        dummy_label = tf.zeros((1, 1))  # adjust if using multi-class
        gen_input = tf.concat([dummy_noise, dummy_label], axis=1)
        fake_signal = generator(gen_input)

        dummy_cond = tf.zeros((1, 1024, 1))
        disc_input = tf.concat([fake_signal, dummy_cond], axis=2)
        _ = discriminator(disc_input)

        # Build full model by calling once
        gen_input = tf.concat([dummy_noise, dummy_label], axis=1)
        _ = gan(gen_input)

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

    # gan.build(input_shape=[(None, CODINGS_SIZE), (None, 1)])  # Do we need this?

    return gan, BATCH_SIZE, EPOCHS, gan_config
