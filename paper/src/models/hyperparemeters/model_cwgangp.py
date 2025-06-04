import sys
import os

# Add the correct path to trainers folder
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'trainers')))

#sys.path.append('./src/models/trainers')

from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv1DTranspose, ReLU, Conv1D, LeakyReLU, Concatenate, Add, LayerNormalization, Wrapper
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input, Model
from tensorflow.keras import backend as K


import tensorflow as tf 
#from tensorflow_addons.layers import SpectralNormalization

from cwgangp import CWGANGP

# def call(self, inputs):
#     noise, labels = inputs
#     return self.generator(tf.concat([noise, labels], axis=1))

class SpectralNormalization(Wrapper):
    def __init__(self, layer, power_iterations=1, **kwargs):
        super().__init__(layer, **kwargs)
        self.power_iterations = power_iterations

    def build(self, input_shape):
        super().build(input_shape)
        if not hasattr(self.layer, "kernel"):
            raise ValueError("Wrapped layer must have a 'kernel' attribute")
        self.w = self.layer.kernel
        self.u = self.add_weight(
            shape=(1, self.w.shape[-1]),
            initializer="random_normal",
            trainable=False,
            name="sn_u"
        )

    def compute_spectral_norm(self):
        w_reshaped = tf.reshape(self.w, [-1, self.w.shape[-1]])
        u = self.u
        for _ in range(self.power_iterations):
            v = tf.math.l2_normalize(tf.matmul(u, tf.transpose(w_reshaped)))
            u = tf.math.l2_normalize(tf.matmul(v, w_reshaped))
        sigma = tf.matmul(tf.matmul(v, w_reshaped), tf.transpose(u))
        self.u.assign(u)
        return self.w / sigma

    def call(self, inputs, training=None):
        w_bar = self.compute_spectral_norm()

        if isinstance(self.layer, tf.keras.layers.Dense):
            output = tf.matmul(inputs, w_bar)
            if self.layer.use_bias:
                output = tf.nn.bias_add(output, self.layer.bias)
            if self.layer.activation is not None:
                return self.layer.activation(output)
            return output

        elif isinstance(self.layer, tf.keras.layers.Conv1D):
            output = K.conv1d(
                inputs,
                w_bar,
                strides=self.layer.strides[0],
                padding=self.layer.padding.lower(),  # FIXED
                data_format="channels_last",
                dilation_rate=self.layer.dilation_rate[0],
            )
            if self.layer.use_bias:
                output = tf.nn.bias_add(output, self.layer.bias)
            if self.layer.activation is not None:
                return self.layer.activation(output)
            return output

        elif isinstance(self.layer, tf.keras.layers.Conv2D):
            output = K.conv2d(
                inputs,
                w_bar,
                strides=self.layer.strides,
                padding=self.layer.padding.upper(),
                data_format="channels_last",
                dilation_rate=self.layer.dilation_rate,
            )
            if self.layer.use_bias:
                output = tf.nn.bias_add(output, self.layer.bias)
            if self.layer.activation is not None:
                return self.layer.activation(output)
            return output

        else:
            raise NotImplementedError(f"SpectralNormalization not implemented for {type(self.layer)}")

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def compute_output_spec(self, inputs, training=None):
        return self.layer.compute_output_spec(inputs)

def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)

    return fake_loss - real_loss


def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)

def residual_block_1d_generator(x, filters, kernel_size=25, strides=1):
    """
    1D residual block with Conv1DTranspose and layer normalization for generator
    Input x: shape (N, L, C), where N is the batch size, L=1024 is the signal length, and C in the number of input channels 
    Output: shape (N, L * strides, filters), where filters is the number of filters
    """
    C = x.shape[-1]
    
    # Main path
    y = Conv1DTranspose(filters, kernel_size, strides=strides, padding='same')(x) # (N, L*strides, filters)
    y = LayerNormalization()(y) # (N, L*strides, filters)
    y = ReLU()(y) # (N, L*strides, filters); note L*strides = L for stides = 1

    # Residual shortcut w/ shape matching (Conv1DTranspose preserves linearity)
    if C != filters or strides != 1:
        shortcut = Conv1DTranspose(filters, 1, strides=strides, padding='same')(x) # (N, L*strides, filters)
    else:
        shortcut = x # (N, L, filters)

    output = ReLU()(Add()([y, shortcut]))  # Output = ReLU(MainPath(x) + ShortCut(x)); (N, L*strides, filters)

    return output

def residual_block_1d_discriminator(x, filters, kernel_size=25, strides=1):
    """
    1D residual block with Conv1D spectral normalization for discriminator
    Input x: shape (N, L, C), where N is the batch size, L=1024 is the signal length, and C in the number of input channels
    Output: shape (N, L // strides, filters), where filters is the number of filters
    """
    C = x.shape[-1]

    y = SpectralNormalization(Conv1D(filters, kernel_size, strides=strides, padding='same'))(x) # (N, L//strides, filters)
    y = LeakyReLU(0.2)(y) # (N, L//strides, filters); note L//strides = L for strides = 1

    # Residual shortcut w/ shape matching (Conv1D and spectram normalization preserve linearity)
    if C != filters or strides != 1:
        shortcut = SpectralNormalization(Conv1D(filters, 1, strides=strides, padding='same'))(x) # (N, L//strides, filters)
    else:
        shortcut = x # (N, L, C)

    output = Add()([y, shortcut]) # output = MainPath(x) + ShortCut(x), (N, L//strides, filters)

    return output


def build_generator(codings_size, label_dim):
    
    noise_input = Input(shape=(codings_size,), name="noise_input") # (N, codings_size)
    label_input = Input(shape=(label_dim,), name="label_input") # (N, label_dim)

    # Embed real-valued label into a dense vector
    label_embedding = Dense(16, activation='relu', name='label_embedding')(label_input) # (N, 16)
    combined_input = Concatenate(name="concat_noise_label")([noise_input, label_embedding]) # (N, codings_size + 16)

    # Reshape to give proper shape to residual 1DConv blocks
    x = Dense(4 * 256)(combined_input) # (N, 1024)
    x = Reshape((4, 256))(x) # (N, 4, 256)

    # x = Conv1DTranspose(128, 25, strides=4, padding='same')(x)
    # x = LayerNormalization()(x)
    # x = ReLU()(x)

    # x = Conv1DTranspose(64, 25, strides=4, padding='same')(x)
    # x = LayerNormalization()(x)
    # x = ReLU()(x)

    # x = Conv1DTranspose(32, 25, strides=4, padding='same')(x)
    # x = LayerNormalization()(x)
    # x = ReLU()(x)

    x = residual_block_1d_generator(x, 128, strides=4) # (N, 16, 128)
    x = residual_block_1d_generator(x, 64, strides=4) # (N, 64, 64)
    x = residual_block_1d_generator(x, 32, strides=4) # (N, 256, 32)

    output = Conv1DTranspose(16, 25, strides=4, padding='same', activation='tanh')(x) # (N, 1024, 16)

    return Model([noise_input, label_input], output, name="generator")


def build_discriminator(input_shape, label_dim=1):
    # Input: radar signal
    signal_input = Input(shape=input_shape, name="signal_input") # (N, codings_size)
    # Input: label broadcasted across time (e.g., (1024, 1))
    label_input = Input(shape=(input_shape[0], label_dim), name="label_input") # (N, label_dim)

    # Concatenate signal and label along the channel axis: (1024, 16 + 1)
    x = Concatenate(axis=-1)([signal_input, label_input])  # (1024, 17)

    # x = SpectralNormalization(Conv1D(32, kernel_size=25, strides=4, padding='same'))(x)
    # x = LeakyReLU(0.2)(x)

    # x = SpectralNormalization(Conv1D(64, kernel_size=25, strides=4, padding='same'))(x)
    # x = LeakyReLU(0.2)(x)

    # x = SpectralNormalization(Conv1D(128, kernel_size=25, strides=4, padding='same'))(x)
    # x = LeakyReLU(0.2)(x)

    # x = SpectralNormalization(Conv1D(256, kernel_size=25, strides=4, padding='same'))(x)
    # x = LeakyReLU(0.2)(x)

    x = residual_block_1d_discriminator(x, 32, strides=4) # (N, 256, 32)
    x = residual_block_1d_discriminator(x, 64, strides=4) # (N, 64, 64)
    x = residual_block_1d_discriminator(x, 128, strides=4) # (N, 16, 128)
    x = residual_block_1d_discriminator(x, 256, strides=4) # (N, 4, 256)

    x = Flatten()(x) # (N, 1024)
    output = SpectralNormalization(Dense(1, activation='linear'))(x) # (N, 1)

    return Model([signal_input, label_input], output, name='discriminator')

def build_gan(num_epochs=1, batch_size=32):
    BATCH_SIZE = batch_size
    EPOCHS = num_epochs
    CODINGS_SIZE = 100
    SIGNAL_LENGTH = 1024
    NUM_CHANNELS = 16
    LABEL_DIM = 1

    generator = build_generator(CODINGS_SIZE, LABEL_DIM)
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

    # gan.build(input_shape=[(None, CODINGS_SIZE), (None, 1)])  # Do we need this?

    return gan, BATCH_SIZE, EPOCHS, gan_config
