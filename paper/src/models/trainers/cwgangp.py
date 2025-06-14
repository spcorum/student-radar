from tensorflow.keras import Model
from tensorflow.keras.layers import Wrapper
from tensorflow.keras.metrics import Mean
import tensorflow as tf

tf.random.set_seed(1729)

class SpectralNormalization(Wrapper):
    def __init__(self, layer, power_iterations=1, **kwargs):
        super(SpectralNormalization, self).__init__(layer, **kwargs)
        self.power_iterations = power_iterations

    def build(self, input_shape):
        self.layer.build(input_shape)
        self.w = self.layer.kernel
        self.u = self.add_weight(shape=(1, self.w.shape[-1]), initializer='random_normal', trainable=False, name='sn_u')
        super(SpectralNormalization, self).build()

    def call(self, inputs):
        w_reshaped = tf.reshape(self.w, [-1, self.w.shape[-1]])
        u = self.u
        for _ in range(self.power_iterations):
            v = tf.nn.l2_normalize(tf.matmul(u, tf.transpose(w_reshaped)))
            u = tf.nn.l2_normalize(tf.matmul(v, w_reshaped))
        sigma = tf.matmul(tf.matmul(v, w_reshaped), tf.transpose(u))
        w_bar = self.w / sigma
        self.layer.kernel = w_bar
        self.u.assign(u)
        return self.layer(inputs)


class CWGANGP(Model):

    def __init__(self, discriminator, generator, latent_dim, discriminator_extra_steps=3, gp_weight=10.0, **kwargs):
        super().__init__(**kwargs)
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight
        self.d_loss_tracker = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_tracker = tf.keras.metrics.Mean(name="g_loss")

    def get_config(self):
      return {
            "latent_dim": self.latent_dim,
            "discriminator_extra_steps": self.d_steps,
            "gp_weight": self.gp_weight,
      }

    @classmethod
    def from_config(cls, config):
        return cls(generator=None, discriminator=None, **config)

    def call(self, inputs, training=False):
        noise, labels = inputs
        return self.generator([noise, labels], training=training)

    @property
    def metrics(self):
        return [self.d_loss_tracker, self.g_loss_tracker]

    def compile(self, discriminator_optimizer, generator_optimizer, discriminator_loss_fn, generator_loss_fn,weighted_metrics=[]):
        super(CWGANGP, self).compile()
        self.d_optimizer = discriminator_optimizer
        self.g_optimizer = generator_optimizer
        self.d_loss_fn = discriminator_loss_fn
        self.g_loss_fn = generator_loss_fn


    def gradient_penalty(self, batch_size, real_images, fake_images):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        # interpolated shape: (batch_size, 1024, signal_dim + label_dim)
        # signal_dim = self.discriminator.input_shape[0][-1] - 1  # assuming label_dim = 1
        signal_dim = tf.shape(real_images)[-1] - 1
        signal_part = interpolated[:, :, :signal_dim]
        label_part = interpolated[:, :, signal_dim:]
        
        with tf.GradientTape() as gp_tape:
            # gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            # pred = self.discriminator(interpolated, training=True)
            # pred = self.discriminator([signal_part, label_part], training=True)
            gp_tape.watch([signal_part, label_part])
            pred = self.discriminator([signal_part, label_part], training=True)

        grads = gp_tape.gradient(pred, [signal_part, label_part])
        grad_signal, grad_label = grads
        grads_combined = tf.concat([grad_signal, grad_label], axis=-1)

        norm = tf.sqrt(tf.reduce_sum(tf.square(grads_combined), axis=[1, 2]) + 1e-12)
        # 2. Calculate the gradients w.r.t to this interpolated image.
        # grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        # norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)

        return gp

    def predict(self, noise, labels, training=False):
        """
        Inputs:
            noise (N,L,C)
            labels (N,)
            training - wether to run generator in training mode

        Outputs:
            Tensor of generated radar signals: shape (batch_size, 1024, 16)
        """
        return self.generator([noise, labels], training=training)


    def train_step(self, X_batch):
        real_signals, labels_generator, labels_discriminator = X_batch
        
        # Get the batch size
        batch_size = tf.shape(real_signals)[0]

        # Made one channel with labels
        # labels_discriminator_channel = tf.repeat(
        #     labels_discriminator, repeats=[1024]
        # )
        # labels_discriminator_channel = tf.reshape(
        #     labels_discriminator_channel, (-1, 1024, 1)
        # )
        labels_discriminator_channel = tf.repeat(
            tf.expand_dims(labels_discriminator, axis=1), repeats=1024, axis=1
        )

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            # Concate latent vector with the labels
            # random_latent_labels = tf.concat(
            #     [random_latent, labels_generator], axis=1
            # )
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                # fake_signals = self.generator(random_latent_labels, training=True)
                fake_signals = self.generator([random_latent, labels_generator], training=True)
                # # Concat generations with the labels
                # fake_signals_labels = tf.concat(
                #     [fake_signals, labels_discriminator_channel], axis=2
                # )
                # Get the logits for the fake images
                # fake_logits = self.discriminator(fake_signals_labels, training=True)
                fake_logits = self.discriminator([fake_signals, labels_discriminator_channel], training=True)
                # # Concat real with the labels
                # real_signals_labels = tf.concat(
                #     [real_signals, labels_discriminator_channel], axis=2
                # )
                # Get the logits for the real images
                # real_logits = self.discriminator(real_signals_labels, training=True)
                real_logits = self.discriminator([real_signals, labels_discriminator_channel], training=True)

                # Calculate the discriminator loss using the fake and real image logits
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)

                # Concat generations with the labels & real with the labels as gradient_penality expects this
                fake_signals_labels = tf.concat(
                    [fake_signals, labels_discriminator_channel], axis=2
                )
                real_signals_labels = tf.concat(
                    [real_signals, labels_discriminator_channel], axis=2
                )

                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_signals_labels, fake_signals_labels)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator
        # Get the latent vector
        random_latent = tf.random.normal(shape=(batch_size, self.latent_dim))
        # Concate latent vectors with the labels
        random_latent_labels = tf.concat(
            [random_latent, labels_generator], axis=1
        )
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_signals = self.generator([random_latent, labels_generator], training=True)
            # Concat generations with the labels
            generated_signals_labels = tf.concat(
                [generated_signals, labels_discriminator_channel], axis=2
            )
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator([generated_signals, labels_discriminator_channel], training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )

        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)
        #self.add_metric(d_loss, name="discriminator_loss", aggregation="mean")
        #self.add_metric(g_loss, name="generator_loss", aggregation="mean")
        return {
            "discriminator_loss": self.d_loss_tracker.result(),
            "generator_loss": self.g_loss_tracker.result()
        }

    def test_step(self, X_batch):
        real_signals, labels_generator, labels_discriminator = X_batch

        # Get the batch size
        batch_size = tf.shape(real_signals)[0]

        # Make one channel with labels
        # labels_discriminator_channel = tf.repeat(
        #     labels_discriminator, repeats=[1024]
        # )
        # labels_discriminator_channel = tf.reshape(
        #     labels_discriminator_channel, (-1, 1024, 1)
        # )
        labels_discriminator_channel = tf.repeat(
            tf.expand_dims(labels_discriminator, axis=1), repeats=1024, axis=1
        )  # (N, 1024, 1)


        # Get the latent vector
        random_latent = tf.random.normal(shape=(batch_size, self.latent_dim))
        # Concatenate latent vectors with the labels
        random_latent_labels = tf.concat(
            [random_latent, labels_generator], axis=1
        )

        # Generate fake signals from the latent vector
        fake_signals = self.generator([random_latent, labels_generator], training=False)
        # Concatenate generations with the labels
        fake_signals_labels = tf.concat(
            [fake_signals, labels_discriminator_channel], axis=2
        )
        # Concatenate real signals with the labels
        real_signals_labels = tf.concat(
            [real_signals, labels_discriminator_channel], axis=2
        )

        # Get the logits for the fake and real signals
        fake_logits = self.discriminator([fake_signals, labels_discriminator_channel], training=False)
        real_logits = self.discriminator([real_signals, labels_discriminator_channel], training=False)

        # Calculate the discriminator and generator losses
        d_loss = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
        g_loss = self.g_loss_fn(fake_logits)


        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)
        #self.add_metric(d_loss, name="val_discriminator_loss", aggregation="mean")
        #self.add_metric(g_loss, name="val_generator_loss", aggregation="mean")
        return {
            "discriminator_loss": self.d_loss_tracker.result(),
            "generator_loss": self.g_loss_tracker.result()
    }