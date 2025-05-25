from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import wandb


class WandbCallbackGANConditional(Callback):

    def __init__(self, wandb_module):
        self.wandb = wandb_module
        self.times = [i for i in range(1024)]
        self.model_path = f'./checkpoints/model-{wandb_module.run.id}'
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        # *** SAVING GENERATIONS AS IMAGE ***
        noise = tf.random.normal(shape=(3, 100))
        labels = tf.cast(tf.reshape(tf.linspace(23, 5, 3), shape=(3, 1)), tf.float32)
        labels = (labels - 10.981254577636719) / 7.1911773681640625
        noise_and_labels = tf.concat([noise, labels], axis=1)

        generations = self.model.generator(noise_and_labels)

        a, b = -1, 1
        data_min, data_max = -2444.0, 2544.0
        generations_denormalized = data_min + (generations - a) * (data_max - data_min) / (b - a)

        fig, axes = plt.subplots(1, 3, figsize=(40, 10))
        for i in range(3):
            for channel in range(16):
                axes[i].plot(self.times, generations_denormalized[i, :, channel])
            axes[i].grid(True)

        # Log as a static image to avoid needing Plotly
        self.wandb.log({
            'epoch': epoch,
            'discriminator_loss': logs.get('discriminator_loss'),
            'generator_loss': logs.get('generator_loss'),
            'generations': self.wandb.Image(fig)
        })
        plt.close(fig)

        # *** SAVE MODEL WEIGHTS ***
        # Ensure model is built before saving
        try:
                dummy_noise = tf.random.normal((1, self.model.latent_dim))
                dummy_label = tf.zeros((1, 1))  # adjust if your labels are multi-dimensional
                gen_input = tf.concat([dummy_noise, dummy_label], axis=1)
                fake_signal = self.model.generator(gen_input)

                dummy_cond = tf.zeros((1, 1024, 1))  # ensure shape matches discriminator input
                disc_input = tf.concat([fake_signal, dummy_cond], axis=2)
                _ = self.model.discriminator(disc_input)

                print("[INFO] Submodels are initialized and ready for saving.")
        except Exception as e:
                print(f"[Warning] Could not build submodels before saving: {e}")
                return

        # Save periodically and also to wandb directory
        filename = f"model-{self.wandb.run.id}-epoch-{epoch+1}.weights.h5"
        full_path = os.path.join(self.model_path, filename)
        self.model.save_weights(full_path)
        print(f"[Saved] Model weights saved to: {full_path}")

        if self.wandb.run:
            wandb_weights_path = os.path.join(self.wandb.run.dir, f"model-{self.wandb.run.id}.weights.h5")
            self.model.save_weights(wandb_weights_path)
            # wandb.save() expects relative path, so we use only filename
            #self.wandb.save(wandb_weights_path)
            print(f"[Saved] Model weights saved to: {wandb_weights_path}")
