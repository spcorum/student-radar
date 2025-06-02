from scipy.signal import welch
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
import os
import wandb
import numpy as np
import tensorflow as tf


class WandbCallbackGANConditional(Callback):

    def __init__(self, wandb_module, real_sample=None):
        self.wandb = wandb_module
        self.real_sample = real_sample
        self.range_bins = np.arange(1024)
        self.model_path = f'./checkpoints/model-{wandb_module.run.id}'
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        super().__init__()

    def log_channelwise_statistics(self, real, fake, epoch):
        """
        real, fake: (N, 1024, C)
        Logs signal mean and standard deviation
        """
        real_mean = tf.reduce_mean(real, axis=[0, 1]).numpy()  # shape: (C,)
        fake_mean = tf.reduce_mean(fake, axis=[0, 1]).numpy()

        real_std = tf.math.reduce_std(real, axis=[0, 1]).numpy()
        fake_std = tf.math.reduce_std(fake, axis=[0, 1]).numpy()

        for i in range(real.shape[-1]):
            self.wandb.log({
                f"mean_diff_channel_{i}": abs(real_mean[i] - fake_mean[i]),
                f"std_diff_channel_{i}": abs(real_std[i] - fake_std[i]),
            }, step=epoch)

    def log_psd(self, real, fake, epoch, sample_rate=20000000.0):
        """
        real, fake: (N, 1024, C)
        Logs PSD comparison for first 2 channels
        Note: uses 'welch' instead of FFT for reduced noise. welch computes the 
        power spectrum with the following steps:
            1  Splits the signal into overlapping segments
            2. Applies a Hanning window function
            3. Computes the FFT of the combined signal (original * window)
            4. Computes the periodogram (squared magnitude of FFT)
            5. Averages the periodograms
        welch returns the following:
            - freqs: array of frequency bins
            - psd: The estimated PSD (power per Hz) at each frequency bins
        """
        real = real.numpy()
        fake = fake.numpy()
        N, L, C = real.shape

        mse_list = []
        corr_list = []

        for ch in range(C):

            freqs, psd_real = welch(real[:, :, ch], fs=sample_rate, axis=1)
            _, psd_fake = welch(fake[:, :, ch], fs=sample_rate, axis=1)

            psd_real_mean = psd_real.mean(axis=0)
            psd_fake_mean = psd_fake.mean(axis=0)

            # save plots for the first two channels
            if ch < 2:
                plt.figure()
                plt.semilogy(freqs, psd_real_mean, label='Real')
                plt.semilogy(freqs, psd_fake_mean, label='Fake')
                plt.title(f'PSD Comparison - Channel {ch}')
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Power/Frequency (dB/Hz)')
                plt.legend()
                plt.grid(True)
                self.wandb.log({f"psd_channel_{ch}": wandb.Image(plt)}, step=epoch)
                plt.tight_layout()
                plt.close()

            # calculate metrics
            mse = np.mean((psd_real_mean - psd_fake_mean) ** 2)
            corr, _ = pearsonr(psd_real_mean, psd_fake_mean)
            mse_list.append(mse)
            corr_list.append(corr)

        # logging
        self.wandb.log({
            "psd_mse_mean": float(np.mean(mse_list)),
            "psd_mse_std": float(np.std(mse_list)),
            "psd_corr_mean": float(np.mean(corr_list)),
            "psd_corr_std": float(np.std(corr_list)),
            "epoch": epoch
        })


    

    def on_epoch_end(self, epoch, logs=None):
        # *** SAVING GENERATIONS AS IMAGE ***
        tf.random.set_seed(epoch)
        noise = tf.random.normal(shape=(3, 100))
        labels = tf.cast(tf.reshape(tf.linspace(23, 5, 3), shape=(3, 1)), tf.float32)
        labels = (labels - 10.981254577636719) / 7.1911773681640625
        # noise_and_labels = tf.concat([noise, labels], axis=1)

        # generations = self.model.generator([noise_and_labels])
        generations = self.model.generator([noise, labels])

        a, b = -1, 1
        data_min, data_max = -2444.0, 2544.0
        generations_denormalized = data_min + (generations - a) * (data_max - data_min) / (b - a)

        fig, axes = plt.subplots(1, 3, figsize=(40, 10))
        for i in range(3):
            for channel in range(16):
                axes[i].plot(self.range_bins, generations_denormalized[i, :, channel])
            axes[i].grid(True)

        # Log as a static image to avoid needing Plotly
        self.wandb.log({
            'epoch': epoch,
            'train/discriminator_loss': logs.get('discriminator_loss'),
            'train/generator_loss': logs.get('generator_loss'),
            'val/discriminator_loss': logs.get('val_discriminator_loss'),
            'val/generator_loss': logs.get('val_generator_loss'),
            'generations': self.wandb.Image(fig)
        })
        plt.tight_layout()
        plt.close(fig)

        # if real sample is passed in, log comparision metrics
        if self.real_sample is not None:
            self.log_channelwise_statistics(self.real_sample, generations, epoch)
            self.log_psd(self.real_sample, generations, epoch)

        # *** SAVE MODEL WEIGHTS ***
        # Ensure model is built before saving
        try:
                tf.random.set_seed(epoch)
                dummy_noise = tf.random.normal((1, self.model.latent_dim))
                dummy_label = tf.zeros((1, 1))  # adjust if your labels are multi-dimensional
                # gen_input = tf.concat([dummy_noise, dummy_label], axis=1)
                # fake_signal = self.model.generator(gen_input)
                fake_signal = self.model.generator([dummy_noise, dummy_label])

                dummy_cond = tf.zeros((1, 1024, 1))  # ensure shape matches discriminator input
                _ = self.model.discriminator([fake_signal, dummy_cond])

                print("[INFO] Submodels are initialized and ready for saving.")
        except Exception as e:
                print(f"[Warning] Could not build submodels before saving: {e}")
                return

        # Save periodically and also to wandb directory
        filename = f"model-{self.wandb.run.id}-epoch-{epoch+1}.weights.h5"
        full_path = os.path.join(self.model_path, filename)
        self.model.save_weights(full_path)
        print(f"[Saved] Model weights saved to: {full_path}")

        # Save backup weights at each epoch
        backup_filename = f"backup-epoch-{epoch+1}.h5"
        backup_path = os.path.join(self.model_path, backup_filename)
        self.model.save_weights(backup_path)
        print(f"[Backup] Model weights saved to: {backup_path}")

        # Save latest weights to wandb directory for resuming
        if self.wandb.run:
            wandb_resume_path = os.path.join(self.wandb.run.dir, f"model-{self.wandb.run.id}.h5")
            self.model.save_weights(wandb_resume_path)
            print(f"[Resume] Model weights saved to: {wandb_resume_path}")

            # ✅ Log that weights were saved
            self.wandb.log({"weights_saved_epoch": epoch + 1}, step=epoch)

        # Log current epoch
        self.wandb.log({"epoch_logged": epoch + 1}, step=epoch)

        # Save last epoch for resuming 
        self.wandb.config.update({"last_epoch": epoch + 1}, allow_val_change=True)
