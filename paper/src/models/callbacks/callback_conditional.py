from scipy.signal import welch
from scipy.stats import pearsonr
from scipy.stats import ks_2samp
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
import os
import wandb
import numpy as np
import tensorflow as tf
import glob


class WandbCallbackGANConditional(Callback):

    def __init__(self, wandb_module, real_sample=None, save_subdir="student", model_type="student"):
        self.wandb = wandb_module
        self.real_sample = real_sample
        self.model_type = model_type
        self.range_bins = np.arange(1024)
        # self.model_path = f'./checkpoints/model-{wandb_module.run.id}'
        self.model_path = os.path.join("./checkpoints", save_subdir, f"model-{wandb_module.run.id}")
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
            "similarity/psd_mse_mean": float(np.mean(mse_list)),
            "similarity/psd_mse_std": float(np.std(mse_list)),
            "similarity/psd_corr_mean": float(np.mean(corr_list)),
            "similarity/psd_corr_std": float(np.std(corr_list)),
            "epoch": epoch
        }, step=epoch)

    def log_similarity_metrics(self, real, fake, epoch):
        """
        Calculates and logs similarity metrics between real and fake radar signals.
        Metrics:
            - PSD L2 similarity
            - MMD using RBF kernel
            - Average KS test p-value across channels
        """
        real = real.numpy()
        fake = fake.numpy()
        N, L, C = real.shape  # number of samples, length of signal, number of channels

        # Calculate PSD_L2_Similarity
        psd_l2_scores = []
        for i in range(C):
            # Compute PSD for real and fake signals on this channel
            freqs_real, psd_real = welch(real[:, :, i], fs=20000000.0, axis=1)
            freqs_fake, psd_fake = welch(fake[:, :, i], fs=20000000.0, axis=1)

            # Average PSD across samples
            psd_real_avg = np.mean(psd_real, axis=0)
            psd_fake_avg = np.mean(psd_fake, axis=0)

            # Compute L2 distance between PSDs
            l2_diff = np.sqrt(np.sum((psd_real_avg - psd_fake_avg) ** 2))
            psd_l2_scores.append(l2_diff)

        psd_l2_mean = float(np.mean(psd_l2_scores))  # Final score

        # Calculate MMD_RBF
        # Flatten each sample into a single vector
        real_flat = real.reshape((N, -1))
        fake_flat = fake.reshape((N, -1))

        # Define a function to compute RBF kernel
        def rbf_kernel(X, Y, sigma=1.0):
            # Squared Euclidean distance
            dists = cdist(X, Y, metric='sqeuclidean')
            return np.exp(-dists / (2 * sigma ** 2))

        # Compute kernel matrices
        K_xx = rbf_kernel(real_flat, real_flat)
        K_yy = rbf_kernel(fake_flat, fake_flat)
        K_xy = rbf_kernel(real_flat, fake_flat)

        # MMD formula
        mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
        mmd_rbf = float(mmd)

        # Calculate KS_Avg_p_value
        ks_p_values = []
        for i in range(C):
            # Flatten the signal data for KS test
            real_vals = real[:, :, i].flatten()
            fake_vals = fake[:, :, i].flatten()

            # KS test between real and fake
            stat, p_value = ks_2samp(real_vals, fake_vals)
            ks_p_values.append(p_value)

        # Average p-value (higher means more similar distributions)
        ks_avg_p = float(np.mean(ks_p_values))

        # og to WandB 
        self.wandb.log({
            "similarity/psd_l2_mean": psd_l2_mean,
            "similarity/mmd_rbf": mmd_rbf,
            "similarity/ks_avg_p_value": ks_avg_p
        }, step=epoch)

        print(f"[Log] Similarity metrics at epoch {epoch} -> PSD_L2: {psd_l2_mean:.4f}, MMD: {mmd_rbf:.4f}, KS_p: {ks_avg_p:.4f}")

    def cleanup_old_backups(self, max_backups=5):
        # Match files like "backup-epoch-*.h5"
        backup_files = sorted(glob.glob(os.path.join(self.model_path, "backup-epoch-*.h5")))

        if len(backup_files) > max_backups:
            # Delete oldest files
            files_to_delete = backup_files[:-max_backups]
            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                    print(f"[Cleanup] Deleted old backup: {file_path}")
                except Exception as e:
                    print(f"[Warning] Could not delete {file_path}: {e}")
    

    def on_epoch_end(self, epoch, logs=None):
        # *** SAVING GENERATIONS AS IMAGE ***
        tf.random.set_seed(epoch)
        noise = tf.random.normal(shape=(3, 100))
        labels = tf.cast(tf.reshape(tf.linspace(23, 5, 3), shape=(3, 1)), tf.float32)
        labels = (labels - 10.981254577636719) / 7.1911773681640625

        if self.model_type == "student":
             generations = self.model.generator([noise, labels])
        else:  # original
            noise_and_labels = tf.concat([noise, labels], axis=1)
            generations = self.model.generator(noise_and_labels)

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

        }, step=epoch)
        plt.tight_layout()
        plt.close(fig)

        # if real sample is passed in, log comparision metrics
        if self.real_sample is not None:
            self.log_channelwise_statistics(self.real_sample, generations, step=epoch)
            self.log_psd(self.real_sample, generations, step=epoch)
            self.log_similarity_metrics(self.real_sample, generations, step=epoch)

        # *** SAVE MODEL WEIGHTS ***
        # Ensure model is built before saving
        try:
            tf.random.set_seed(epoch)
            dummy_noise = tf.random.normal((1, self.model.latent_dim))
            dummy_label = tf.zeros((1, 1))  # adjust if your labels are multi-dimensional

            if self.model_type == "student":
                gen_input = tf.concat([dummy_noise, dummy_label], axis=1)
                fake_signal = self.model.generator(gen_input)
            else:  # original
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

        # Save generator model
        gen_save_path = os.path.join(self.model_path, f"generator-epoch-{epoch+1}")
        self.model.generator.save(f"{gen_save_path}.h5")
        print(f"[Saved] Generator model saved to: {gen_save_path}")

        # Clean up old backups
        self.cleanup_old_backups(max_backups=10)
