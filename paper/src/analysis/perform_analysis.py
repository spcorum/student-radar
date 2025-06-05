from imageio import save
import tensorflow as tf
import numpy as np
import sys
import os 
import time
import argparse
import glob
import wandb
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import pearsonr, ks_2samp
from scipy.spatial.distance import cdist
import pprint
import json
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA
from joblib import Parallel, delayed

def truncate_long_lists(obj, max_len=10):
    if isinstance(obj, dict):
        return {k: truncate_long_lists(v, max_len) for k, v in obj.items()}
    elif isinstance(obj, list):
        if len(obj) > max_len:
            return obj[:max_len] + ['... ({} more elements)'.format(len(obj) - max_len)]
        return obj
    else:
        return obj

def print_results_table(results, float_precision=6):

    def fmt(x):
      if isinstance(x, float):
        if abs(x) < 1e-6:
            return f"{x:.6e}"
        else:
            return f"{x:.6f}"
      elif x is None:
        return "N/A"
      else:
        return str(x)

    # Get model info from the results
    model_type = results.get("meta", {}).get("model_type", "unknown").capitalize()
    model_name = results.get("meta", {}).get("model_name", "N/A")
    total_epochs = results.get("meta", {}).get("num_total_epochs", "N/A")

    print()
    print(f"=== Results Table for {model_type}GAN Model ===")
    print(f"Model name: {model_name}")
    print(f"Total training epochs: {total_epochs}")
    print()

    print(f"{'Metric':<22} {'Train':>12} {'Val':>12} {'Test':>12}")
    print("-" * 60)

    for key in results:
        if key == "meta":
            continue
        values = results[key]
        if isinstance(values, dict) and all(k in values for k in ["train", "val", "test"]):
            train_val = fmt(values["train"])
            val_val = fmt(values["val"])
            test_val = fmt(values["test"])
            print(f"{key:<25} {train_val:>15} {val_val:>15} {test_val:>15}")

    print("-" * 60)
    print("Note: All values are averages over evaluation iterations.\n")

def channelwise_statistics(data, gen_data, print_time = False):
    """
    data, gen_data: (N, 1024, 16)
    calculates signal mean and standard deviation over the data
    """

    start_time = time.time()

    data_mean = np.mean(data, axis=(0, 1))
    gen_mean = np.mean(gen_data, axis=(0, 1))
    data_std = np.std(data, axis=(0, 1))
    gen_std = np.std(gen_data, axis=(0, 1))

    mean_diff = np.abs(data_mean - gen_mean)
    std_diff = np.abs(data_std - gen_std)

    if print_time: 
        print(f"[INFO] PSD Similarity: {time.time()-start_time} s")  

    return mean_diff, std_diff, time.time()-start_time

def compute_psd_similarity(real, fake, sample_rate=20e6, print_time=False):
    """
    Vectorized computation of PSD MSE, Pearson correlation, and L2 difference per channel.
    """
    N, L, C = real.shape
    start_time = time.time()

    # Combine channel and batch dimension for vectorized Welch
    real_reshaped = real.transpose(2, 0, 1).reshape(C * N, L) # (C*N, L)
    fake_reshaped = fake.transpose(2, 0, 1).reshape(C * N, L)

    freqs, psd_real = welch(real_reshaped, fs=sample_rate, axis=1)
    _, psd_fake = welch(fake_reshaped, fs=sample_rate, axis=1)

    # Reshape back to original data dimension
    F = psd_real.shape[1]
    psd_real = psd_real.reshape(C, N, F)
    psd_fake = psd_fake.reshape(C, N, F)

    # PSD means
    psd_real_mean = psd_real.mean(axis=1)
    psd_fake_mean = psd_fake.mean(axis=1)

    # MSE per channel
    mse_array = np.mean((psd_real_mean - psd_fake_mean) ** 2, axis=1)

    # Pearson correlation per channel
    corr_list = [
        pearsonr(psd_real_mean[i], psd_fake_mean[i])[0] for i in range(C)
    ]

    # L2 difference per channel
    l2_array = np.sqrt(np.sum((psd_real_mean - psd_fake_mean) ** 2, axis=1))

    if print_time:
        print(f"[INFO] PSD Similarity: {elapsed:.4f} s")

    # Return the means of the metrics
    return (
        np.mean(mse_array), np.std(mse_array),
        np.mean(corr_list), np.std(corr_list),
        np.mean(l2_array), time.time() - start_time
    )

def compute_psd_l2(real, fake, sample_rate=20e6, print_time = False):
    N, L, C = real.shape
    l2_scores = []

    start_time = time.time()

    for ch in range(C):
        _, psd_real = welch(real[:, :, ch], fs=sample_rate, axis=1)
        _, psd_fake = welch(fake[:, :, ch], fs=sample_rate, axis=1)

        psd_real_avg = psd_real.mean(axis=0)
        psd_fake_avg = psd_fake.mean(axis=0)

        l2_diff = np.sqrt(np.sum((psd_real_avg - psd_fake_avg) ** 2))
        l2_scores.append(l2_diff)

    if print_time: 
        print(f"[INFO] PSD Similarity: {time.time()-start_time} s")  

    return np.mean(l2_scores)

def compute_mmd_rbf(real, fake, sigma=1.0, print_time = False):
    min_samples = min(real.shape[0], fake.shape[0])
    real_flat = real[:min_samples].reshape(min_samples, -1)
    fake_flat = fake[:min_samples].reshape(min_samples, -1)

    start_time = time.time()

    def rbf_kernel(X, Y):
        dists = cdist(X, Y, 'sqeuclidean')
        return np.exp(-dists / (2 * sigma ** 2))

    K_xx = rbf_kernel(real_flat, real_flat)
    K_yy = rbf_kernel(fake_flat, fake_flat)
    K_xy = rbf_kernel(real_flat, fake_flat)

    mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()

    if print_time: 
        print(f"[INFO] PSD Similarity: {time.time()-start_time} s")  

    return mmd, time.time()-start_time

def compute_mmd_rbf_tf(real, fake, sigma=1.0, print_time=False):
    """
    Compute the MMD with RBF kernel between two batches of samples using TensorFlow.
    """
    min_samples = min(real.shape[0], fake.shape[0])
    real = real[:min_samples]
    fake = fake[:min_samples]

    start_time = time.time()

    real_flat = tf.convert_to_tensor(real.reshape(min_samples, -1), dtype=tf.float32)
    fake_flat = tf.convert_to_tensor(fake.reshape(min_samples, -1), dtype=tf.float32)

    def rbf_kernel_tf(x, y):
        x_sq = tf.reduce_sum(tf.square(x), axis=1, keepdims=True)
        y_sq = tf.reduce_sum(tf.square(y), axis=1, keepdims=True)
        cross_term = tf.matmul(x, y, transpose_b=True)
        dists = x_sq - 2.0 * cross_term + tf.transpose(y_sq)
        return tf.exp(-dists / (2.0 * sigma ** 2))

    K_xx = rbf_kernel_tf(real_flat, real_flat)
    K_yy = rbf_kernel_tf(fake_flat, fake_flat)
    K_xy = rbf_kernel_tf(real_flat, fake_flat)

    # Drop diagonal to avoid bias in MMD
    N = tf.cast(tf.shape(real_flat)[0], tf.float32)
    def zero_diag(mat):
        return mat - tf.linalg.diag(tf.linalg.diag_part(mat))

    mmd = (
        tf.reduce_sum(zero_diag(K_xx)) / (N * (N - 1)) +
        tf.reduce_sum(zero_diag(K_yy)) / (N * (N - 1)) -
        2.0 * tf.reduce_mean(K_xy)
    )

    elapsed_time = time.time() - start_time
    if print_time:
        print(f"[INFO] MMD Computation Time: {elapsed_time:.4f} s")

    return mmd.numpy(), elapsed_time


def compute_frechet_distance(real, fake, print_time=False, n_components=256):
    """
    FID computation, PCA is used to speed things up
    """
    start_time = time.time()

    # Flatten and apply PCA
    real_flat = real.reshape(real.shape[0], -1)
    fake_flat = fake.reshape(fake.shape[0], -1)

    pca = PCA(n_components=n_components).fit(real_flat)
    real_pca = pca.transform(real_flat)
    fake_pca = pca.transform(fake_flat)

    # Compute means and covariances on PCA-projected data
    mu_real = np.mean(real_pca, axis=0)
    mu_fake = np.mean(fake_pca, axis=0)
    sigma_real = np.cov(real_pca, rowvar=False)
    sigma_fake = np.cov(fake_pca, rowvar=False)

    # Matrix square root
    covmean = sqrtm(sigma_real @ sigma_fake)
    if np.iscomplexobj(covmean):
        covmean = covmean.real  # Fix small imaginary parts

    fid = np.sum((mu_real - mu_fake)**2) + np.trace(sigma_real + sigma_fake - 2 * covmean)

    elapsed = time.time() - start_time
    if print_time:
        print(f"[INFO] Fréchet Distance Time: {elapsed:.4f} s")

    return fid, elapsed
 

def compute_ks_p_value(real, fake, print_time=False, n_jobs=-1):
    """
    KS test p-values across all channels.
    """
    C = real.shape[-1]
    start_time = time.time()

    def compute_p(ch):
        real_vals = real[:, :, ch].ravel()
        fake_vals = fake[:, :, ch].ravel()
        return ks_2samp(real_vals, fake_vals)[1]  # return p-value

    p_values = Parallel(n_jobs=n_jobs)(
        delayed(compute_p)(ch) for ch in range(C)
    )

    if print_time:
        print(f"[INFO] KS Test Time: {elapsed:.4f}s")

    return np.mean(p_values), time.time() - start_time



def perform_and_save_analysis(
        model_name,
        model_type, 
        epoch, 
        generations_name, 
        train_data_file,
        train_labels_file,
        val_data_file,
        val_labels_file,
        test_data_file,
        test_labels_file,
        working_dir,
        num_iterations = 100,
        seed = 1729
    ):

    if model_name is None:
        raise ValueError("You must specify an --model_name to save the output.")

    if model_type == "student":
        model_type_cap = "Student"
    elif model_type == "original":
        model_type_cap = "Original"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    np.random.seed(seed)

    # Create output directory
    output_dir = os.path.join(working_dir, 'student-radar', 'paper', 'data', 'analysis', model_type, f'model-{model_name}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Output directory: {output_dir}")

    # Get loss curves from wandb
    # wandb.login(key="85019a9c3b05de9fa0211a19fd654750ad845f1f")
    api = wandb.Api()
    run = api.run(f"spcorum-aerostar-international/student-generative-radar/{model_name}")
    df = run.history(keys=["epoch", "val/generator_loss", "val/discriminator_loss"], samples=5000)

    # Drop rows with missing values in either loss
    df_clean = df.dropna(subset=["val/generator_loss", "val/discriminator_loss"])

    # Total unique epochs (may include invalid ones)
    total_epochs = df["epoch"].nunique()

    # Valid epoch numbers where both generator and discriminator loss are present
    valid_epochs = df_clean["epoch"].unique()
    valid_epochs = sorted(valid_epochs)  # sort for plotting

    # Extract loss values for valid epochs only
    generator_loss_values = df_clean["val/generator_loss"].values
    discriminator_loss_values = df_clean["val/discriminator_loss"].values

    # Print summary
    print(f"[INFO] Total unique epochs: {total_epochs}")
    print(f"[INFO] Valid epochs with both losses present: {len(valid_epochs)}")
    if len(valid_epochs) == len(generator_loss_values) == len(discriminator_loss_values):
        pass
    else:
        print(f"[INFO] Mismatch in lengths of epoch and loss arrays")

    # Plot loss curves
    plt.plot(df["epoch"], df["val/generator_loss"], label="Generator Val Loss")
    plt.plot(df["epoch"], df["val/discriminator_loss"], label="Discriminator Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    title_list = ["Validation Loss Curves", model_type_cap, "Model", model_name]
    title_string = " ".join(title_list)
    plt.title(title_string)
    plt.grid(True)
    loss_plot_path = os.path.join(output_dir, "loss_plot.png")
    plt.savefig(loss_plot_path, dpi=600)
    print(f"[INFO] Loss plot saved to {loss_plot_path}")

    # Generations file paths
    generations_dir = os.path.join(working_dir, 'student-radar', 'paper', 'data', 'generated', model_type, f'model-{model_name}', generations_name)
    generations_data_file = os.path.join(generations_dir, 'generator_data.npy')
    generations_labels_file = os.path.join(generations_dir, 'generator_labels.npy')
    
    # Load data
    train_data = np.load(train_data_file)
    val_data = np.load(val_data_file)
    test_data = np.load(test_data_file)
    gen_data = np.load(generations_data_file)
    gen_data = np.transpose(gen_data, (0, 2, 1)) 

    print(f"[INFO] Train Data: {train_data.shape}")
    print(f"[INFO] Validation Data: {val_data.shape}")
    print(f"[INFO] Test Data: {test_data.shape}")
    print(f"[INFO] Generated Data]: {gen_data.shape}")

    mean_train = float(np.mean(train_data))
    sd_train = float(np.std(train_data))
    mean_val = float(np.mean(val_data))
    sd_val = float(np.std(val_data))
    mean_test = float(np.mean(test_data))
    sd_test = float(np.std(test_data))
    mean_gen = float(np.mean(gen_data))
    sd_gen = float(np.std(gen_data))

    print(f"[INFO] Train Data: mean = {mean_train:.4f}, sd = {sd_train:.4f}, shape = {train_data.shape}")
    print(f"[INFO] Validation Data: mean = {mean_val:.4f}, sd = {sd_val:.4f}, shape = {val_data.shape}")
    print(f"[INFO] Test Data: mean = {mean_test:.4f}, sd = {sd_test:.4f}, shape = {test_data.shape}")
    print(f"[INFO] Generated Data]: mean = {mean_gen:.4f}, sd = {sd_gen:.4f}, shape = {gen_data.shape}")

    # Initialize lists for storing results

    # Channel-wise statistics
    channel_mean_list_train = []
    channel_mean_list_val = []
    channel_mean_list_test = []
    channel_std_list_train = []
    channel_std_list_val = []
    channel_std_list_test = []

    # PSD MSE
    psd_mse_mean_list_train = []
    psd_mse_mean_list_val = []
    psd_mse_mean_list_test = []
    psd_mse_std_list_train = []
    psd_mse_std_list_val = []
    psd_mse_std_list_test = []

    # PSD Correlation
    psd_corr_mean_list_train = []
    psd_corr_mean_list_val = []
    psd_corr_mean_list_test = []
    psd_corr_std_list_train = []
    psd_corr_std_list_val = []
    psd_corr_std_list_test = []

    # PSD L2 Distance
    psd_l2_mean_list_train = []
    psd_l2_mean_list_val = []
    psd_l2_mean_list_test = []

    # MMD RBF
    mmd_rbf_list_train = []
    mmd_rbf_list_val = []
    mmd_rbf_list_test = []

    # KS Average P-Value
    ks_avg_p_list_train = []
    ks_avg_p_list_val = []
    ks_avg_p_list_test = []

    # FID
    fid_train_list = []
    fid_val_list = []
    fid_test_list = []

    # Analysis loop
    skip_val = 1
    start_time = time.time()
    prev_time = start_time
    for i in range(num_iterations):

        iteration_start = time.time()

        # Sample the training, validation, and generation data
        gen_indices_for_train = np.random.choice(gen_data.shape[0], size=train_data.shape[0], replace=False)
        gen_indices_for_val = np.random.choice(gen_data.shape[0], size=val_data.shape[0], replace=False)
        gen_indices_for_test = np.random.choice(gen_data.shape[0], size=test_data.shape[0], replace=False)
        sampled_gen_data_for_train = gen_data[gen_indices_for_train]
        sampled_gen_data_for_val = gen_data[gen_indices_for_val]
        sampled_gen_data_for_test = gen_data[gen_indices_for_test]

        # Calculate metrics on all data
        # Channel-wise statistics
        channel_mean_train, channel_std_train, time1 = channelwise_statistics(train_data, sampled_gen_data_for_train)
        channel_mean_val, channel_std_val, time2 = channelwise_statistics(val_data, sampled_gen_data_for_val)
        channel_mean_test, channel_std_test, time3 = channelwise_statistics(test_data, sampled_gen_data_for_test)

        print(f"[INFO] Iteration {i + 1} Channelwise stats computation time = {time1+time2+time3:.2f} seconds")

        # PSD similarity (MSE + Pearson correlation + L2 distance)
        psd_mse_mean_train, psd_mse_std_train, psd_corr_mean_train, psd_corr_std_train, psd_l2_mean_train, time1 = \
            compute_psd_similarity(train_data, sampled_gen_data_for_train)
        psd_mse_mean_val, psd_mse_std_val, psd_corr_mean_val, psd_corr_std_val, psd_l2_mean_val, time2 = \
            compute_psd_similarity(val_data, sampled_gen_data_for_val)
        psd_mse_mean_test, psd_mse_std_test, psd_corr_mean_test, psd_corr_std_test, psd_l2_mean_test, time3 = \
            compute_psd_similarity(test_data, sampled_gen_data_for_test)

        print(f"[INFO] Iteration {i + 1} PSD computation time = {time1+time2+time3:.2f} seconds")

        # MMD using RBF kernel
        mmd_rbf_train, time1 = compute_mmd_rbf_tf(train_data, sampled_gen_data_for_train)
        mmd_rbf_val, time2 = compute_mmd_rbf_tf(val_data, sampled_gen_data_for_val)
        mmd_rbf_test, time3 = compute_mmd_rbf_tf(test_data, sampled_gen_data_for_test)

        print(f"[INFO] Iteration {i + 1} MMF (RBF) computation time = {time1+time2+time3:.2f} seconds")

        # KS test average p-value
        ks_avg_p_train, time2 = compute_ks_p_value(train_data, sampled_gen_data_for_train)
        ks_avg_p_val, time2 = compute_ks_p_value(val_data, sampled_gen_data_for_val)
        ks_avg_p_test, time3 = compute_ks_p_value(test_data, sampled_gen_data_for_test)

        print(f"[INFO] Iteration {i + 1} KS test computation time = {time1+time2+time3:.2f} seconds")

        # Fréchet distance
        fid_train, time1 = compute_frechet_distance(train_data, sampled_gen_data_for_train)
        fid_val, time2 = compute_frechet_distance(val_data, sampled_gen_data_for_val)
        fid_test, time3 = compute_frechet_distance(test_data, sampled_gen_data_for_test) 
        print(f"[INFO] Iteration {i + 1} FID computation time = {time1+time2+time3:.2f} seconds")       

        ## Collect all results

        # Channel-wise mean and std
        channel_mean_list_train.append(channel_mean_train)
        channel_std_list_train.append(channel_std_train)
        channel_mean_list_val.append(channel_mean_val)
        channel_std_list_val.append(channel_std_val)
        channel_mean_list_test.append(channel_mean_test)
        channel_std_list_test.append(channel_std_test)

        # PSD MSE and correlation
        psd_mse_mean_list_train.append(psd_mse_mean_train)
        psd_mse_std_list_train.append(psd_mse_std_train)
        psd_corr_mean_list_train.append(psd_corr_mean_train)
        psd_corr_std_list_train.append(psd_corr_std_train)

        psd_mse_mean_list_val.append(psd_mse_mean_val)
        psd_mse_std_list_val.append(psd_mse_std_val)
        psd_corr_mean_list_val.append(psd_corr_mean_val)
        psd_corr_std_list_val.append(psd_corr_std_val)

        psd_mse_mean_list_test.append(psd_mse_mean_test)
        psd_mse_std_list_test.append(psd_mse_std_test)
        psd_corr_mean_list_test.append(psd_corr_mean_test)
        psd_corr_std_list_test.append(psd_corr_std_test)

        # PSD L2 distance
        psd_l2_mean_list_train.append(psd_l2_mean_train)
        psd_l2_mean_list_val.append(psd_l2_mean_val)
        psd_l2_mean_list_test.append(psd_l2_mean_test)

        # MMD RBF
        mmd_rbf_list_train.append(mmd_rbf_train)
        mmd_rbf_list_val.append(mmd_rbf_val)
        mmd_rbf_list_test.append(mmd_rbf_test)

        # KS Avg P-value
        ks_avg_p_list_train.append(ks_avg_p_train)
        ks_avg_p_list_val.append(ks_avg_p_val)
        ks_avg_p_list_test.append(ks_avg_p_test)

        # FID
        fid_train_list.append(fid_train)
        fid_val_list.append(fid_val)
        fid_test_list.append(fid_test)

        iteration_end = time.time()
        iteration_time = iteration_end - prev_time
        prev_time = iteration_end


        print(f"[INFO] Iteration {i + 1}: Elapsed time = {iteration_time:.2f} seconds")


    # Channel-wise Mean and Std
    channel_mean_train = np.mean(channel_mean_list_train)
    channel_std_train = np.mean(channel_std_list_train)
    channel_mean_val = np.mean(channel_mean_list_val)
    channel_std_val = np.mean(channel_std_list_val)
    channel_mean_test = np.mean(channel_mean_list_test)
    channel_std_test = np.mean(channel_std_list_test)

    # PSD MSE
    psd_mse_mean_train = np.mean(psd_mse_mean_list_train)
    psd_mse_std_train = np.mean(psd_mse_std_list_train)
    psd_mse_mean_val = np.mean(psd_mse_mean_list_val)
    psd_mse_std_val = np.mean(psd_mse_std_list_val)
    psd_mse_mean_test = np.mean(psd_mse_mean_list_test)
    psd_mse_std_test = np.mean(psd_mse_std_list_test)

    # PSD Correlation
    psd_corr_mean_train = np.mean(psd_corr_mean_list_train)
    psd_corr_std_train = np.mean(psd_corr_std_list_train)
    psd_corr_mean_val = np.mean(psd_corr_mean_list_val)
    psd_corr_std_val = np.mean(psd_corr_std_list_val)
    psd_corr_mean_test = np.mean(psd_corr_mean_list_test)
    psd_corr_std_test = np.mean(psd_corr_std_list_test)

    # PSD L2 Distance
    psd_l2_mean_train = np.mean(psd_l2_mean_list_train)
    psd_l2_mean_val = np.mean(psd_l2_mean_list_val)
    psd_l2_mean_test = np.mean(psd_l2_mean_list_test)

    # MMD RBF
    mmd_rbf_mean_train = np.mean(mmd_rbf_list_train)
    mmd_rbf_mean_val = np.mean(mmd_rbf_list_val)
    mmd_rbf_mean_test = np.mean(mmd_rbf_list_test)

    # KS Average P-value
    ks_avg_p_train = np.mean(ks_avg_p_list_train)
    ks_avg_p_val = np.mean(ks_avg_p_list_val)
    ks_avg_p_test = np.mean(ks_avg_p_list_test)

    # FID
    fid_mean_train = np.mean(fid_train_list)
    fid_mean_val = np.mean(fid_val_list)
    fid_mean_test = np.mean(fid_test_list)

    # Collect results in a dict
    results = {
        "channel_mean": {
            "train": channel_mean_train,
            "val": channel_mean_val,
            "test": channel_mean_test,
        },
        "channel_std": {
            "train": channel_std_train,
            "val": channel_std_val,
            "test": channel_std_test,
        },
        "psd_mse_mean": {
            "train": psd_mse_mean_train,
            "val": psd_mse_mean_val,
            "test": psd_mse_mean_test,
        },
        "psd_mse_std": {
            "train": psd_mse_std_train,
            "val": psd_mse_std_val,
            "test": psd_mse_std_test,
        },
        "psd_corr_mean": {
            "train": psd_corr_mean_train,
            "val": psd_corr_mean_val,
            "test": psd_corr_mean_test,
        },
        "psd_corr_std": {
            "train": psd_corr_std_train,
            "val": psd_corr_std_val,
            "test": psd_corr_std_test,
        },
        "psd_l2_mean": {
            "train": psd_l2_mean_train,
            "val": psd_l2_mean_val,
            "test": psd_l2_mean_test,
        },
        "mmd_rbf": {
            "train": mmd_rbf_train,
            "val": mmd_rbf_val,
            "test": mmd_rbf_test,
        },
        "ks_avg_p_value": {
            "train": ks_avg_p_train,
            "val": ks_avg_p_val,
            "test": ks_avg_p_test,
        },
        "fid": {
            "train": fid_mean_train,
            "val": fid_mean_val,
            "test": fid_mean_test,
        },
        "mean": {
            "train": mean_train,
            "val": mean_val,
            "test": mean_test,
            "generated": mean_gen
        },
        "std": {
            "train": sd_train,
            "val": sd_val,
            "test": sd_test,
            "generated": sd_gen
        },
        "meta": {
            "model_name": model_name,
            "model_type": model_type,
            "num_total_epochs": int(df["epoch"].nunique()),
            "num_valid_epochs": int(len(valid_epochs)),
            "valid_epoch_numbers": list(valid_epochs),
            "generator_loss_values": generator_loss_values.tolist(),
            "discriminator_loss_values": discriminator_loss_values.tolist()
        }
    }

    # Define the output path
    results_path = os.path.join(output_dir, "results.json")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4, default=lambda o: float(o) if isinstance(o, np.floating) else int(o) if isinstance(o, np.integer) else str(o))   

    print(f"[INFO] Results saved to: {results_path}")

    print_results_table(results)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Model identifier (e.g., wandb run ID)", default = None)
    parser.add_argument("--generations_name", type=str, help="Generation filename (without extension)", default = "generations_10000")
    parser.add_argument("--num_iterations", type=int, default=100, help="Number of repeated iterations to perform analysis")
    parser.add_argument('--model_type', type=str,choices=['student', 'original'], default='student')
    parser.add_argument("--epoch", type=int, help="Epoch number to load weights from", default = -1)
    parser.add_argument('--train_data_file', type=str, default='/content/drive/MyDrive/cs231n/student-radar/paper/data/preprocessed/EXP_17_M_chirps_scaled_train_reduced.npy', help='Path to training data')
    parser.add_argument('--train_labels_file', type=str, default='/content/drive/MyDrive/cs231n/student-radar/paper/data/preprocessed/EXP_17_M_chirps_labels_train_reduced.npy', help='Path to training labels')
    parser.add_argument('--val_data_file', type=str, default='/content/drive/MyDrive/cs231n/student-radar/paper/data/preprocessed/EXP_17_M_chirps_scaled_validation.npy', help='Path to validation data')
    parser.add_argument('--val_labels_file', type=str, default='/content/drive/MyDrive/cs231n/student-radar/paper/data/preprocessed/EXP_17_M_chirps_labels_validation.npy', help='Path to validation labels')
    parser.add_argument('--test_data_file', type=str, default='/content/drive/MyDrive/cs231n/student-radar/paper/data/preprocessed/EXP_17_M_chirps_scaled_test.npy', help='Path to test data')
    parser.add_argument('--test_labels_file', type=str, default='/content/drive/MyDrive/cs231n/student-radar/paper/data/preprocessed/EXP_17_M_chirps_labels_test.npy', help='Path to test labels')
    parser.add_argument('--working_dir', type=str, default='/content/drive/MyDrive/cs231n/', help='Base working directory')
    parser.add_argument('--seed', type=int, default=1729, help='Random seed')
    args = parser.parse_args()
    perform_and_save_analysis(
        model_name=args.model_name,
        model_type=args.model_type, 
        epoch=args.epoch, 
        generations_name=args.generations_name,
        train_data_file=args.train_data_file,
        train_labels_file=args.train_labels_file,
        val_data_file=args.val_data_file,
        val_labels_file=args.val_labels_file,
        test_data_file=args.test_data_file,
        test_labels_file=args.test_labels_file,
        working_dir=args.working_dir,
        num_iterations=args.num_iterations,
        seed=args.seed
    )

if __name__ == "__main__":
    main()