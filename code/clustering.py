import logging
import math
import os
import random
import time
from functools import wraps
from typing import Tuple

import heat as ht
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import wandb
from flatten import heat_flatten
from mpi4py import MPI
from sklearn import metrics

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
FORMAT = "[%(asctime)s]: %(message)s"
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
formatter = logging.Formatter(FORMAT, TIME_FORMAT)
ch.setFormatter(formatter)
logger.addHandler(ch)

TIMESTAMP = time.time()


def only_root(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        response = None
        if MPI.COMM_WORLD.Get_rank() == 0:
            response = func(*args, **kwargs)
        return response

    return wrapped


def load_dataset(data_path: str, subset: str, dataset: str) -> ht.dndarray:
    path = f"{data_path}{subset}.h5"
    return ht.load(path, dataset=dataset, split=0)


def take_percentage(dataset: ht.DNDarray, percentage: float) -> ht.DNDarray:
    if np.isclose(percentage, 1.0):
        return dataset

    elements = int(dataset.shape[0] * percentage)

    return dataset[:elements]


def calc_weak_scaling_percentage(num_processes: int, max_processes: int) -> float:
    return (num_processes / max_processes) ** 2


def load_data(datapath: str, subset: str, dataset: str, percentage: float):
    # Load dataset from hdf5 file
    dataset = load_dataset(data_path=datapath, subset=subset, dataset=dataset)
    labels = load_dataset(data_path=datapath, subset=subset, dataset="label")
    dataset = take_percentage(dataset, percentage)
    labels = take_percentage(labels, percentage)
    dataset.balance_()
    labels.balance_()
    return dataset, labels


def normalize(dataset: ht.DNDarray):
    channel_mean = ht.mean(dataset, axis=(0, 1, 2))
    channel_std = ht.std(dataset, axis=(0, 1, 2))
    return (dataset - channel_mean) / channel_std


def flatten(dataset, labels):
    logger.debug(f"Reshaping the dataset {dataset.shape}")
    dataset = heat_flatten(dataset, start_dim=1)
    logger.debug(f"New shape {dataset.shape}")
    logger.debug("Getting max labels")
    labels = ht.argmax(labels, axis=1)
    logger.debug("Gathering the labels on root node")
    labels = ht.resplit(labels, axis=None).numpy()

    return dataset, labels


def cluster(dataset: ht.dndarray, config: dict) -> Tuple[ht.cluster.Spectral, np.array]:
    logger.debug("Starting clustering")
    c = ht.cluster.Spectral(
        n_clusters=config["n_clusters"],
        gamma=config["gamma"],
        metric=config["metric"],
        laplacian=config["laplacian"],
        n_lanczos=config["n_lanczos"],
        threshold=config["threshold"],
        boundary=config["boundary"],
        assign_labels=config["assign_labels"],
    )
    for i in range(config["num_fittings"]):
        logger.info(f"RUN {i} start")
        start = time.perf_counter()
        c = c.fit(dataset)
        stop = time.perf_counter()
        log_compute_time(start, stop)
        logger.info(f"RUN {i} finished in {stop-start:.2f}s")
    labels_pred = c.predict(dataset).squeeze()
    labels_pred = ht.resplit(labels_pred, axis=None).numpy()

    return c, labels_pred


@only_root
def plot_label_compare(labels: np.array, labels_pred: np.array, config: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.hist(labels_pred)
    ax1.set_title("Predicted")
    ax2.hist(labels)
    ax2.set_title("True Labels")
    os.makedirs("plots", exist_ok=True)

    wandb.log({"label_compare": fig})
    plt.savefig(
        f"plots/{config['dataset']}_{config['subset']}_{config['n_clusters']}_{TIMESTAMP}_Label_Count.png"
    )
    plt.close(fig)


@only_root
def plot_cluster_composition(labels: np.array, labels_pred: np.array, config: dict):
    bins = np.max(labels) + 1
    n_clusters = np.max(labels_pred) + 1

    cols = 4
    rows = n_clusters // cols + 1
    fig, axes = plt.subplots(nrows=rows, ncols=cols)
    axes = axes.flatten()

    for i in range(n_clusters):
        cluster_labels = labels[labels_pred == i]
        axes[i].hist(cluster_labels, range=(0, n_clusters), bins=bins)
        axes[i].set_title(f"Cluster {i}")

    fig.tight_layout()
    wandb.log({"cluster_composition": fig})

    os.makedirs("plots", exist_ok=True)
    plt.savefig(
        f"plots/{config['dataset']}_{config['subset']}_{config['n_clusters']}_{TIMESTAMP}_Cluster_Composition.png"
    )
    plt.close(fig)


@only_root
def log_metrics(labels: np.array, labels_pred: np.array, prefix=""):
    logged_metrics = {
        f"{prefix}Adjusted Rand Index": metrics.adjusted_rand_score(
            labels, labels_pred
        ),
        f"{prefix}Mutual Information Score": metrics.adjusted_mutual_info_score(
            labels, labels_pred
        ),
        f"{prefix}V-Measure": metrics.v_measure_score(labels, labels_pred),
    }
    wandb.log(logged_metrics)
    logger.info(logged_metrics)


@only_root
def log_eigenvalues(value: np.array):
    ev_table = wandb.Table({"eigenvalues": value})
    wandb.log({"eigenvalues": ev_table})


@only_root
def plot_confusion(labels: np.array, labels_pred: np.array):

    cm = metrics.confusion_matrix(labels, labels_pred)
    logger.info(cm)
    wandb.log({"conf_mat": wandb.plot.confusion_matrix(labels_pred, labels)})


@only_root
def init_wandb():
    wandb.init(project="satellite-heat")
    wandb.config["n_processes"] = size
    if not (wandb.config["weak_scaling_max"] == 0):
        logger.info("Weak Scaling run")
        wandb.config.update(
            {
                "dataset_percentage": calc_weak_scaling_percentage(
                    size, wandb.config["weak_scaling_max"]
                )
            },
            allow_val_change=True,
        )
    return wandb.config._as_dict()


@only_root
def log_compute_time(start, stop):
    wandb.log({"Computing Time": stop - start})


def main():
    ht.random.seed(1234)

    config = init_wandb()
    config = comm.bcast(config)

    logger.info(f"Config broadcasted {config}")
    dataset, labels = load_data(
        config["datapath"],
        config["subset"],
        dataset=config["dataset"],
        percentage=config["dataset_percentage"],
    )
    logger.info("Data loaded")
    if config["normalize"]:
        dataset = normalize(dataset)
        logger.info("Data normalized")
    dataset, labels = flatten(dataset, labels)
    logger.info("Data flattened")
    c, labels_pred = cluster(dataset, config=config)
    logger.info("Clustering finishes")

    logger.info(f"Shapes: {labels_pred.shape} {labels.shape}")
    log_metrics(labels, labels_pred)
    plot_label_compare(labels, labels_pred, config)
    plot_confusion(labels, labels_pred)
    plot_cluster_composition(labels, labels_pred, config)
    logger.info("Training Finished")

    logger.info("Getting eigenvalues")
    eigenvalues, eigenvectors = c._spectral_embedding(dataset)
    eigenvalues, _ = ht.sort(eigenvalues)[:100]
    eigenvalues = ht.stack(
        (ht.arange(eigenvalues.shape[0], split=eigenvalues.split), eigenvalues), axis=1
    )
    eigenvalues = ht.resplit(eigenvalues, axis=None).numpy()
    log_eigenvalues("eigenvalues")
    logger.info("Sent eigenvalues")

    val_data, val_labels = load_data(
        config["datapath"], "validation", dataset=config["dataset"], percentage=1.0
    )
    if config["normalize"]:
        val_data = normalize(val_data)
        logger.info("Val Data normalized")
    val_data, val_labels = flatten(val_data, val_labels)
    val_pred = c.predict(val_data).squeeze()
    val_pred = ht.resplit(val_pred, axis=None).numpy()

    logger.info("Log Validation result")
    log_metrics(val_labels, val_pred, prefix="val: ")
    logger.info("Finished")


if __name__ == "__main__":
    main()
