import random
import matplotlib.pyplot as plt
from mpi4py import MPI
import heat as ht
import os
import time
from sklearn import metrics
import numpy as np


NUM_CLUSTERS = 17
DATASET = 'sen2'
PLOT_DIR = './plots'
SUBSET='validation'

TIMESTAMP = time.time()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

ht.random.seed(1234)

def print_root(*values: object):
    if rank == 0:
        print(*values)

def load_dataset(subset: str, dataset: str):
    path = f"./data/{subset}.h5"
    return ht.load(path, dataset=dataset, split=0)


def take_subset(data: ht.dndarray, num_items=1000):
    new_data = data[0:num_items]
    new_data.balance_()
    return new_data

print("Loading Dataset")
# Load dataset from hdf5 file
dataset = take_subset(load_dataset(subset=SUBSET, dataset=DATASET))
labels = take_subset(load_dataset(subset=SUBSET, dataset="label"))
print("Done")

print("Z-Score Normalization")
channel_mean = ht.mean(dataset, axis=(0, 1, 2))
channel_std = ht.std(dataset, axis=(0, 1, 2))
print(channel_mean.numpy(), channel_std.numpy())
dataset = (dataset - channel_mean) / channel_std

# Flatten the images and channels into features
print(f"{rank} Reshaping the features")
print(f"{rank} {dataset.shape}")

dataset = ht.reshape(dataset, (dataset.shape[0], dataset.shape[1] * dataset.shape[2] * dataset.shape[3]))
labels = ht.argmax(labels, axis=1)
labels = ht.resplit(labels, axis=None).numpy()

print(f"{rank} New Shape: {dataset.shape}")
print("Done")


print("Clustering")
# c = ht.cluster.KMeans(n_clusters=NUM_CLUSTERS, init="kmeans++", max_iter=1000)
c = ht.cluster.Spectral(n_clusters=NUM_CLUSTERS, n_lanczos=300, metric='rbf')
labels_pred = c.fit_predict(dataset).squeeze()
labels_pred = ht.resplit(labels_pred, axis=None).numpy()
print("Clustering done")

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.hist(labels_pred)
ax1.set_title("Predicted")
ax2.hist(labels)
ax2.set_title("True Labels")
if rank == 0:
    #print(labels_sub)
    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.savefig(f"{PLOT_DIR}/{DATASET}_{SUBSET}_{NUM_CLUSTERS}_{TIMESTAMP}_Label_Count.png")

def plot_cluster_composition(labels: np.array, labels_pred: np.array):
    bins = np.max(labels)
    n_clusters = np.max(labels_pred)

    cols = 4
    rows = n_clusters // cols + 1
    fig, axes = plt.subplots(nrows=rows, ncols=cols)
    axes = axes.flatten()

    for i in range(n_clusters):
        cluster_labels = labels[labels_pred == i]
        axes[i].hist(cluster_labels, range=(0, n_clusters), bins=bins)
        axes[i].set_title(f'Cluster {i}')

    fig.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{DATASET}_{SUBSET}_{NUM_CLUSTERS}_{TIMESTAMP}_Cluster_Composition.png")
    plt.close(fig)

plot_cluster_composition(labels, labels_pred)




def print_metrics(labels, labels_pred):
    print(f"Adjusted Rand Index: {metrics.adjusted_rand_score(labels, labels_pred)} (1.0)")
    print(f"Mutual Information Score: {metrics.adjusted_mutual_info_score(labels, labels_pred)} (1.0)")
    print(f"V-measure: {metrics.v_measure_score(labels, labels_pred)} (1.0)")


print_metrics(labels, labels_pred)