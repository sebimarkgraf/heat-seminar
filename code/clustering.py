import random
import matplotlib.pyplot as plt
from mpi4py import MPI
import heat as ht
import os
import time
from sklearn import metrics


NUM_CLUSTERS = 17
DATASET = 'sen1'
PLOT_DIR = './plots'
SUBSET='validation'

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

print(f"{rank} New Shape: {dataset.shape}")
print("Done")


print("Clustering")
# c = ht.cluster.KMeans(n_clusters=NUM_CLUSTERS, init="kmeans++", max_iter=1000)
c = ht.cluster.Spectral(n_clusters=NUM_CLUSTERS, n_lanczos=300, metric='rbf')
labels_pred = c.fit_predict(dataset).squeeze()
print("Clustering done")

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.hist(labels_pred.numpy())
ax1.set_title("Predicted")
ax2.hist(labels.numpy())
ax2.set_title("True Labels")
if rank == 0:
    #print(labels_sub)
    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.savefig(f"{PLOT_DIR}/{DATASET}_{SUBSET}_{NUM_CLUSTERS}_{time.time()}_Label_Count.png")


def print_metrics(labels, labels_pred):
    print(f"Adjusted Rand Index: {metrics.adjusted_rand_score(labels, labels_pred)} (1.0)")
    print(f"Mutual Information Score: {metrics.adjusted_mutual_info_score(labels, labels_pred)} (1.0)")
    print(f"V-measure: {metrics.v_measure_score(labels, labels_pred)} (1.0)")


print_metrics(labels.numpy(), labels_pred.numpy())