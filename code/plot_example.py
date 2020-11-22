import heat as ht
import matplotlib.pyplot as plt
import os
import numpy as np

PLOT_DIR = './plots/examples'
DATASET = 'sen1'
EXAMPLE = 20
SUBSET = 'validation'

def plot_example(data, ax, title: str = 'Satellite Example'):
    ax.imshow(data,cmap=plt.cm.get_cmap('gray'))
    ax.set_title(title)
    ax.axis('off')


# Plot All Channels
data = ht.load(f"./data/{SUBSET}.h5", dataset=DATASET)[EXAMPLE]
data = np.moveaxis(data.numpy(), -1, 0)
print(data.shape)

labels = ht.load(f"./data/{SUBSET}.h5", dataset="label")[EXAMPLE]
label = ht.argmax(labels).numpy()[0]


#print(labels_sub)
os.makedirs(PLOT_DIR, exist_ok=True)

channels = data.shape[0]

rows = 2
cols = channels // rows
fig, axes = plt.subplots(nrows=rows, ncols=cols)
axes = axes.flatten()

for index, channel in enumerate(data):    
    plot_example(channel, axes[index], f'Channel {index}')

fig.suptitle(f'Example {EXAMPLE} from {DATASET} Zone {label}')
plt.savefig(f"{PLOT_DIR}/{DATASET}_{SUBSET}_{EXAMPLE}_CHANNELS.png", bbox_inches='tight')
plt.close(fig)

