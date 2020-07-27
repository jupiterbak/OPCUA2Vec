import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ampligraph.datasets import load_from_csv
from ampligraph.discovery import find_clusters
from ampligraph.evaluation import train_test_split_no_unseen
from ampligraph.utils import restore_model
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import seaborn as sns

GENERAGE_TSNE = False
DATASET_LOCATION = 'data/'
DATASET_FILE = 'dataOpcua-DATASETONE-all.txt'
DATASET_BROWSE_NAME_FILE = 'dataOpcua-DATASETONE-BrowseNameMap.txt'
DATASET_BROWSE_NAME_COLOR_FILE = 'dataOpcua-DATASETONE-BrowseNameMap_color.txt'
RESULT_EXPORT_LOCATION = 'export/DATASETONE/'
MODEL_FILE = 'opcua_autoTransE.pkl'
TOP_NODES = [
    "ns=0;i=84",
    "ns=0;i=16243",
    "ns=0;i=85",
    "ns=0;i=86",
    "ns=0;i=88",
    "ns=0;i=89",
    "ns=0;i=91",
    "ns=0;i=90",
    "ns=0;i=17708",
    "ns=0;i=16304",
    "ns=0;i=14100"

]


def set_value(row_number, assigned_value):
    return assigned_value[row_number]


X = load_from_csv(DATASET_LOCATION, DATASET_FILE, sep='\t')

# Train test split
X_train, X_test = train_test_split_no_unseen(X, test_size=1000)

# Restore the model
restored_model = restore_model(model_name_path=RESULT_EXPORT_LOCATION + MODEL_FILE)

# Get the teams entities and their corresponding embeddings
triples_df = pd.DataFrame(X, columns=['s', 'p', 'o'])
uniques = triples_df.s.unique()
uniques_embeddings = dict(zip(uniques, restored_model.get_embeddings(uniques)))
uniques_embeddings_array = np.array([i for i in uniques_embeddings.values()])

# get the labels
labels = load_from_csv(DATASET_LOCATION, DATASET_BROWSE_NAME_COLOR_FILE, sep='\t')
labels_df = {labels[i][0]: labels[i][1] for i in range(len(labels))}
colors_df = {labels[i][0]: labels[i][2] for i in range(len(labels))}
unique_labels = [labels_df.get(e, "Root") for e in uniques]
unique_colors = [colors_df.get(e, "Root") for e in uniques]

# Find clusters of embeddings using KMeans
kmeans = KMeans(n_clusters=4, n_init=100, max_iter=500)
clusters = find_clusters(uniques, restored_model, kmeans, mode='entity')

# Project embeddings into 2D space via PCA
if GENERAGE_TSNE is True:
    embeddings_2d = TSNE(n_components=2, perplexity=30.0).fit_transform(uniques_embeddings_array)  # opcua_autoTransE.pkl
    # save embedding 2D
    with open(RESULT_EXPORT_LOCATION + 'TSNE_2D.npy', 'wb') as f:
        np.save(f, embeddings_2d)
else:
    with open(RESULT_EXPORT_LOCATION + 'TSNE_2D.npy', 'rb') as f:
        embeddings_2d = np.load(f)

# PCA(n_components=2).fit_transform(uniques_embeddings_array)

# get the annotation
annotation_nodes = TOP_NODES
annotation_indexes = [list(uniques_embeddings.keys()).index(e) for e in annotation_nodes]
annotations = [dict(text=  labels_df.get(uniques[i], "Root") , x=embeddings_2d[i, 0], y=embeddings_2d[i, 1]) for k, i in
               enumerate(annotation_indexes)] # + " (" + TOP_NODES[k] + ")"

plot_df = pd.DataFrame({"uniques": uniques,
                        "clusters": pd.Series(clusters).astype(str),
                        "Entity type": unique_colors,
                        "embedding1": embeddings_2d[:, 0],
                        "embedding2": embeddings_2d[:, 1],
                        "label": unique_labels})

# Configure the Plot Style
plt.style.use('classic')
fig, ax = plt.subplots()

# # sns.set_style("whitegrid")
# # sns.set_style("white")
# ax = sns.scatterplot(data=plot_df, x="embedding1", y="embedding2", hue="Entity type", alpha=0.6, marker='+',
#                      edgecolor=None, s=20)
# sns.despine()

color_dictionary = {
    'Variable': 'teal',
    'ObjectType': 'b',
    'Object': 'orange',
    'DataType': 'r',
    'Method': 'g',
    "VariableType": 'yellow',
    'Root': 'maroon'
}
# plot_df["colors"] = plot_df['Entity type'].apply(set_value, args=(color_dictionary, ))

for k, v in color_dictionary.items():
    plot_df_sub = plot_df[plot_df['Entity type'] == k]
    ax.scatter(plot_df_sub['embedding1'], plot_df_sub['embedding2'], s=30, c=v, alpha=0.8, marker='o',
               label=k, facecolor='0.6', lw=0.8)
    # ax = plot_df_sub.plot.scatter(x='embedding1', y='embedding2', c=v,  s=20, ax=ax) # colormap='tab10'

#### SET ZOOM EFFECT 1
# Set the zoom out
axisw = zoomed_inset_axes(ax, 12, loc=4)
for k, v in color_dictionary.items():
    plot_df_sub = plot_df[plot_df['Entity type'] == k]
    axisw.scatter(plot_df_sub['embedding1'], plot_df_sub['embedding2'], s=50, c=v, alpha=0.9, marker='o',
                  edgecolor=None, label=k, facecolor='0.6', lw=0.6)

# add annotations one by one with a loop
for i, an in enumerate(annotations):
    _loc = (10, -20)
    if i == 0:
        _loc = (10, 40)
    elif i == 2:
        _loc = (-50, 40)
    elif i == 3:
        _loc = (50, 20)
    elif i == 4:
        _loc = (60, 0)
    elif i == 5:
        _loc = (70, -20)
    elif i == 6:
        _loc = (-50, -60)
    elif i == 7:
        _loc = (0, 30)
    elif i == 8:
        _loc = (70, -70)

    axisw.annotate(an['text'], xy=(an['x'], an['y']), xycoords='data',
                   xytext=_loc, textcoords='offset points', ha='center',
                   bbox=dict(boxstyle="round", fc="0.8"),
                   arrowprops=dict(arrowstyle="->"))

# Specify the limits of Zoom In
axisw.set_xlim(annotations[0]['x'] - 2, annotations[0]['x'] + 2)  # apply the x-limits
axisw.set_ylim(annotations[0]['y'] - 4, annotations[0]['y'] + 2)  # apply the y-limits
plt.setp(axisw.get_xticklabels(), visible=False)
plt.setp(axisw.get_yticklabels(), visible=False)
# Add Zom Effect
mark_inset(ax, axisw, loc1=1, loc2=3, fc="none", ec="0.5")

#### SET ZOOM EFFECT 2
# Set the zoom out
axine = zoomed_inset_axes(ax, 4, loc=2)
for k, v in color_dictionary.items():
    plot_df_sub = plot_df[plot_df['Entity type'] == k]
    axine.scatter(plot_df_sub['embedding1'], plot_df_sub['embedding2'], s=50, c=v, alpha=0.9, marker='o',
                  edgecolor=None, label=k, facecolor='0.6', lw=0.6)

# add annotations one by one with a loop
for i, an in enumerate(annotations):
    _loc = (10, -30)
    if i == 1:
        _loc = (-20, 20)

    axine.annotate(an['text'], xy=(an['x'], an['y']), xycoords='data',
                   xytext=_loc, textcoords='offset points', ha='center',
                   bbox=dict(boxstyle="round", fc="0.8"),
                   arrowprops=dict(arrowstyle="->"))


# Specify the limits of Zoom In
axine.set_xlim(annotations[1]['x'] - 4, annotations[1]['x'] + 10)  # apply the x-limits
axine.set_ylim(annotations[1]['y'] - 10, annotations[1]['y'] + 4)  # apply the y-limits
plt.setp(axine.get_xticklabels(), visible=False)
plt.setp(axine.get_yticklabels(), visible=False)


# Add Zom Effect
mark_inset(ax, axine, loc1=1, loc2=4, fc="none", ec="0.5")

ax.set_xlabel('t-SNE Feature 1')
ax.set_ylabel('t-SNE Feature 2')
ax.grid(ls='--')
ax.legend(framealpha=1, frameon=True, loc='best')

# plt.title("Visualizing TSNE")
fig.set_size_inches(16, 9, forward=True)
fig.tight_layout()


_save_path_pdf = os.path.normpath('{}/{}_export.pdf'.format(RESULT_EXPORT_LOCATION, MODEL_FILE))
plt.savefig(_save_path_pdf, dpi=300)
plt.show()

# color_dictionary ={
#     'Variable': 'r',
#     'ObjectType': 'g',
#     'Object': 'b',
#     'DataType': 'b',
#     'Method': 'c',
#     "VariableType": 'm',
#     'None': 'chocolate'
# }
# plot_df["colors"] = plot_df['type'].apply(set_value, args=(color_dictionary, ))


# ax = plot_df.plot.scatter(x='embedding1', y='embedding2', c= "colors", s=10)
# for k, v in color_dictionary.items():
#     _x = plot_df[plot_df["type"] == v].embedding1
#     _y = plot_df[plot_df["type"] == v].embedding2
#     plt.scatter(_x, _y, s=10, c=v, alpha=0.6, marker='o')

# ax.set_prop_cycle(color=[
#         '#9467bd', '#ff7f0e', '#2ca02c',
#         '#d62728', '#bcbd22', '#8c564b',
#         '#e377c2', '#7f7f7f', '#17becf'])
#
# ax.legend(loc='best')
# # subplot.set_title(title, fontsize='9')
# ax.set_xlabel('First', fontsize='9')
# ax.set_ylabel('Second', fontsize='9')
# ax.grid(ls='--')
# ax.legend(loc='best')
# ax.legend(frameon=True, fontsize=8)
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
# plt.tight_layout()
# plt.show()

# fig = px.scatter(plot_df,
#                  x="embedding1",
#                  y="embedding2",
#                  hover_data=['uniques', 'label'],
#                  size_max=0.5,
#                  template='simple_white',
#                  color='colors'
#                  # color='clusters'
#                  )
# fig.update_layout(
#     title="TSNE of " + MODEL_FILE + " on " + DATASET_FILE,
#     annotations=annotations
# )
# fig.show()
