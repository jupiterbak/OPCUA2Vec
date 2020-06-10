import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ampligraph.datasets import load_from_csv
from ampligraph.discovery import find_clusters
from ampligraph.evaluation import train_test_split_no_unseen
from ampligraph.utils import restore_model
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import mpld3
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

X = load_from_csv('data', 'Opcua-all.txt', sep='\t')

# Train test split
X_train, X_test = train_test_split_no_unseen(X, test_size=1000)

# Restore the model
restored_model = restore_model(model_name_path='export/opcua_autoTransE.pkl')

# Get the teams entities and their corresponding embeddings
triples_df = pd.DataFrame(X, columns=['s', 'p', 'o'])
uniques = triples_df.s.unique()
uniques_embeddings = dict(zip(uniques, restored_model.get_embeddings(uniques)))
uniques_embeddings_array = np.array([i for i in uniques_embeddings.values()])

# get the labels
labels = load_from_csv('data', 'dataOpcua-ANSI-BrowseNameMap.txt', sep='\t')
labels_df = {labels[i][0]: labels[i][1] for i in range(len(labels))}
unique_labels = [labels_df.get(e, None) for e in uniques]

# Find clusters of embeddings using KMeans
kmeans = KMeans(n_clusters=4, n_init=100, max_iter=500)
clusters = find_clusters(uniques, restored_model, kmeans, mode='entity')

# Project embeddings into 2D space via PCA
embeddings_2d = TSNE(n_components=2).fit_transform(uniques_embeddings_array)
# PCA(n_components=2).fit_transform(uniques_embeddings_array)

# get the annotation
annotation_nodes = ["ns=0;i=84", "ns=0;i=85", "ns=0;i=86", "ns=0;i=88", "ns=0;i=58",
                    "ns=4;i=1003", "ns=4;i=5065", "ns=4;s=Demo.Static.Arrays.Int32", "ns=4;s=Demo.Static.Arrays.Int16", "ns=4;s=Demo.BoilerDemo.Boiler1"]
annotation_indexes = [list(uniques_embeddings.keys()).index(e) for e in annotation_nodes]
annotations= [dict(text=labels_df.get(uniques[i], "None"), x=embeddings_2d[i, 0], y=embeddings_2d[i, 1]) for i in annotation_indexes]

plot_df = pd.DataFrame({"uniques": uniques,
                        "clusters": pd.Series(clusters).astype(str),
                        "embedding1": embeddings_2d[:, 0],
                        "embedding2": embeddings_2d[:, 1],
                        "label": unique_labels})

fig = px.scatter(plot_df,
                 x="embedding1",
                 y="embedding2",
                 hover_data=['uniques', 'label'],
                 size_max=0.5,
                 template='simple_white',
                 color='clusters'
                 )
fig.update_layout(
    title="TransE_TSNE",
    annotations=annotations
)
fig.show()
