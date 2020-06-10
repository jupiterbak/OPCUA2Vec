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
import mpld3
import plotly.graph_objects as go
import plotly.express as px

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
kmeans = KMeans(n_clusters=6, n_init=100, max_iter=500)
clusters = find_clusters(uniques, restored_model, kmeans, mode='entity')

# Project embeddings into 3D space via PCA
embeddings_3d = PCA(n_components=3).fit_transform(uniques_embeddings_array)

plot_df = pd.DataFrame({"uniques": uniques,
                        "clusters": pd.Series(clusters).astype(str),
                        "embedding1": embeddings_3d[:, 0],
                        "embedding2": embeddings_3d[:, 1],
                        "embedding3": embeddings_3d[:, 2],
                        "label": unique_labels})

np.random.seed(555)

fig = px.scatter_3d(plot_df,
                    x="embedding1",
                    y="embedding2",
                    z="embedding3",
                    hover_data=['uniques', 'label'],
                    color="clusters",
                    size_max=1
                    )
fig.update_layout(
    title="TransE",
)
fig.show()
