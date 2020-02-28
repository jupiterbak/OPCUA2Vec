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
restored_model = restore_model(model_name_path='export/opcua_ComplEx.pkl')

# Get the teams entities and their corresponding embeddings
triples_df = pd.DataFrame(X, columns=['s', 'p', 'o'])
uniques = triples_df.s.unique()
uniques_embeddings = dict(zip(uniques, restored_model.get_embeddings(uniques)))
uniques_embeddings_array = np.array([i for i in uniques_embeddings.values()])

# Find clusters of embeddings using KMeans
kmeans = KMeans(n_clusters=6, n_init=100, max_iter=500)
clusters = find_clusters(uniques, restored_model, kmeans, mode='entity')

# Project embeddings into 2D space via PCA
embeddings_2d = PCA(n_components=2).fit_transform(uniques_embeddings_array)

plot_df = pd.DataFrame({"uniques": uniques,
                        "clusters": pd.Series(clusters).astype(str),
                        "embedding1": embeddings_2d[:, 0],
                        "embedding2": embeddings_2d[:, 1]})

# Top 20 nodes
top20Node = ["ns=0;i=84", "ns=0;i=85", "ns=0;i=86", "ns=0;i=87", "ns=4;s=Demo", "ns=0;i=58", "ns=0;i=31", "ns=0;i=62",
             "ns=0;i=2041", "ns=0;i=92", "ns=0;i=93", "ns=0;i=24"]

np.random.seed(0)

# Plot 2D embeddings with country labels
def plot_clusters(hue):
    fig = px.scatter(plot_df,
                     x="embedding1",
                     y="embedding2",
                     hover_data=['uniques'],
                     color="clusters"
                     )
    fig.update_layout(
        title="ComplEx",
    )
    fig.show()
    # fig = plt.figure(figsize=(12, 12))
    # plt.title("{} embeddings".format(hue).capitalize())
    # ax = sns.scatterplot(data=plot_df,
    #                      x="embedding1",
    #                      y="embedding2",
    #                      alpha=0.3,
    #                      )
    # labels = plot_df["uniques"].values.tolist()
    # tooltip = mpld3.plugins.PointLabelTooltip(ax, labels=labels)
    # mpld3.plugins.connect(fig, tooltip)
    # mpld3.show()

    # texts = []
    # for i, point in plot_df.iterrows():
    #     if point["uniques"] in top20Node or np.random.random() < 0.01:
    #         texts.append(plt.text(point['embedding1']+0.02,
    #                      point['embedding2']+0.01,
    #                      str(point["uniques"])))
plot_clusters("plot")

