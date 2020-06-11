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

DATASET_NAME = 'ANSI'
DATASET_LOCATION = 'data/'
DATASET_FILE = 'dataOpcua-ANSI-all.txt'
RESULT_EXPORT_LOCATION = 'export/ANSI/'

# Prepare the dataset
X = load_from_csv(DATASET_LOCATION, DATASET_FILE, sep='\t')

# Get the teams entities and their corresponding embeddings
triples_df = pd.DataFrame(X, columns=['s', 'p', 'o'])

unique_s = triples_df.s
unique_r = triples_df.p.unique()
unique_o = triples_df.o
unique_entities = unique_s.append(unique_o).unique()
unique_triples = triples_df.drop_duplicates()

print("Data set name: %s" % DATASET_NAME)
print("Unique triples: %.2f" % unique_triples.size)
print("Unique relations: %.2f" % unique_r.size)
print("Unique entities: %.2f" % unique_entities.size)

# Output the results into File
with open(RESULT_EXPORT_LOCATION + 'dataset_' + DATASET_NAME + '_statistics.txt', 'w') as f:
    print("Data set name: %s" % DATASET_NAME, file=f)
    print("Unique triples: %.2f" % unique_triples.size, file=f)
    print("Unique relations: %.2f" % unique_r.size, file=f)
    print("Unique entities: %.2f" % unique_entities.size, file=f)
