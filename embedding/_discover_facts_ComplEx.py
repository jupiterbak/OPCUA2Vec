import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ampligraph.datasets import load_from_csv
from ampligraph.discovery import discover_facts, query_topn
from ampligraph.evaluation import train_test_split_no_unseen
from ampligraph.utils import restore_model
from sklearn.decomposition import PCA
import mpld3
import plotly.graph_objects as go
import plotly.express as px

X = load_from_csv('data', 'Opcua-all.txt', sep='\t')

# Train test split
X_train, X_test = train_test_split_no_unseen(X, test_size=1000)

# Restore the model
restored_model = restore_model(model_name_path='export/opcua_ComplEx.pkl')

# find top N cadidates for root element
top_n = query_topn(restored_model, top_n=30, head='ns=0;i=2299', relation='ns=0;i=45', tail=None,
                   ents_to_consider=None, rels_to_consider=None)

print('TOP N from root: {}'.format(top_n))

# Get the teams entities and their corresponding embeddings
facts = discover_facts(X, restored_model, top_n=3, max_candidates=200, strategy='entity_frequency',
               target_rel='ns=0;i=40', seed=42)

print(facts)

