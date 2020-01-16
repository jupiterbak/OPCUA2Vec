import numpy as np
import pandas as pd
import requests

from ampligraph.datasets import load_from_csv
from ampligraph.latent_features import ComplEx, TransE, DistMult
from ampligraph.evaluation import evaluate_performance
from ampligraph.evaluation import mr_score, mrr_score, hits_at_n_score
from ampligraph.evaluation import train_test_split_no_unseen
from ampligraph.utils import save_model

X = load_from_csv('.', 'Opcua-all.txt', sep='\t')

# Train test split
X_train, X_test = train_test_split_no_unseen(X, test_size=1000)

# ComplEx model
model = DistMult(batches_count=50,
                epochs=300,
                k=100,
                eta=20,
                optimizer='adam',
                optimizer_params={'lr':1e-4},
                loss='multiclass_nll',
                regularizer='LP',
                regularizer_params={'p':3, 'lambda':1e-5},
                seed=0,
                verbose=True)

model.fit(X_train)

filter_triples = np.concatenate((X_train, X_test))
ranks = evaluate_performance(X_test,
                             model=model,
                             filter_triples=filter_triples,
                             use_default_protocol=True,
                             verbose=True)

mr = mr_score(ranks)
mrr = mrr_score(ranks)

print("MRR: %.2f" % (mrr))
print("MR: %.2f" % (mr))

hits_10 = hits_at_n_score(ranks, n=10)
print("Hits@10: %.2f" % (hits_10))
hits_3 = hits_at_n_score(ranks, n=3)
print("Hits@3: %.2f" % (hits_3))
hits_1 = hits_at_n_score(ranks, n=1)
print("Hits@1: %.2f" % (hits_1))

save_model(model, model_name_path = 'export/opcua_DistMult.pkl')

y_pred_after = model.predict(np.array([['ns=0;i=16572',	'ns=0;i=40', 'ns=0;i=68']]))
print(y_pred_after)

embs = model.get_embeddings(['ns=0;i=16572'], embedding_type='entity')