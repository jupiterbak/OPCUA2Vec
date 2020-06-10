import numpy as np
import pandas as pd
import requests

from ampligraph.datasets import load_from_csv
from ampligraph.latent_features import ComplEx, TransE, DistMult
from ampligraph.evaluation import evaluate_performance
from ampligraph.evaluation import mr_score, mrr_score, hits_at_n_score
from ampligraph.evaluation import train_test_split_no_unseen
from ampligraph.utils import save_model

# Prepare the dataset
X = load_from_csv('data', 'dataOpcua-DATASETONE-all.txt', sep='\t')
# To split the graph in train, validation, and test the method must be called twice:
X_train_valid, X_test = train_test_split_no_unseen(X, test_size=1000, allow_duplication=True)
X_train, X_valid = train_test_split_no_unseen(X_train_valid, test_size=1000, allow_duplication=True)
filter_triples = np.concatenate((X_train, X_test))

# ComplEx model
model = DistMult(batches_count=10,
                 epochs=2000,
                 k=200,
                 seed=0,
                 eta=5,
                 embedding_model_params={
                        # generate corruption using all entities during training
                        "negative_corruption_entities": "all"
                 },
                 optimizer='adam',
                 optimizer_params={
                     'lr': 1e-2,
                     "momentum": 0.8,
                 },
                 loss="self_adversarial",
                 loss_params={
                        # margin corresponding to both pairwise and adverserial loss
                        "margin": 20,
                        # alpha corresponding to adverserial loss
                        "alpha": 0.5
                 },
                 regularizer='LP',
                 regularizer_params={
                     'p': 2,
                     'lambda': 1e-5
                 },
                 verbose=True)

model.fit(X_train,
          # Early stopping
          early_stopping=True,
          # Early stopping parameters
          early_stopping_params={
              'x_valid': X_valid,
              'criteria': 'mrr',
              'burn_in': 300,
              'check_interval': 100
          })

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

save_model(model, model_name_path ='export/DATASETONE/opcua_DistMult.pkl')