import json

import numpy as np
from ampligraph.datasets import load_from_csv
from ampligraph.evaluation import train_test_split_no_unseen, evaluate_performance, mr_score, mrr_score, hits_at_n_score
from ampligraph.utils import restore_model
from numpyencoder import NumpyEncoder

DATASET_LOCATION = 'data/'
DATASET_FILE = 'dataOpcua-OPCUA-all.txt'
RESULT_EXPORT_LOCATION = 'export/OPCUA/'
MODEL_FILE = 'opcua_autoComplEx.pkl'

print("########### Load DATASET: " + DATASET_FILE + " ##################")
X = load_from_csv(DATASET_LOCATION, DATASET_FILE, sep='\t')
# Train test split
X_train, X_test = train_test_split_no_unseen(X, test_size=1000, seed=0)

# Restore the model
restored_model = restore_model(model_name_path=RESULT_EXPORT_LOCATION + MODEL_FILE)
print("########### Model Hyper-Parameters ##################")
hyper_param_dict = restored_model.get_hyperparameter_dict()
print(json.dumps(hyper_param_dict, indent=4, cls=NumpyEncoder))

# Evaluate resulting Model
print("########### Model Evaluation ##################")
filter_triples = np.concatenate((X_train, X_test))
ranks = evaluate_performance(X,
                             model=restored_model,
                             filter_triples=filter_triples,
                             use_default_protocol=True,
                             verbose=False)

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