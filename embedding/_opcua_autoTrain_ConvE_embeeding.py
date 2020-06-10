import json

import numpy as np
from ampligraph.datasets import load_from_csv
from ampligraph.evaluation import evaluate_performance, select_best_model_ranking
from ampligraph.evaluation import mr_score, mrr_score, hits_at_n_score
from ampligraph.evaluation import train_test_split_no_unseen
from ampligraph.latent_features.models import ConvE
from ampligraph.utils import save_model
from numpyencoder import NumpyEncoder

DATASET_LOCATION = 'data/'
DATASET_FILE = 'dataOpcua-DATASETONE-all.txt'
RESULT_EXPORT_LOCATION = 'export/DATASETONE/'

# Prepare the dataset
X = load_from_csv(DATASET_LOCATION, DATASET_FILE, sep='\t')
# To split the graph in train, validation, and test the method must be called twice:
X_train_valid, X_test = train_test_split_no_unseen(X, test_size=1000, seed=0, allow_duplication=True)
X_train, X_valid = train_test_split_no_unseen(X_train_valid, test_size=1000, seed=0, allow_duplication=True)

# model selection
model_class = ConvE

# Use the template given below for doing grid search.
param_grid = {
    "k": [50, 200],
    "eta": [5],
    "epochs": [2000],
    "batches_count": [10],
    "seed": 0,
    "embedding_model_params": {
        # generate corruption using all entities during training
        "negative_corruption_entities": "all",
        'num_filters': [32],
        'filter_sizes': [1],
        'dropout': [0.1]
    },
    "optimizer": ["adam", "adagrad"],
    "optimizer_params": {
        "lr": [0.01],
        "momentum": [0.8],
    },
    "loss": ["bce"],
    # We take care of mapping the params to corresponding classes
    "loss_params": {
        # Learning rate for loss computation
        "lr": [0.01],
        # applies label smoothing to one-hot outputs.
        "label_smoothing": [0.1],
        # applies label smoothing to one-hot outputs.
        "label_weighting": [True]
    },
    "regularizer": ["LP"],
    "regularizer_params": {
        "p": [2],
        "lambda": [1e-5]
    },
    "verbose": True
}

# Train the model on all possibile combinations of hyperparameters.
# Models are validated on the validation set.
# It returnes a model re-trained on training and validation sets.
best_model, best_params, best_mrr_train, ranks_test, mrr_test, experimental_history = \
    select_best_model_ranking(model_class,
                              # Class handle of the model to be used
                              # Dataset
                              X_train,
                              X_valid,
                              X_test,
                              # Parameter grid
                              param_grid,
                              # Maximum Combination
                              # max_combinations=150,
                              # Early stopping
                              early_stopping=True,
                              # Early stopping parameters
                              early_stopping_params={
                                  'x_valid': X_valid,
                                  'criteria': 'mrr',
                                  'burn_in': 300,
                                  'check_interval': 100,
                                  'corrupt_side': 'o'
                              },
                              # Use filtered set for eval
                              use_filter=True,
                              # corrupt subject and objects separately during eval
                              use_default_protocol=True,
                              # Log all the model hyperparams and evaluation stats
                              verbose=True)

print(type(best_model).__name__, best_params, best_mrr_train, mrr_test)
save_model(best_model, model_name_path=RESULT_EXPORT_LOCATION + 'opcua_auto' + best_model.name + '.pkl')
# Print out the hyper-parameters
print("########### Model Hyper-Parameters ##################")
print("##" + best_model.name)
print("#####################################################")
hyper_param_dict = best_model.get_hyperparameter_dict()
print(json.dumps(hyper_param_dict, indent=4, cls=NumpyEncoder))
with open(RESULT_EXPORT_LOCATION + 'opcua_auto' + best_model.name + '.json', 'w') as outfile:
    json.dump(hyper_param_dict, outfile, indent=4, cls=NumpyEncoder)

# Evaluate resulting Model
filter_triples = np.concatenate((X_train, X_test))
ranks = evaluate_performance(X,
                             model=best_model,
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

# Output the results into File
with open(RESULT_EXPORT_LOCATION + 'opcua_auto' + best_model.name + '_performance.txt', 'w') as f:
    print("MRR: %.2f" % (mrr), file=f)
    print("MR: %.2f" % (mr), file=f)
    print("Hits@10: %.2f" % (hits_10), file=f)
    print("Hits@3: %.2f" % (hits_3), file=f)
    print("Hits@1: %.2f" % (hits_1), file=f)

print("#####################################################")