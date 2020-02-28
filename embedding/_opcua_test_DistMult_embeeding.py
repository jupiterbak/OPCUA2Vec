import numpy as np

from ampligraph.latent_features import ComplEx, DistMult
from ampligraph.utils import restore_model

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
                seed=555,
                verbose=True)


# Restore the model
restored_model = restore_model(model_name_path='export/opcua_DistMult.pkl')

y_pred_after = restored_model.predict(np.array([['ns=0;i=16572',	'ns=0;i=40', 'ns=0;i=68']]))
print('embedding of [\'ns=0;i=16572\',	\'ns=0;i=40\', \'ns=0;i=68\'] is {}'.format(y_pred_after))

embs = restored_model.get_embeddings(['ns=0;i=16572'], embedding_type='entity')
print(embs)