import numpy as np

from ampligraph.latent_features import ComplEx, TransE
from ampligraph.utils import restore_model

# Restore the model
restored_model = restore_model(model_name_path='export/opcua_autoTransE.pkl')

y_pred_after = restored_model.predict(np.array([['ns=0;i=16572',	'ns=0;i=40', 'ns=0;i=68']]))
print('embedding of [\'ns=0;i=16572\',	\'ns=0;i=40\', \'ns=0;i=68\'] is {}'.format(y_pred_after))

embs = restored_model.get_embeddings(['ns=0;i=16572'], embedding_type='entity')
print(embs)