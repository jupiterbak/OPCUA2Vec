import gym
from gym import spaces
from ampligraph.datasets import load_from_csv
from ampligraph.discovery import find_clusters, query_topn
from ampligraph.evaluation import train_test_split_no_unseen
from ampligraph.utils import restore_model
import numpy as np
import pandas as pd


class ReasoningEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, data_path, embedding_model_path, max_step=10):
        super(ReasoningEnv, self).__init__()
        self.max_step = max_step
        # Load the data set which describe the graph
        self.triples = load_from_csv('data', data_path, sep='\t')
        # Get the unique entities and their corresponding embeddings
        triples_df = pd.DataFrame(self.triples, columns=['s', 'p', 'o'])
        self.unique_entities = triples_df.s.unique()
        self.unique_relations = triples_df.r.unique()

        # Restore the model
        self.restored_model = restore_model(model_name_path=embedding_model_path)
        params = self.restored_model.get_hyperparameter_dict()

        self.embedding_space_dim = params['k']

        # Compute Embeddings for entities
        uniques_entities_embeddings = dict(zip(self.unique_entities,
                                               self.restored_model.get_embeddings(self.unique_entities)))
        self.unique_entities_embeddings = np.array([i for i in uniques_entities_embeddings.values()])

        # Compute Embeddings for relations
        uniques_relation_embeddings = dict(zip(self.unique_entities,
                                               self.restored_model.get_embeddings(self.unique_relations,
                                                                                  embedding_type='relation')))
        self.unique_relation_embeddings = np.array([i for i in uniques_relation_embeddings.values()])

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Discrete(self.unique_relations.size + 1)

        # Example for using image as input:
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3 * self.embedding_space_dim, 1),
                                            dtype=np.float32)
        self.current_node = ''
        self.query_subject = ''
        self.query_relation = ''
        self.query_object = ''
        self.current_node_embeddings = np.zeros((self.unique_entities.size, 1), dtype=np.float32)
        self.query_subject_embeddings = np.zeros((self.unique_entities.size, 1), dtype=np.float32)
        self.query_relation_embeddings = np.zeros((self.unique_entities.size, 1), dtype=np.float32)
        self.query_object_embeddings = np.zeros((self.unique_entities.size, 1), dtype=np.float32)
        self.current_observation = np.zeros((3 * self.unique_entities.size, 1), dtype=np.float32)
        self.last_reward = 0.0
        self.done = False
        self.step = 0

    def infer_next_object(self, selected_relation):
        top_n = query_topn(self.restored_model, top_n=30, head=self.current_node, relation=selected_relation, tail=None,
                           ents_to_consider=None, rels_to_consider=None)
        return top_n[0][2], top_n[1][0]

    def step(self, action):
        # Execute one time step within the environment
        _score = 0
        if action < self.unique_relations.size:
            self.current_node, _score = self.infer_next_object(self.unique_relations[action])

        # Increment the step
        self.step = self.step + 1

        # Compute the next observation
        self.current_node_embeddings = np.float32(self.restored_model.get_embeddings(self.current_node))
        self.current_observation = np.concatenate((self.current_node_embeddings, self.query_subject_embeddings,
                                                   self.query_relation_embeddings), axis=None)
        # Compute the reward
        self.last_reward = 1 if self.current_node == self.query_object else _score

        # Compute Done
        self.done = self.current_node == (self.current_node == self.query_object) or (self.step % self.max_step == 0)

    def reset(self):
        # Reset the state of the environment to an initial state
        # Sample a triple
        _sample = self.triples.sample(n_samples=1)
        self.current_node = _sample.p[0]
        self.query_subject = _sample.p[0]
        self.query_relation = _sample.r[0]
        self.query_object = _sample.o[0]

        self.query_subject_embeddings = np.float32(self.restored_model.get_embeddings(self.query_subject))
        self.query_relation_embeddings = np.float32(self.restored_model.get_embeddings(self.query_relation))
        self.query_object_embeddings = np.float32(self.restored_model.get_embeddings(self.query_object))
        self.current_node_embeddings = np.float32(self.restored_model.get_embeddings(self.current_node))
        self.current_observation = np.concatenate((self.current_node_embeddings, self.query_subject_embeddings,
                                                   self.query_relation_embeddings), axis=None)
        self.last_reward = 0.0
        self.done = False
        self.step = 0
        return self.current_observation

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass