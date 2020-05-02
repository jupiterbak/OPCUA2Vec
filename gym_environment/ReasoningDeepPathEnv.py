import gym
from gym import spaces
from ampligraph.datasets import load_from_csv
from ampligraph.discovery import find_clusters, query_topn
from ampligraph.evaluation import train_test_split_no_unseen
from ampligraph.utils import restore_model
import numpy as np
import pandas as pd


class ReasoningDeepPathEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, data_file_path, embedding_model_path, max_step=10):
        super(ReasoningDeepPathEnv, self).__init__()
        self.max_step = max_step
        # Load the data set which describe the graph
        self.triples = load_from_csv('data', data_file_path, sep='\t')
        # Get the unique entities and their corresponding embeddings
        self.triples_df = pd.DataFrame(self.triples, columns=['s', 'p', 'o'])
        self.unique_entities = self.triples_df.s.unique()
        self.unique_relations = self.triples_df.p.unique()

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
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(
                (2 * self.unique_entities_embeddings.shape[1]),), dtype=np.float32)
        self.last_node = ''
        self.last_predicate = ''
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
        self.step_number = 0
        self.path_history = []

    def infer_next_object(self, selected_relation):
        top_n = query_topn(self.restored_model, top_n=1, head=self.current_node, relation=selected_relation, tail=None,
                           ents_to_consider=None, rels_to_consider=None)
        return top_n[0][2], top_n[1]

    def step(self, action):
        # Execute one time step within the environment
        _score = 100000
        if action < self.unique_relations.size:
            self.last_node = self.current_node
            self.last_predicate = self.unique_relations[action]
            self.current_node, _score = self.infer_next_object(self.unique_relations[action])
        else:
            self.last_node = self.current_node
            self.last_predicate = 'NOOP'

        # Compute the next observation
        self.current_node_embeddings = np.float32(self.restored_model.get_embeddings(self.current_node))
        self.current_observation = np.concatenate((self.current_node_embeddings, self.query_object_embeddings
                                                   - self.current_node_embeddings), axis=None)

        # Save the last action into the path reasoning history
        self.path_history.append(self.last_predicate)
        # Increment the step
        self.step_number = self.step_number + 1

        # Compute done signal
        self.done = (self.current_node.lower() == self.query_object.lower()) or (self.step_number % self.max_step == 0)

        # Compute the reward
        efficiency_reward = 1.0 / (len(self.path_history))
        if self.current_node.lower() == self.query_object.lower():
            global_reward = 1.0
            # self.last_reward = 1.0
            self.last_reward = global_reward + efficiency_reward
        # elif (self.step_number % self.max_step) == 0:
        #     global_reward = -1.0
        #     self.last_reward = global_reward
        else:
            global_reward = 0.0
            self.last_reward = global_reward
            # self.last_reward = (1/(1 + abs(_score)))

        # self.last_reward = 1.0 if self.current_node == self.query_object else 0.0

        return self.current_observation, float(self.last_reward), self.done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        # Sample a triple
        _sample = self.triples_df.sample(n=1)
        self.last_node = _sample.s.values[0]
        self.last_predicate = _sample.p.values[0]
        self.current_node = _sample.s.values[0]
        self.query_subject = _sample.s.values[0]
        self.query_relation = _sample.p.values[0]
        self.query_object = _sample.o.values[0]

        self.query_subject_embeddings = np.float32(self.restored_model.get_embeddings(self.query_subject))
        self.query_relation_embeddings = np.float32(self.restored_model.get_embeddings(self.query_relation,
                                                                                       embedding_type='relation'))
        self.query_object_embeddings = np.float32(self.restored_model.get_embeddings(self.query_object))
        self.current_node_embeddings = self.query_subject_embeddings

        # self.current_observation = np.concatenate((self.current_node_embeddings, self.query_subject_embeddings,
        #                                            self.query_relation_embeddings), axis=None)
        self.current_observation = np.concatenate((self.current_node_embeddings, self.query_object_embeddings
                                                   - self.current_node_embeddings), axis=None)

        self.last_reward = 0.0
        self.done = False
        self.step_number = 0
        self.path_history = []
        # Render the TASK
        print('###############################################################################################')
        print('New Task:  [{:<12}] -> ({:<12}) -> ?'.format(self.query_subject, self.query_relation))
        print('     -->:  [{:<12}] -> ({:<12}) -> ({:<12})'.format(self.query_subject, self.query_relation,
                                                                   self.query_object))
        print('###############################################################################################')
        return self.current_observation

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print('Step {:<10}: [{:<12}] -> ({:<12}) -> [{:<12}] - Reward: {:.2f}'.format(self.step_number,
                                                                                      self.last_node,
                                                                                      self.last_predicate,
                                                                                      self.current_node,
                                                                                      self.last_reward))
