import tensorflow as tf
import gym
from stable_baselines.common import make_vec_env
from stable_baselines.common.env_checker import check_env

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from gym.envs.registration import register

from gym_environment.ReasoningDeepPathEnv import ReasoningDeepPathEnv
from gym_environment.ReasoningEnv import ReasoningEnv

# Device to load the neural network on.
# Useful if you're training a model on the same
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

# Number of environment to instantiate
NUM_ENV = 1
# sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=False, log_device_placement=False))

env = None
if NUM_ENV == 1:
    # Instantiate a single environment
    _env = ReasoningDeepPathEnv(data_file_path='Opcua-all.txt', embedding_model_path='export/opcua_autoTransE.pkl',
                               max_step=4)
    env = DummyVecEnv([lambda: _env])
else:
    # multiprocess environment
    env_options = {'data_file_path': 'Opcua-all.txt', 'embedding_model_path': 'export/opcua_autoTransE.pkl',
                   'max_step': 8}
    env = make_vec_env(ReasoningDeepPathEnv, n_envs=NUM_ENV, env_kwargs=env_options, )

# Check the environment
# check_env(env)

# Train using PPO
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])
with tf.device(DEVICE):
    RL_model = PPO2(
        MlpLstmPolicy,
        env,
        policy_kwargs={'n_lstm': 128},
        cliprange=0.2,
        n_steps=64,
        seed=0, # 998, 345, 2, 786, 23, 134, 34, 799, 4587, 5
        learning_rate=20e-4,
        verbose=1,
        tensorboard_log="tmp/ppo_ReasoningDeepPathEnv_autoTransE_7/",
        nminibatches=1,
        n_cpu_tf_sess=4)
    RL_model.learn(total_timesteps=40000, tb_log_name='PPO2_MlpPolicy')
    RL_model.save("export/ppo2_MlpPolicy_autoTransE_DeepPath_7")

    # test the model
    RL_model = PPO2.load("export/ppo2_MlpPolicy_autoTransE_DeepPath_7")
    obs = env.reset()
    for i in range(128):
        env.render()
        action, _states = RL_model.predict(obs)
        obs, rewards, dones, info = env.step(action)
