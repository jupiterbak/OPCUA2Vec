import gym
from stable_baselines.common import make_vec_env
from stable_baselines.common.env_checker import check_env

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from gym.envs.registration import register

from gym_environment.ReasoningDeepPathEnv import ReasoningDeepPathEnv
from gym_environment.ReasoningEnv import ReasoningEnv

# Instantiate a single environment
env = ReasoningDeepPathEnv( data_file_path='Opcua-all.txt', embedding_model_path='export/opcua_TransE.pkl', max_step=10)
# multiprocess environment
# env_options = {'data_file_path':'Opcua-all.txt', 'embedding_model_path':'export/opcua_TransE.pkl', 'max_step':10}
# env = make_vec_env(ReasoningDeepPathEnv, n_envs=8, env_kwargs=env_options,)

# Check the environment
# check_env(env)

# Train using PPO
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

RL_model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="tmp/ppo_ReasoningDeepPathEnv_TransE/", nminibatches=1)
RL_model.learn(total_timesteps=20000, tb_log_name='PPO2_MlpPolicy')
RL_model.save("export/ppo2_MlpLstmPolicy_TransE_DeepPath")

# test the model
RL_model = PPO2.load("export/ppo2_MlpLstmPolicy_TransE_DeepPath")
obs = env.reset()
for i in range(100):
    action, _states = RL_model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        env.reset()