import gym
from stable_baselines.common import make_vec_env
from stable_baselines.common.env_checker import check_env

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from gym.envs.registration import register

from gym_environment.ReasoningEnv import ReasoningEnv

# Instantiate the environment
env = ReasoningEnv( data_file_path='Opcua-all.txt', embedding_model_path='export/opcua_TransE.pkl', max_step=10)
# multiprocess environment
# env = make_vec_env(env, n_envs=4)

# Check the environment
# check_env(env)

# Train using PPO
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])
model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="tmp/ppo_ReasoningEnv_transE/", nminibatches=1)
model.learn(total_timesteps=100, tb_log_name='PPO2_MlpPolicy')
model.save("export/ppo2_MlpLstmPolicy_transE")

obs = env.reset()
for i in range(100):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()