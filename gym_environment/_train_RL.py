import gym
from stable_baselines.common.env_checker import check_env

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from gym.envs.registration import register

from gym_environment.ReasoningEnv import ReasoningEnv

# Instantiate the environment
env = ReasoningEnv( data_file_path='Opcua-all.txt', embedding_model_path='export/opcua_TransE.pkl')

# Check the environment
# check_env(env)

# Train using PPO
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])
model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="tmp/ppo_ReasoningEnv_transE/")
#model.learn(total_timesteps=1000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()