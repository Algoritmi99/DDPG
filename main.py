import json
import sys

import torch
import gymnasium as gym

from DDPG.Agent.Actor import Actor
from DDPG.Agent.Agent import Agent, save_agent, load_agent
from DDPG.Agent.Critic import Critic
from DDPG.Agent.Noise.RandomNoise import RandomNoise
from DDPG.DDPG_Evaluator import Evaluator
from DDPG.DDPG_Trainer import Trainer
from DDPG.Plotter import Plotter
from DDPG.ReplayBuffer import ReplayBuffer
from DDPG.StaticAlgorithms import supported, isMujoco


def train_and_save(settings: dict):
    # Settings
    learning_rate = settings['learning_rate']
    discount_factor = settings['discount_factor']
    tau = settings['tau']
    replay_buffer_capacity = int(settings["replayBuffer_capacity"])
    batch_size = int(settings["batch_size"])
    num_episodes = int(settings['num_trainingEpisodes'])
    max_steps = int(settings["max_training_steps"])
    environment_name = settings['environment_name']

    if not supported(environment_name):
        raise Exception("The specified environment is not supported. Please change it in the settings.json file.")

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device: ", device)

    # Training
    train_env = gym.make(environment_name, render_mode=None)
    eval_env = gym.make(environment_name, render_mode=None)

    actor = Actor(train_env.observation_space.shape[0], train_env.action_space.shape[0], device=device)
    critic = Critic(train_env.observation_space.shape[0], train_env.action_space.shape[0], device=device)

    actor_optimizer = (torch.optim.AdamW(params=actor.parameters(), lr=learning_rate))
    critic_optimizer = torch.optim.AdamW(params=critic.parameters(), lr=learning_rate)

    agent = Agent(
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        RandomNoise(torch.from_numpy(train_env.action_space.sample()), 0.1),
        discount_factor,
        tau,
        device=device
    )

    replay_buffer = ReplayBuffer(replay_buffer_capacity, batch_size)

    plotter = Plotter(num_episodes)

    evaluator = Evaluator(eval_env, agent, plotter=plotter, device=device, mujoco_mode=isMujoco(environment_name))

    trainer = Trainer(
        train_env,
        agent,
        replay_buffer,
        plotter=plotter,
        evaluator=evaluator,
        device=device,
        mujoco_mode=isMujoco(environment_name)
    )

    trainer.train(num_episodes, max_steps)

    save_agent(agent, "./agent/" + environment_name, "trainedAgent.agent")

    plotter.plot_rewards(environment_name)


def evaluate_visually(settings: dict, path_to_saved_model: str, number_of_episodes: int):
    env = gym.make(settings["environment_name"], render_mode="human")
    agent = load_agent(path_to_saved_model)
    agent.to('cpu')
    evaluator = Evaluator(env, agent, mujoco_mode=isMujoco(settings["environment_name"]))
    eval_result = evaluator.evaluate(number_of_episodes)
    print("Average return:", eval_result)


if __name__ == "__main__":
    args = sys.argv
    if len(args) < 2 or args[1] in ["-h", "--help"]:
        print("Usage:"
              "python main.py <Work>")
        print("Accepted arguments for <Work>: \n"
              "{\n"
              "\t --train_and_save, \n"
              "\t --visual_evaluation <path to saved model> <number of episodes>")

    elif args[1] == "--train_and_save":
        with open("./settings.json") as f:
            train_and_save(json.load(f))

    elif args[1] == "--visual_evaluation":
        with open("./settings.json") as f:
            evaluate_visually(json.load(f), args[2], int(args[3]))
