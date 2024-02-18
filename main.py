import torch
import gymnasium as gym

from DDPG.Agent.Actor import Actor
from DDPG.Agent.Agent import Agent
from DDPG.Agent.Critic import Critic
from DDPG.Agent.Noise.RandomNoise import RandomNoise
from DDPG.DDPG_Evaluator import Evaluator
from DDPG.DDPG_Trainer import Trainer
from DDPG.Plotter import Plotter
from DDPG.ReplayBuffer import ReplayBuffer


def main():
    # Settings
    learning_rate = 0.001
    discount_factor = 0.99
    tau = 0.05
    replay_buffer_capacity = 10000
    batch_size = 32
    num_episodes = 200
    max_steps = 3000
    num_eval_episodes = 10

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    print("Using device: ", device)

    # Training
    env = gym.make("Pendulum-v1", render_mode=None)

    actor = Actor(env.observation_space.shape[0], env.action_space.shape[0])
    critic = Critic(env.observation_space.shape[0], env.action_space.shape[0])

    actor_optimizer = torch.optim.AdamW(params=actor.parameters(), lr=learning_rate)
    critic_optimizer = torch.optim.AdamW(params=critic.parameters(), lr=learning_rate)

    agent = Agent(
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        RandomNoise(torch.from_numpy(env.action_space.sample()), 0.1),
        discount_factor,
        tau
    )

    replay_buffer = ReplayBuffer(replay_buffer_capacity, batch_size)

    plotter = Plotter(num_episodes)

    trainer = Trainer(env, agent, replay_buffer, plotter=plotter)

    trainer.train(num_episodes, max_steps)

    # Evaluation
    eval_env = gym.make("Pendulum-v1", render_mode='human')
    evaluator = Evaluator(eval_env, trainer.get_agent())
    evaluation = evaluator.evaluate(num_eval_episodes)
    print(f"Average return of {evaluation}.")

    plotter.plot_rewards()


if __name__ == "__main__":
    main()
