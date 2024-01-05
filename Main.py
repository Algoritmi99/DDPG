import gymnasium as gym
import torch
import json

from torch.optim import Adam
import matplotlib.pyplot as plt

from GymnasiumDDPGTrainer import Actor, Critic

from GymnasiumDDPGTrainer.DDPG.DDPG import DDPG
from GymnasiumDDPGTrainer.OUNoise import OUNoise


def main():
    settingsFile = open("cc_paper_settings.json")
    settings = json.load(settingsFile)
    settingsFile.close()
    assert isinstance(settings, dict)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    environment = gym.make('HumanoidStandup-v4')

    actor_net = Actor.ActorNet(
        environment.observation_space.shape[0], environment.action_space.shape[0], device, torch.double
    )
    critic_net = Critic.CriticNet(
        environment.observation_space.shape[0], environment.action_space.shape[0], device, torch.double
    )

    actor_optimizer = Adam(actor_net.parameters(), lr=settings["HYPERPARAMS"]["ACTOR_LR"])
    critic_optimizer = Adam(critic_net.parameters(), lr=settings["HYPERPARAMS"]["CRITIC_LR"],
                            weight_decay=settings["HYPERPARAMS"]["WEIGHT_DECAY"])

    target_actor_net = Actor.ActorNet(
        environment.observation_space.shape[0], environment.action_space.shape[0], device, torch.double
    )
    target_critic_net = Critic.CriticNet(
        environment.observation_space.shape[0], environment.action_space.shape[0], device, torch.double
    )

    target_actor_net.load_state_dict(actor_net.state_dict())
    target_critic_net.load_state_dict(critic_net.state_dict())

    noise = OUNoise(environment.action_space.shape[0])

    agent = DDPG(
        environment, actor_net, critic_net, target_actor_net, target_critic_net, noise, settings["HYPERPARAMS"], device
    )
    agent.train(actor_optimizer, critic_optimizer, max_episodes=settings["HYPERPARAMS"]["MAX_EPISODES"])

    rew_list = agent.get_reward_list()
    greedy_rew_list = agent.get_greedy_reward_list()
    avg_rew_list = agent.get_avg_reward_list()
    greedy_avg_rew_list = agent.get_greedy_avg_reward_list()

    fig, ax = plt.subplots()
    ax.plot(greedy_rew_list, label="Greedy policy reward value")
    ax.plot(greedy_avg_rew_list, label="Mean greedy policy reward value")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward value")
    ax.legend()

    fig.savefig("evaluation.png")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(rew_list, label="Reward value during training")
    ax.plot(avg_rew_list, label="Mean reward value during training")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward value")
    ax.legend()

    fig.savefig("training.png")
    plt.show()


if __name__ == "__main__":
    main()

