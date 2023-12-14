import gymnasium as gym
import torch
import json

from torch.optim import Adam
import matplotlib.pyplot as plt

from GymnasiumDDPGTrainer import Actor, Critic

from GymnasiumDDPGTrainer.DDPG.DDPG import DDPG

if __name__ == '__main__':
    settingsFile = open("settings.json")
    settings = json.load(settingsFile)
    settingsFile.close()
    assert isinstance(settings, dict)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    environment = gym.make("MountainCarContinuous-v0")

    actor_net = Actor.ActorNet(environment.observation_space.shape[0], 400, device, torch.double)
    critic_net = Critic.CriticNet(
        environment.observation_space.shape[0], environment.action_space.shape[0], 400, device, torch.double
    )

    actor_optimizer = Adam(actor_net.parameters(), lr=settings["HYPERPARAMS"]["LR"])
    critic_optimizer = Adam(critic_net.parameters(), lr=settings["HYPERPARAMS"]["LR"])

    target_actor_net = Actor.ActorNet(environment.observation_space.shape[0], 400, device, torch.double)
    target_critic_net = Critic.CriticNet(
        environment.observation_space.shape[0], environment.action_space.shape[0], 400, device, torch.double
    )

    target_actor_net.load_state_dict(actor_net.state_dict())
    target_critic_net.load_state_dict(critic_net.state_dict())

    agent = DDPG(
        environment, actor_net, critic_net, target_actor_net, target_critic_net, settings["HYPERPARAMS"], device
    )
    agent.train(actor_optimizer, critic_optimizer, max_episodes=settings["HYPERPARAMS"]["MAX_EPISODES"])

    rew_list = agent.get_reward_list()
    plt.plot(rew_list)
    plt.show()

    actor_error_hist, critic_error_hist = agent.get_losses()
    fig, ax = plt.subplots()
    ax.plot(actor_error_hist, label="Actor error")
    ax.plot(critic_error_hist, label="Critic error")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Error value")
    ax.legend()

    fig.savefig("400_layer_both.png")
    plt.show()
