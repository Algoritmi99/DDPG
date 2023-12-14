import torch
import gymnasium as gym
import torch.nn as nn

from GymnasiumDDPGTrainer.Actor import ActorNet
from GymnasiumDDPGTrainer.Critic import CriticNet
from GymnasiumDDPGTrainer.DDPG.ReplayBuffer import ReplayBuffer
from GymnasiumDDPGTrainer.Ecosystem import Ecosystem

dtype = torch.double


def network_update(loss, optim):
    optim.zero_grad()
    loss.backward()
    optim.step()


def target_network_update(net_params, target_net_params, polyak):
    for net_param, target_net_param in zip(net_params, target_net_params):
        target_net_param.data = polyak * net_param.data + (1 - polyak) * target_net_param.data


class DDPG:
    def __init__(self, environment: gym.Env,
                 actor: ActorNet,
                 critic: CriticNet,
                 target_actor: ActorNet,
                 target_critic: CriticNet,
                 hyper_params: dict,
                 device: str = "cpu"
                 ):
        self.__env = environment
        self.__actor = actor
        self.__critic = critic
        self.__target_actor = target_actor
        self.__target_critic = target_critic
        self.__hyper_params = hyper_params
        self.__device = device

        self.__replay_buffer = ReplayBuffer(hyper_params["MAX_BUFFER_SIZE"], hyper_params["BATCH_SIZE"])
        self.__ecosystem = Ecosystem(self.__env, self.__actor, self.__device, dtype)
        self.__train_rewards_list = None
        self.__actor_error_history = []
        self.__critic_error_history = []

    def train(self, actor_optimizer, critic_optimizer, max_episodes=50):
        print("Starting Training:\nTraining on " + str(self.__device))

        self.__train_rewards_list = []
        for episode in range(max_episodes):
            state = torch.tensor(self.__ecosystem.reset_environment(), device=self.__device, dtype=dtype).unsqueeze(0)
            episode_reward = 0

            for time in range(self.__hyper_params["MAX_TIME_STEPS"]):
                # take an action according to the Actor NN and store to replay buffer
                self.__env, action, next_state, reward, done = self.__ecosystem.take_action(state)
                self.__replay_buffer.store_transition(state, action, reward, next_state, done)
                episode_reward += float(reward[0][0])
                state = next_state

                # Make a sample batch according to the replay buffer
                sample_batch = self.__replay_buffer.sample_batch()

                # train the critic network and save the history
                with torch.no_grad():
                    target = sample_batch.reward + \
                             (1 - sample_batch.done) * self.__hyper_params["GAMMA"] * \
                             self.__target_critic.forward(sample_batch.next_state,
                                                          self.__target_actor(sample_batch.next_state))

                critic_loss = nn.MSELoss()(
                    target, self.__critic.forward(sample_batch.state, sample_batch.action)
                )
                self.__critic_error_history.append(critic_loss.item())
                network_update(critic_loss, critic_optimizer)

                # train the actor network and save the history
                actor_loss = -1 * torch.mean(
                    self.__critic.forward(sample_batch.state, self.__actor(sample_batch.state))
                )
                assert isinstance(actor_loss, torch.Tensor)
                self.__actor_error_history.append(actor_loss.item())
                network_update(actor_loss, actor_optimizer)

                # update both actor and critic networks
                target_network_update(
                    self.__actor.parameters(), self.__target_actor.parameters(), self.__hyper_params["POLYAK"]
                )
                self.__ecosystem.set_actor(self.__actor)

                target_network_update(
                    self.__critic.parameters(), self.__target_critic.parameters(), self.__hyper_params["POLYAK"]
                )

                if done:
                    print("Completed episode {}/{}".format(
                        episode + 1, max_episodes))
                    break

            self.__train_rewards_list.append(episode_reward)

        self.__env.close()

    def get_reward_list(self) -> list[int]:
        return self.__train_rewards_list

    def get_losses(self) -> tuple[list, list]:
        return self.__actor_error_history, self.__critic_error_history

    def get_actor(self) -> ActorNet:
        return self.__actor

    def get_ecosystem(self) -> Ecosystem:
        return self.__ecosystem
