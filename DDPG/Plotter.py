import os

import matplotlib.pyplot as plt


class Plotter(object):
    def __init__(self, num_episodes: int):
        self.__trainingRewards = [[] for _ in range(num_episodes)]
        self.__evaluationRewards = []

    def add_trainingReward(self, reward, episode):
        self.__trainingRewards[episode].append(reward)

    def add_evaluationReward(self, reward):
        self.__evaluationRewards.append(reward)

    def plot_rewards(self, env_name):
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        data = [sum(i) for i in self.__trainingRewards]
        plt.plot(data)
        if not os.path.exists("./" + env_name + "/Plots"):
            os.makedirs("./Plots/" + env_name)
        plt.savefig("./Plots/" + env_name + "/Training Rewards.png")

        plt.cla()
        plt.plot(self.__evaluationRewards)
        plt.savefig("./Plots/" + env_name + "/Evaluation Rewards.png")
