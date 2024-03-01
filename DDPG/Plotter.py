import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class Plotter(object):
    def __init__(self, num_of_iter: int, num_episodes: int):
        self.__trainingRewards = [[[] for _ in range(num_episodes)] for _ in range(num_of_iter)]
        self.__evaluationRewards = [[] for _ in range(num_of_iter)]

    def add_trainingReward(self, reward, iteration, episode):
        self.__trainingRewards[iteration][episode].append(reward)

    def add_evaluationReward(self, reward, iteration):
        self.__evaluationRewards[iteration].append(reward)

    def plot_rewards(self, env_name):
        temp = []
        for i in self.__trainingRewards:
            temp.append([sum(j) for j in i])

        self.__trainingRewards = temp

        for key, values in {'training_df': [self.__trainingRewards, "Training"],
                            'evaluation_df': [self.__evaluationRewards, "Evaluation"]}.items():
            df = pd.DataFrame(values[0])
            df = df.T
            df['step'] = df.index
            df = df.rename(columns={0: 'reward_value_1', 1: 'reward_value_2', 2: 'reward_value_3'})
            df = pd.melt(df, id_vars=['step'],
                         value_vars=['reward_value_1', 'reward_value_2', 'reward_value_3'],
                         var_name='reward_type',
                         value_name='reward')

            plot = sns.lineplot(x='step', y='reward', data=df, label='Mean reward value')
            fig = plot.get_figure()
            plt.xlabel('Episode')
            plt.ylabel('Reward Values')
            plt.title(values[1] + 'Rewards')
            plt.legend()
            plt.show()
            if not os.path.exists("./" + "/Plots/" + env_name):
                os.makedirs("./Plots/" + env_name)
            fig.savefig("./Plots/" + env_name + "/" + values[1] + "Rewards.png")
