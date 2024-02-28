import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class Plotter(object):
    def __init__(self, num_episodes: int):
        self.__trainingRewardsPerStep = [[], [], []]
        self.__evaluationRewardsPerStep = [[], [], []]

    def add_trainingRewardPerStep(self, reward, iteration):
        self.__trainingRewardsPerStep[iteration].append(reward)

    def add_evaluationRewardPerStep(self, reward, iteration):
        self.__evaluationRewardsPerStep[iteration].append(reward)

    def plot_rewards(self, env_name):
        for df, values in {'training_df': [self.__trainingRewardsPerStep, "Training"],
                           'evaluation_df': [self.__trainingRewardsPerStep, "Evaluation"]}.items():
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
            plt.xlabel('Step Number')
            plt.ylabel('Reward Values')
            plt.title(values[1] + 'Rewards')
            plt.legend()
            plt.show()
            if not os.path.exists("./" + "/Plots/" + env_name):
                os.makedirs("./Plots/" + env_name)
            fig.savefig("./Plots/" + env_name + "/" + values[1] + "Rewards.png")