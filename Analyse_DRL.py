import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##########################################
######### Values.txt reading

df_values = pd.read_csv("./Values.txt", sep="\t", header=0)
print(df_values.head())

### Reward plot

nb_env = 8 # Number of parallel environments
nb_av = 3 # Number of reloads
index_row = np.array(df_values.index.values.tolist())
nb_gen = df_values.at[ index_row[-1], 'Index'] + 1
# print(nb_gen)
list_gen = np.array(range(nb_gen))
list_env = np.array(range(nb_env))

for i in range(1, nb_av+1):

    plt.figure("Reward evolution run n_"+str(i), figsize=(12,8))
    plt.title("Reward per episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    
    for env in list_env:

        # print(np.where( (index_row%nb_env)==env )[0])
        print(np.shape(list_gen), np.shape(np.where( (index_row[(i-1)*nb_gen*nb_env:i*nb_gen*nb_env]%nb_env)==env )[0]))
        plt.plot(list_gen, df_values.loc[ np.where( (index_row[(i-1)*nb_gen*nb_env:i*nb_gen*nb_env]%nb_env)==env )[0], "Reward"], label="env_"+str(env))

    plt.legend(loc="best")
    plt.show()
