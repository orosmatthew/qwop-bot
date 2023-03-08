# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 21:23:12 2023

@author: wilds
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('expand_frame_repr', False)
df = pd.read_json("network.json")

df.head()
df.describe()


norm = pd.json_normalize(df['network'])  #Separate network column
norm.head()

df = df.drop(['network'], axis = 1)  #Drop original network column
df.head()


new_df = pd.concat([df, norm], axis = 1) #Concat color and normalized network 

new_df.head()

new_df.plot.bar(xlabel = 'Runner #', ylabel = 'Weight', stacked = True)
