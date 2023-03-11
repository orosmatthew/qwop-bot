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
new_df['bias_ih'].describe()

new_df.plot.bar(xlabel = 'Runner #', ylabel = 'Weight', stacked = True)
new_df[0:10].plot.bar(xlabel = 'Runner #', ylabel = 'Weight', stacked = True)
new_df[10:20].plot.bar(xlabel = 'Runner #', ylabel = 'Weight', stacked = True)
new_df[20:30].plot.bar(xlabel = 'Runner #', ylabel = 'Weight', stacked = True)
new_df[30:40].plot.bar(xlabel = 'Runner #', ylabel = 'Weight', stacked = True)
new_df[40:50].plot.bar(xlabel = 'Runner #', ylabel = 'Weight', stacked = True)
new_df[50:60].plot.bar(xlabel = 'Runner #', ylabel = 'Weight', stacked = True)
new_df[60:70].plot.bar(xlabel = 'Runner #', ylabel = 'Weight', stacked = True)
new_df[70:80].plot.bar(xlabel = 'Runner #', ylabel = 'Weight', stacked = True)
new_df[80:90].plot.bar(xlabel = 'Runner #', ylabel = 'Weight', stacked = True)
new_df[90:100].plot.bar(xlabel = 'Runner #', ylabel = 'Weight', stacked = True)


agg_df = pd.DataFrame([norm[0:10].mean(), new_df[10:20].mean(),
          new_df[20:30].mean(), new_df[30:40].mean(),
          new_df[40:50].mean(), new_df[50:60].mean(), 
          new_df[60:70].mean(), new_df[70:80].mean(),
          new_df[80:90].mean(), new_df[90:100].mean() ])
          
          
agg_df.plot.bar(title = 'Averaged Weights', xlabel = 'Runner #', ylabel = 'Weight', stacked = True)
