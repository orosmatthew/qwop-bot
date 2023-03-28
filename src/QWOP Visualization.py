# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 21:23:12 2023

@author: wilds
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import glob
pd.set_option('expand_frame_repr', False)
df = pd.read_json("out/0/1.json")

directory = "out/0"

'''
WIP - Loop through directory and append individual files into single dataframe

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        print(f)


for filename in os.scandir(directory):
    if filename.is_file():
        print(filename.path)
        
        
#filelist = glob.iglob(f'{directory}/*')
 filelist = glob.glob(os.path.join(directory, 'FV/*.json'))
for filename in :
    print(filename)
        
        
'''

df.head()
df.describe()


df1 = pd.read_json("out/0/1")
df2 = pd.read_json("out/0/2")
df3 = pd.read_json("out/0/3")
df4 = pd.read_json("out/0/4")
df5 = pd.read_json("out/0/5")
df6 = pd.read_json("out/0/6")
df7 = pd.read_json("out/0/7")
df8 = pd.read_json("out/0/8")
df9 = pd.read_json("out/0/9")
df10 = pd.read_json("out/0/10")

agg_df = df1.append([df2, df3, df4, df5, df6,
                     df7, df8, df9, df10])

agg_df = agg_df.drop(['network'], axis = 1)

agg_df.plot.line(xlabel = 'Generation #', ylabel = 'Fitness')

agg_df.plot.bar(xlabel = 'Generation #', ylabel = 'Fitness')



avg_df = pd.DataFrame( [ agg_df[0:100].mean() ,agg_df[100:200].mean(),
                        agg_df[200:300].mean(), agg_df[300:400].mean(),
                        agg_df[400:500].mean(), agg_df[500:600].mean(), 
                        agg_df[600:700].mean(), agg_df[700:800].mean(),
                        agg_df[800:900].mean(), agg_df[900:1000].mean() ])


avg_df.plot.line(xticks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], xlabel = 'Generation #', ylabel = 'Fitness')

avg_df.plot.bar(xlabel = 'Generation #', ylabel = 'Fitness')

#Rearrange code 
norm = pd.json_normalize(df['network'])  #Separate network column(biases)
norm.head()

df = df.drop(['network'], axis = 1)  #Drop original network column
df.head()




df['fitness'].describe()




'''Old 
df.plot.bar(ylabel = 'Fitness') #TO-DO remove x-ticks
df[1:20].plot.bar()
df[21:40].plot.bar()
df[41:60].plot.bar()
df[61:80].plot.bar()
df[81:100].plot.bar()
     
df.plot.line(ylabel = 'Fitness') 




agg_df = pd.DataFrame( [ df[0:10].mean() ,df[10:20].mean(),df[20:30].mean(), df[30:40].mean(),
          df[40:50].mean(), df[50:60].mean(), 
          df[60:70].mean(), df[70:80].mean(),
          df[80:90].mean(), df[90:100].mean() ])

agg_df.plot.bar(xlabel = 'Generation #', ylabel = 'Fitness')
agg_df.plot.line(xticks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                 xlabel = 'Generation #', ylabel = 'Fitness')
'''