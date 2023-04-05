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
#df = pd.read_json("out/0/1.json")
#df.head()
#df.describe()

directory = "out/0"

temp = []                           
agg_df = pd.DataFrame()             

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        temp.append(f)
      
def sortLen(item):
    return len(item)

temp.sort(key=sortLen)       
      
for i in temp:
    agg_df = pd.concat([agg_df, pd.read_json(i)] ) 


agg_df = agg_df.drop(['network'], axis = 1)

agg_df.plot.line(xlabel = 'Generation #', ylabel = 'Fitness')

#agg_df.plot.bar(xlabel = 'Generation #', ylabel = 'Fitness') #TO-DO fix or remove


avg_df = agg_df.drop(['color'], axis = 1)
avg_df = pd.DataFrame( [ agg_df[0:100].mean() ,agg_df[100:200].mean(),
                        agg_df[200:300].mean(), agg_df[300:400].mean(),
                        agg_df[400:500].mean(), agg_df[500:600].mean(), 
                        agg_df[600:700].mean(), agg_df[700:800].mean(),
                        agg_df[800:900].mean(), agg_df[900:1000].mean() ])


avg_df.plot.line(xticks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], xlabel = 'Generation #', ylabel = 'Fitness')

avg_df.plot.bar(xlabel = 'Generation #', ylabel = 'Fitness')






'''Old 

#If Network Data needs to be cleaned  
#norm = pd.json_normalize(df['network'])  #Separate network column(biases)
#norm.head()

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