# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 12:14:27 2019

@author: PLDAPRO1
"""

import pandas as pd  # convention to import and use pandas like this
import matplotlib.pyplot as plt

df = pd.read_csv("model_IMG_SIZE_50__EPOCHS_60__t_1576360062.002.log")

df = df[2:]

df['Train_loss MA'] = df['Train Loss'].rolling(10).mean()
df['Valid_loss MA'] = df['Validation Loss'].rolling(10).mean()

df['Train_accuracy MA'] = df['Train Accuracy'].rolling(10).mean()
df['Valid_accuracy MA'] = df['Validation Accuracy'].rolling(10).mean()



print(df.head())


fig = plt.figure()
ax1 = plt.subplot2grid((2,1), (0,0))
ax2 = plt.subplot2grid((2,1), (1,0),sharex=ax1)

#ax1.plot(df.index,df['Train Loss']     ,"r",label='Train Loss (In-Sample)')
#ax1.plot(df.index,df['Validation Loss'],"g",label='Validation Loss'       )

ax1.plot(df.index,df['Train_loss MA'],"y",label='Train Loss - MA'         )
ax1.plot(df.index,df['Valid_loss MA'],"b",label='Validation Loss - MA'    )
ax1.legend(loc=2)
ax1.set_xscale('log')
ax1.grid()


#ax2.plot(df.index,df['Train Accuracy']     ,"r",label='Train Loss (In-Sample)')
#ax2.plot(df.index,df['Validation Accuracy'],"g",label='Validation Loss'       )

ax2.plot(df.index,df['Train_accuracy MA'],"y",label='Train Loss - MA'         )
ax2.plot(df.index,df['Valid_accuracy MA'],"b",label='Validation Loss - MA'    )
ax2.legend(loc=2)
ax2.set_xscale('log')
ax2.grid()

plt.show()


print(f"\n\n",df[158:159])
