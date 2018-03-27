# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 10:09:03 2018

@author: Dhruv
"""

import numpy as np
import matplotlib.pyplot as plt



N = 5
men_means = (90, 85, 100, 33, 67)
x=(1,2,3,4,5)
men_std = ("Walking", "Standing", "Sitting", "Gather", "Split")

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
plt.ylim(0,130)
rects=ax.bar(x, men_means, align='center', alpha=1)
ax.set_ylabel('Accuracy')
ax.set_title('')
ax.set_xticks(x)
#ax.set_xticklabels(('Walking', 'Standing', 'Sitting', 'Gather', 'Split'), ha='center')
rects[0].set_label('Walking')
rects[1].set_label('Standing')
rects[2].set_label('Sitting')
rects[3].set_label('Gather')
rects[4].set_label('Split')


rects[0].set_color('limegreen')
rects[1].set_color('firebrick')
rects[2].set_color('steelblue')
rects[3].set_color('goldenrod')
rects[4].set_color('darkmagenta')
plt.legend(loc=1)
fig.savefig('D:\Hidden markov model\hmm.png')
plt.show()
       
