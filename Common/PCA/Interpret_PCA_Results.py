#!/usr/bin/env python
# coding: utf-8

# ### Your Turn! (Solution)
# 
# In the last video, you saw two of the main aspects of principal components:
# 
# 1. **The amount of variability captured by the component.**
# 2. **The components themselves.**
# 
# In this notebook, you will get a chance to explore these a bit more yourself.  First, let's read in the necessary libraries, as well as the data.

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from helper_functions import show_images, do_pca, scree_plot, plot_component
import test_code as t

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

#read in our dataset
train = pd.read_csv('./data/train.csv')
train.fillna(0, inplace=True)

# save the labels to a Pandas series target
y = train['label']
# Drop the label feature
X = train.drop("label",axis=1)

show_images(30)


# `1.` Perform PCA on the **X** matrix using on your own or using the **do_pca** function from the **helper_functions** module. Reduce the original more than 700 features to only 10 principal components.

# In[ ]:


pca, X_pca = do_pca(10, X)


# `2.` Now use the **scree_plot** function from the **helper_functions** module to take a closer look at the results of your analysis.

# In[ ]:


scree_plot(pca)


# `3.` Using the results of your scree plot, match each letter as the value to the correct key in the **solution_three** dictionary.  Once you are confident in your solution run the next cell to see if your solution matches ours.

# In[ ]:


a = True
b = False
c = 6.13
d = 'The total amount of variability in the data explained by the first two principal components'
e = None

solution_three = {
    '10.42' : d, 
    'The first component will ALWAYS have the most amount of variability explained.': a,
    'The total amount of variability in the data explained by the first component': c,
    'The sum of the variability explained by all the components can be greater than 100%': b
}


# In[ ]:


#Run this cell to see if your solution matches ours
t.question_3_check(solution_three)


# `4.` Use the **plot_component** function from the **helper_functions** module to look at each of the components (remember they are 0 indexed).  Use the results to assist with question 5.

# In[ ]:


plot_component(pca, 3)


# `5.` Using the results from viewing each of your principal component weights in question 4, change the following values of the **solution_five** dictionary to the **number of the index** for the principal component that best matches the description.  Once you are confident in your solution run the next cell to see if your solution matches ours.

# In[ ]:


solution_five = {
    'This component looks like it will assist in identifying zero': 0,
    'This component looks like it will assist in identifying three': 3
}


# In[ ]:


#Run this cell to see if your solution matches ours
t.question_5_check(solution_five)


# From this notebook, you have had an opportunity to look at the two major parts of PCA:
# 
# `I.` The amount of **variance explained by each component**.  This is called an **eigenvalue**.
# 
# `II.` The principal components themselves, each component is a vector of weights.  In this case, the principal components help us understand which pixels of the image are most helpful in identifying the difference between between digits. **Principal components** are also known as **eigenvectors**.

# In[ ]:




