#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.metrics import cohen_kappa_score


# In[21]:


# Define the confusion matrix data as a numpy array
#array for StackedImage:

stackedImage = np.array([
    [ 20, 0, 0, 0, 1, 0, 0, 0],
    [ 1, 14, 1, 0, 1, 0, 0, 1],
    [ 1, 2, 4, 1, 5, 0, 2, 1],
    [ 1, 0, 6, 12, 2, 0, 1, 0],
    [ 3, 0, 3, 0, 12, 0, 1, 0],
    [ 0, 0, 1, 0, 6, 0, 0, 0],
    [ 2, 1, 0, 0, 5, 0, 2, 3],
    [ 0, 0, 2, 0, 1, 0, 0, 17]
])

stackedVegImage = np.array([
    [20,0,1,0,0,0,0,0],
    [0,14,1,0,0,1,0,2],
    [0,1,4,1,6,0,2,2],
    [1,2,5,14,0,0,0,0],
    [3,0,3,0,10,0,1,2],
    [0,0,1,0,6,0,0,0],
    [1,0,0,2,6,0,2,2],
    [0,0,1,0,1,0,0,18]
])

stackedNormImage = np.array([
    [20,0,1,0,0,0,0,0],
    [0,14,1,0,0,1,0,2],
    [0,1,4,1,6,0,2,2],
    [1,2,5,14,0,0,0,0],
    [3,0,3,0,10,0,1,2],
    [0,0,1,0,6,0,0,0],
    [1,0,0,2,6,0,2,2],
    [0,0,1,0,1,0,0,18]
])

pca = np.array([
    [17,1,0,1,0,0,2,0],
    [1,11,0,0,3,1,1,1],
    [0,5,4,2,0,2,1,2],
    [0,4,5,7,3,0,3,0],
    [1,0,4,4,4,1,3,2],
    [0,2,1,0,1,0,2,1],
    [1,0,0,0,6,0,2,4],
    [0,0,3,0,6,0,0,11]
])

stackedNormImage_noMarch = np.array([
    [20,0,1,0,0,0,0,0],
    [1,15,1,0,0,0,0,1],
    [1,1,4,1,5,0,2,2],
    [0,2,5,13,1,0,1,0],
    [2,0,3,0,12,0,1,1],
    [0,0,1,0,6,0,0,0],
    [2,0,0,0,9,0,0,2],
    [0,0,1,0,1,0,0,18]
])

stackedNormImage_noMarch_hpt = np.array([
 [20,0,1,0,0,0,0,0],
 [1,15,1,0,1,0,0,0],
 [1,1,4,1,5,0,2,2],
 [0,2,5,14,1,0,0,0],
 [1,0,2,1,13,0,1,1],
 [0,0,2,0,5,0,0,0],
 [2,0,0,0,8,0,1,2],
 [0,0,0,0,1,0,0,19]
])

stackedNormImage_noMarch_hpt_noAcPs_AlGl = np.array([
 [20,0,1,0,0,0],
 [1,15,1,0,1,0],
 [0,3,4,1,6,2],
 [0,3,5,14,0,0],
 [2,0,2,0,14,1],
 [0,0,1,0,1,18]
])

tackedNormImage_noMarch_hpt_noQuRo_AcPs_AlGl = np.array([
 [20,0,1,0,0],[1,15,1,0,1],[1,6,7,0,2],[0,2,5,15,0],[0,0,2,0,18]
])


# In[22]:


#Choose which Stacked image you will use
data = tackedNormImage_noMarch_hpt_noQuRo_AcPs_AlGl
title = 'tackedNormImage_noMarch_hpt_noQuRo_AcPs_AlGl'


# In[23]:


# Labels for rows and columns
#labels = ["PiAb", "PoBa", "FrEx", "AlIn", "QuRo", "AcPs", "AlGl", "SaAl"] #for all data
# labels = ["PiAb", "PoBa", "FrEx", "AlIn", "QuRo", "SaAl"]
labels = ["PiAb", "PoBa", "FrEx", "AlIn", "SaAl"]

# Create DataFrame
cf_matrix = pd.DataFrame(data, index=labels, columns=labels)

print(cf_matrix)


# In[24]:


# Overall accuracy
overall_accuracy = np.trace(cf_matrix) / np.sum(cf_matrix.values)

# Producers accuracy (Precision)
producers_accuracy = np.diag(cf_matrix) / np.sum(cf_matrix, axis=0)

# Users accuracy (Recall)
users_accuracy = np.diag(cf_matrix) / np.sum(cf_matrix, axis=1)

# # Cohen's Kappa
# flat_true = cf_matrix.sum(axis=1).values
# flat_pred = cf_matrix.sum(axis=0).values
# kappa = cohen_kappa_score(flat_true, flat_pred)


print(overall_accuracy)
print(producers_accuracy)
print(users_accuracy)
#print(kappa)


# In[25]:


# Create DataFrame for accuracies
accuracy_df = pd.DataFrame({
    'Class': labels,
    'Producer\'s Accuracy': producers_accuracy,
    'User\'s Accuracy': users_accuracy
})

print(accuracy_df)


# In[26]:


# You can change the colour to grey using this custom pallette
# # Create a custom colormap
#from matplotlib.colors import LinearSegmentedColormap
# grey = LinearSegmentedColormap.from_list(
#     name='grey_binary',
#     colors=['#363636', 'white']  # dark grey to white
# )

sns.heatmap(cf_matrix, annot=True, cmap="rocket")

# Get the current axis
ax = plt.gca()

# Define the rotation angle of axis labels
ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

# Set axis names
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels');

#Set the Title, accordingly to the stocked image used
# # Manually set the title based on the data used
# if data is stackedImage: title = 'StackedImage'
# elif data is stackedVegImage: title = 'StackedVegImage'
# elif data is stackedNormImage: title = 'stackedNormImage'
# elif data is pca: title = 'PCA 3 Bands'
# elif data is stackedNormImage_noMarch: title = 'stackedNormImage_noMarch'
# elif data is stackedNormImage_noMarch_hpt: title = 'stackedNormImage_noMarch_HyperParameterTuning'
# elif data is stackedNormImage_noMarch_hpt_noAcPs_AlGl: title = 'stackedNormImage_noMarch_HPT_no_AcPs_AlGl'
# else: title = 'Unknown Data'

ax.set_title(title, pad=20)

# Display accuracies
#plt.figtext(0.5, 0.05, f"Overall Accuracy: {overall_accuracy:.2f}", ha='center')
plt.xlabel('Predicted label\n\nOverall accuracy={:0.4f}'.format(overall_accuracy))
#plt.figtext(0.5, -0.01, f"Cohen's Kappa: {kappa:.2f}", ha='center')

# Show the plot
plt.show()

