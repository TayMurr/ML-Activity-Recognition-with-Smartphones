import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Load datasets
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

train['Data'] = 'Train'
test['Data'] = 'Test'
data = pd.concat([train, test], axis=0).reset_index(drop=True)

data_labels = data.pop('Data')
subject_labels = data.pop('subject')
labels = data.pop('Activity')

# check shape
"""
print('Shape Train:\t{}'.format(train.shape))
print('Shape Test:\t{}\n'.format(test.shape))
"""

# check data types
"""
print(set(data.dtypes))
"""

#data_strip = data.pop('Data')

scaler = StandardScaler()

transformed_data = scaler.fit_transform(data)

pca = PCA(n_components=0.9, random_state=3)
data_r = pca.fit_transform(transformed_data)

data_tsne = TSNE(random_state=3).fit_transform(data_r)

print("shape: {}".format(data_tsne.shape))

label_counts = labels.value_counts()

print(label_counts)

# Plot each activity
for i, group in enumerate(label_counts.index):
    # Mask to separate sets
    mask = (labels==group).values
    plt.scatter(x=data_tsne[mask][:,0], y=data_tsne[mask][:,1], label=group)

plt.show()
