#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wine quality prediction using Decision Tree and Random Forest

@author: ChiNguyen
"""

#general stuff
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# sklearn packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# plotting
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('/Users/kienguyen/Documents/DATA SCIENCE/MSDS/07. MSDS680_Machine Learning/04. Week 4/assign_wk4/winequality-data.csv',index_col='id')

# Do soe EDA and clean data
df.info()
df.describe()

#Check correlation between features
corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(16,14))
sns.heatmap(corr_matrix, annot=True)
ax.set_title('Correlation heatmap of data', fontsize = 18)
plt.show()

#Check outlier of numerical features by boxplot
fig, ax= plt.subplots(figsize=(16,18))
ax.boxplot(df[['fixed.acidity','volatile.acidity','citric.acid','residual.sugar','chlorides','free.sulfur.dioxide','total.sulfur.dioxide','density','pH','sulphates','alcohol']])
ax.set_xticklabels(['fixed.acidity','volatile.acidity','citric.acid','residual.sugar','chlorides','free.sulfur.dioxide','total.sulfur.dioxide','density','pH','sulphates','alcohol'])
plt.show()

#Remove outlier:
outlier_cols = ['fixed.acidity','volatile.acidity','citric.acid','residual.sugar','chlorides','free.sulfur.dioxide','total.sulfur.dioxide','density','pH','sulphates','alcohol']
for i in outlier_cols:
    q3 = df[i].quantile(0.75)  
    q1 = df[i].quantile(0.25)
    IQR = q3 - q1
    lower_limit = q1 - (IQR * 1.5)
    upper_limit = q3 + (IQR * 1.5)
    df=df[~((df[i]<(q1-1.5*IQR)) | (df[i]>(q3+1.5*IQR)))]
    
#Group the features by value range to have more beautiful look
fig, ax= plt.subplots(figsize=(16,12))
ax.boxplot(df[['fixed.acidity','residual.sugar','alcohol']])
ax.set_xticklabels(['fixed.acidity','residual.sugar','alcohol'], fontsize=15)
ax.set_title('fixed.acidity,residual.sugar,alcohol',fontsize=17)
plt.show()

fig, ax= plt.subplots(figsize=(16,12))
ax.boxplot(df[['free.sulfur.dioxide','total.sulfur.dioxide']])
ax.set_xticklabels(['free.sulfur.dioxide','total.sulfur.dioxide'], fontsize=15)
ax.set_title('free.sulfur.dioxide,total.sulfur.dioxide',fontsize=17)
plt.show()

fig, ax= plt.subplots(figsize=(16,10))
ax.boxplot(df[['volatile.acidity','citric.acid','chlorides','density','pH','sulphates']])
ax.set_xticklabels(['volatile.acidity','citric.acid','chlorides','density','pH','sulphates'],fontsize=15)
ax.set_title('volatile.acidity, citric.acid, chlorides, density, pH, sulphates',fontsize=17)
plt.show()

# DECISION TREE
#gather up names of all the columns
cols = df.columns

#set the prediction column and the feature columns for KNN
prediction_col = 'quality'
feature_cols = [c for c in cols if c != prediction_col]

x = df[feature_cols].values
y = df[prediction_col].values

#split the dataset into the train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=41)

# establish and fit the model using DecisonTree()
from sklearn import tree

tree_model = tree.DecisionTreeClassifier()
tree_model.fit(x_train, y_train)

#gathering the predictions
preds = tree_model.predict(x_test)

#display the actuals and predictions for the test set
print('Actuals for test data set')
print(y_test)
print('Predictions for test data set')
print(preds)

# plot the training and test target variable:
test_by_quality=pd.value_counts(y_test, sort= True)
test_count = test_by_quality.tolist()
test= pd.Series(list(test_by_quality.index))
#plot the bar
plt.figsize = (16,15)
test_by_quality.plot(kind= 'bar',color= 'green')

plt.title('Bar chart of training data',fontsize=17)
plt.xlabel('Quality',fontsize = 15)
plt.ylabel('Count',fontsize = 15)
plt.ylim(0,400)

# function to add value labels
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i] + 10 , y[i], ha = 'center', fontsize= 'medium', alpha = None)
addlabels(test,test_count)
plt.show()

preds_by_quality=pd.value_counts(preds, sort= True)
preds_count = preds_by_quality.tolist()
preds=pd.Series(list(preds_by_quality.index))
#plot the bar
plt.figsize = (16,15)
preds_by_quality.plot(kind= 'bar',color= 'blue')

plt.title('Bar chart of test data',fontsize=17)
plt.xlabel('Quality',fontsize = 15)
plt.ylabel('Count',fontsize = 15)
plt.ylim(0,400)

addlabels(preds,preds_count)
plt.show()

# At this step, comparing 2 bar charts, we can hope that the model performance will be good.
# make a confusion matrix to display the results
import seaborn as sns
from sklearn.metrics import confusion_matrix
preds = tree_model.predict(x_test)
cm = confusion_matrix(y_test, preds)
target_labels = np.unique(y_test)

sns.heatmap(cm, square=True, annot=True, fmt='d', cbar=True, cmap="YlGnBu",
            xticklabels=target_labels, yticklabels=target_labels)

plt.xlabel('predicted label')
plt.ylabel('actual label')

#using the sklearn.metrics package to determine the accuracy of the model
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,preds))

# display the importance features
importance_list = list(zip(feature_cols,tree_model.feature_importances_))
sorted_import_feature = sorted(importance_list, key = lambda x: x[1],reverse=True)
sorted_import_feature

max_feature_len = len(max(feature_cols, key=len))
for feature, rank in sorted_import_feature:
    dots = max_feature_len - len(feature)
    print(f'{feature}: {"."*dots} {rank*100:.2f}%')
    
# Visualize data 
import pydotplus
import collections
import graphviz
dot_data = tree.export_graphviz(tree_model,
                                feature_names=feature_cols,
                                out_file=None,
                                filled=True,
                                rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)

colors = ('turquoise', 'orange')
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()    
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

graph.write_png('/Users/kienguyen/Documents/DATA SCIENCE/MSDS/07. MSDS680_Machine Learning/04. Week 4/assign_wk4/decisiontree.png')

#display the png here
from IPython.display import Image

Image(filename='/Users/kienguyen/Documents/DATA SCIENCE/MSDS/07. MSDS680_Machine Learning/04. Week 4/assign_wk4/decisiontree.png') 

# RANDOM FOREST
df1= df.assign(value = np.random.randint(100, size=3203))
df1.info()


#set the prediction column and the feature columns for Random Forest
prediction_col = 'quality'
feature_cols = [c for c in cols if c != prediction_col]

x = df1[feature_cols].values
y = df1[prediction_col].values

#split the dataset into the train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=41)

from sklearn.ensemble import RandomForestClassifier

forest_model = RandomForestClassifier(n_jobs = -1, random_state=41)
forest_model.fit(x_train,y_train)

#gathering the predictions
forest_preds = forest_model.predict(x_test)

#display the actuals and predictions for the test set
print('Actuals for test data set')
print(y_test)
print('Predictions for test data set')
print(forest_preds)

# make a confusion matrix to display the results
cm_forest = confusion_matrix(y_test, forest_preds)
target_labels = np.unique(y_test)

sns.heatmap(cm_forest, square=True, annot=True, fmt='d', cbar=True, cmap="YlGnBu",
            xticklabels=target_labels, yticklabels=target_labels)

plt.xlabel('predicted label')
plt.ylabel('actual label');

print(accuracy_score(y_test,forest_preds))

# display the importance features with our tree

importance_list = list(zip(feature_cols,forest_model.feature_importances_))
sorted_import_feature = sorted(importance_list, key = lambda x: x[1],reverse=True)

max_feature_len = len(max(feature_cols, key=len))
for feature, rank in sorted_import_feature:
    dots = max_feature_len - len(feature)
    print(f'{feature}: {"."*dots} {rank*100:.2f}%')