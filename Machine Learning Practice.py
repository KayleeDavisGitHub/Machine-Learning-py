
# Machine Learning Practice with Python: Kyle Davis
# The Ohio State University
# Contact: MailKyleDavis@gmail.com 
# Using Anaconda: Spyder Python 3.6. date 10/8/2017

# For more info and on inspiration and training consider:
# Tutorial on Machine Learning in Python:
# https://pythonprogramming.net/machine-learning-tutorial-python-introduction/

# Data provided by the University of California Irvine. 

"""
 Example python code for generating data and setting up Python Operations
 Machine Learning Exercise:
     
"""

import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

# read text data as csv:
df = pd.read_csv('breast-cancer-wisconsin.data.txt')
# Replace ? responses to be treated as an outlier to be dropped.
# Rethink this on how our algorithims work with this. 
df.replace('?',-99999, inplace=True)
# Drop id variable. 
df.drop(['id'], 1, inplace=True)

# Features and Labels:
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

# Cross Validations:
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# Define k nearest neighbors using the neighbors packages from sklearn
clf = neighbors.KNeighborsClassifier()

# Now Fit our Training Data using .fit from clf from sklearn (cross_validation) 
clf.fit(X_train, y_train)

# And let's see our overall data's accuracy at predicting these.
accuracy = clf.score(X_test, y_test)
print('KNN Package Accuracy:', accuracy)

# Really high accuracy! But at what cost when we're predicting cancer? 
# If you were to include the ID status it would cloud a lot of this.


# Let's make up example measures: from numpy (np) 
example_measures = np.array([4,2,1,1,1,2,3,2,1])
# Reshape our data to not get a deprication warning:
example_measures = example_measures.reshape(1, -1)
# and run a prediction of k nearest neighbors similar to how we would a regression:
prediction = clf.predict(example_measures)
print('Predicted Group:', prediction)
# we get the prediction Class "2" or (according to the notes on the data) Benign cancer! yay!

# Congrats you just ran k nearest neighbors and specified a point.





"""
Euclidean Distance and K Nearest Neighbors
From Scratch:
        
"""

# by hand how euclidean distance works:
from math import sqrt
plot1 = [1,3]
plot2 = [2,5]
euclidean_distance = sqrt( (plot1[0]-plot2[0] )**2 + (plot1[1]-plot2[1])**2 )
print('Euclidean Distance:', euclidean_distance)

# built in function for euclidean
import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
style.use('fivethirtyeight')


## K Nearest Neighbors Intution:
# Where "k" is the number of nearest neighbors to calculate our nearest points for classification. 
# In this case it's easiest to have k be an odd number to decide ballanced cases. (per amount of groups)
# We can get confidence intervals for each point. To get distances, euclidean distances are calcuated
# but these are really inefficent! So sometimes SVMs are better because they scale better up. 

dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7] # does this new point better apply to k or r?
                     # obviously r, but let's check.
for i in dataset:
    for ii in dataset[i]:
        [[plt.scatter(ii[0],ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0], new_features[1])
plt.show()


## Set up a function that does k nearest neighbors:

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
            warnings.warn('K is set to a value lower than the total voting group! Idiot!')
            distances = []
            for group in data:
                for features in data[group]:
                    euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
                #euclidean_distance = np.sqrt(np.sum(( np.array(features)-np.array(predict))**2 ))
                #euclidean_distance = sqrt( (features[0]-predict[0] )**2 + (features[1]-predict[1])**2 )
                distances.append([euclidean_distance, group])
            
            votes = [i[1] for i in sorted(distances)[:k]]
            print(Counter(votes).most_common(1))
            vote_result = Counter(votes).most_common(1)[0][0]
        
            return vote_result


result = k_nearest_neighbors(dataset, new_features, k=3)
print(result)


dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]
[[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset]
# Same as:
##for i in dataset:
##    for ii in dataset[i]:
##        plt.scatter(ii[0],ii[1],s=100,color=i)
        
plt.scatter(new_features[0], new_features[1], s=100)

result = k_nearest_neighbors(dataset, new_features)
plt.scatter(new_features[0], new_features[1], s=100, color = result)  
plt.title("Classification Result")
plt.show()

# We classified it and ran our model and it turned red (group r) as we hoped!





"""
Using KNN for applied data:
    
"""

import random 
# This package will come in soon to shuffle data.

## Remember our definition of k nearest neighbors
def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance,group])
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

## Remember our data retreival: 
df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-99999, inplace=True)
df.drop(['id'], 1, inplace=True)
full_data = df.astype(float).values.tolist()

# Let's shuffle our data, and assign it into test set training set.
random.shuffle(full_data)
test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]


for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

# Create empty set to fill:
correct = 0
total = 0

# Run for loop to fill in values, then report accuracy:
for group in test_set:
    for data in test_set[group]:
        vote = k_nearest_neighbors(train_set, data, k=5)
        if group == vote:
            correct += 1
        total += 1
print('Accuracy:', correct/total)

# We made a model that predicted well 97% precent of the time. KNN works well for linear and 
# non-linear data, but may downfall with outliers, scaling issues, and any bad data (recall useless
# ID inclusion).




"""
Support Vector Models:
    
"""
print('::Support Vector Model Section::')

# In many ways Support Vector Models can be more helpful at classification. 
# We'll explore this here, along with visualizing these. 

# Packages used:
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

class Support_Vector_Machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1:'r',-1:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
    # train
    def fit(self, data):
        self.data = data
        # { ||w||: [w,b] }
        opt_dict = {}

        transforms = [[1,1],
                      [-1,1],
                      [-1,-1],
                      [1,-1]]

        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        # support vectors yi(xi.w+b) = 1
        

        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      # point of expense:
                      self.max_feature_value * 0.001,
                      ]

        
        
        # extremely expensive
        b_range_multiple = 2
        # we dont need to take as small of steps
        # with b as we do w
        b_multiple = 5
        latest_optimum = self.max_feature_value*10
        
        for step in step_sizes:
            w = np.array([latest_optimum,latest_optimum])
            # we can do this because convex
            optimized = False
            while not optimized:
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple),
                                   self.max_feature_value*b_range_multiple,
                                   step*b_multiple):
                    for transformation in transforms:
                        w_t = w*transformation
                        found_option = True
                        # weakest link in the SVM fundamentally
                        # SMO attempts to fix this a bit
                        # yi(xi.w+b) >= 1
                        # 
                        # #### add a break here later..
                        for i in self.data:
                            for xi in self.data[i]:
                                yi=i
                                if not yi*(np.dot(w_t,xi)+b) >= 1:
                                    found_option = False
                                    #print(xi,':',yi*(np.dot(w_t,xi)+b))
                                    
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t,b]

                if w[0] < 0:
                    optimized = True
                    print('Optimized a step.')
                else:
                    w = w - step

            norms = sorted([n for n in opt_dict])
            #||w|| : [w,b]
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0]+step*2
            
        for i in self.data:
            for xi in self.data[i]:
                yi=i
                print(xi,':',yi*(np.dot(self.w,xi)+self.b))            

    def predict(self,features):
        # sign( x.w+b )
        classification = np.sign(np.dot(np.array(features),self.w)+self.b)
        if classification !=0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])
        return classification

    def visualize(self):
        [[self.ax.scatter(x[0],x[1],s=100,color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        # hyperplane = x.w+b
        # v = x.w+b
        # psv = 1
        # nsv = -1
        # dec = 0
        def hyperplane(x,w,b,v):
            return (-w[0]*x-b+v) / w[1]

        datarange = (self.min_feature_value*0.9,self.max_feature_value*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # (w.x+b) = 1
        # positive support vector hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2], 'k')

        # (w.x+b) = -1
        # negative support vector hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min,hyp_x_max],[nsv1,nsv2], 'k')

        # (w.x+b) = 0
        # positive support vector hyperplane
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min,hyp_x_max],[db1,db2], 'y--')

        plt.show()
        
data_dict = {-1:np.array([[1,7],
                          [2,8],
                          [3,8],]),
             
             1:np.array([[5,1],
                         [6,-1],
                         [7,3],])}

svm = Support_Vector_Machine()
svm.fit(data=data_dict)

predict_us = [[0,10],
              [1,3],
              [3,4],
              [3,5],
              [5,5],
              [5,6],
              [6,-5],
              [5,8]]

for p in predict_us:
    svm.predict(p)

svm.visualize()



        
        
        
        
        
        
        
