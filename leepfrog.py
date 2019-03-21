#!/usr/bin/env python
# coding: utf-8

# <h2> Preprocessing </h2>

# In[28]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


# In[29]:


#load training and eval data
training = pd.read_csv("training.psv", sep= "|")
training['label'] = 'train'
evals = pd.read_csv("eval.psv", sep ="|")
evals['label'] = 'eval'

# Concatinate the two data frames and separate them by labelling either training or eval
concat_df = pd.concat([training , evals], axis = 0,sort=False)

#split course into course name and course number
concat_df[['course','course_number']] = concat_df.course.str.split(":",expand = True,)

#extract the training majors
training_majors = training[['student_id','major']].drop_duplicates()

#encode grades by value. 12 grades taken and rest are dropped
concat_df['grade'] = concat_df.grade.replace(['A+','A','A-','B+','B','B-','C+','C','C-','D+','D','D-'],
            [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])

#dropping the remaining grade entries
dropped_grades = ['S','AUS','AUU','F','IP','N','P','U','R','WX','I']
concat_df = concat_df[~concat_df['grade'].isin(dropped_grades)]


#grouping entries by course and label
concat_df.grade = concat_df.grade.astype(int)
concat_df.course_number = concat_df.course_number.astype(int)

#finding the difficulty of a course by using the course number's first digit
concat_df['difficulty']= (concat_df.course_number/100).astype(int)
concat_df.difficulty = concat_df.difficulty.astype(int)

#now grouping data on the basis of student ids
#taking mean of the weighted grades, maximum of the course diificulty and freq of the number of times
#a course was taken by a student in all years
concat_df['freq'] = 1
grouped = concat_df.groupby(by=['student_id','course','label'], as_index=False).agg({'grade':np.mean, 'freq':np.sum, 'difficulty':np.max})

#now sorting the data on the basis of frequency of a course taken
sorted_df = grouped.sort_values(['student_id', 'freq'], ascending=False)

#taking the first row for each student id. I will train on this reduced dataset
concat_full = sorted_df.groupby('student_id', as_index=False).first().reset_index()

#OneHot encoding course
concat_full['course'] = pd.Categorical(concat_full['course'])

dfDummies = pd.get_dummies(concat_full['course'], prefix = 'course_')

concat_full = pd.concat([concat_full, dfDummies], axis=1)

#OneHot encoding diificulty
concat_full['difficulty'] = pd.Categorical(concat_full['difficulty'])

dfDummies = pd.get_dummies(concat_full['difficulty'], prefix = 'diff_')

concat_full = pd.concat([concat_full, dfDummies], axis=1)

#splitting the concatenated data back to training and eval on the basis of label
training = concat_full[concat_full.label == 'train']
testing = concat_full[concat_full.label == 'eval']

#merge the majors on the training set
training_full = training.merge(training_majors, on='student_id')

#dropping unneccesary data columns
training_data = training_full.drop(['course','student_id','difficulty', 'index','label'], axis=1)
testing_data = testing.drop(['course','student_id','difficulty', 'index','label'], axis=1)


# # Training 

# In[31]:


#splitting the training set into training and testing by a ratio 70%-30%
labels = np.array(training_data['major'])
training_new1= training_data.drop('major', axis = 1)
feature_list = list(training_new1.columns)
training_new1 = np.array(training_new1)

train_features, test_features, train_labels, test_labels = train_test_split(training_new1, labels, test_size = 0.30)

#Using random forest classifier for training
rf = RandomForestClassifier()

#Grid-Search on a few parameters of Random Forest
n_estimators = [200, 300, 400]
max_features = ['log2', 'sqrt']
max_depth = [7, 8, 9]
param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
              }
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 10, n_jobs = -1, verbose = 2)

#fitting the model
grid_search.fit(train_features, train_labels)

#storing the cross-val results
crossval_results = grid_search.cv_results_

#store means for all 10 folds
means = grid_search.cv_results_['mean_test_score']

#get the best parameters
best = grid_search.best_params_

#The best parametrs I got were n_estimators = 400, max_features='auto', max_depth= 9
rf2 = RandomForestClassifier(n_estimators= best['n_estimators'], max_features = best['max_features'], max_depth=best['max_depth'])
rf2.fit(train_features,train_labels)

#Obtaining training and testing accuracy
score_train = rf2.score(train_features, train_labels)
score_test = rf2.score(test_features, test_labels)


# # Prediciton

# In[33]:


#predict model results on testing data
predict = rf2.predict(testing_data)

#store the probabilities of test data for each class
prob = rf2.predict_proba(testing_data)

#to find the corresponding classes, and chose top three classes
classes = rf2.classes_
dictionary_object = list(np.array([zip(classes, p) for p in prob]))

l = []
for d in dictionary_object:
    sorted_data = sorted(d, key=lambda tup: tup[1], reverse = True)
    l.append(sorted_data[0:3])
    
#after sorting data in descending order, we obtain top three majors
testing['major1'] = [row[0][0] for row in l]
testing['major2'] = [row[1][0] for row in l]
testing['major3'] = [row[2][0] for row in l]

#left-merge to original evals data
temp = evals.drop(['major1','major2','major3'], axis=1)
temp = temp.merge(testing, how='left', on='student_id')
temp1 = temp.iloc[:,:4]
temp2 = temp.iloc[:,123:126]
result = temp1.join(temp2)

#final result 
result = result.rename(index=str, columns={'course_x': 'course', 'grade_x':'grade'})
result.to_csv('/Users/akankshabhattacharyya/Downloads/Project_job/assignment.csv')


# In[ ]:




