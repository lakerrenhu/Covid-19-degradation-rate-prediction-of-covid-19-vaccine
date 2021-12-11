# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 19:43:16 2021

@author: laker
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import feyn
from sklearn.model_selection import train_test_split


############# data preparation ####################################################################
data = pd.read_json('C:\\Users\\laker\\Desktop\\pc2\\cs_class\\CAP6545\\project\\train.json', lines=True)
#data_test = pd.read_json('C:\\Users\\laker\\Desktop\\pc2\\cs_class\\CAP6545\\project\\test.json', lines=True)

data = data.drop('index', axis=1) # repetitive index can be dropped

data.query('SN_filter == 1', inplace=True) # filter noise out

length = len(data.iloc[0]['reactivity'])
first_68 = data['structure'].apply(lambda x : x[0: length])  # keep the first 68 bases

# Remove sequences that only contain “.” i.e. unpaired bases
idx_all_dots = [i for i in first_68.index if first_68[i].count('.') == length]
data = data.drop(idx_all_dots)
data.head()

#Preparing the sequences for the QLattice

#each sample consists of one nucleobase (base), and a predicted loop type (loop).
end_pos = len(data.loc[0, 'predicted_loop_type'])

RNA_idx = [j for j in data.index for i in range(0, end_pos)]
pos_idx = [i for j in data.index for i in range(0, end_pos)]

loop_exp = data['predicted_loop_type'].apply(lambda x : list(x)).agg(sum)
base_exp = data['sequence'].apply(lambda x : list(x)).agg(sum)

exp_df = pd.DataFrame({'loop' : loop_exp, 'base': base_exp, 'RNA_idx' : RNA_idx, 'pos_idx' : pos_idx})

react_len = len(data.iloc[0].reactivity)

#add some features about its surrounding neighbours. We call these motifs.
#The motifs are defined by a left-side (5') and right-side (3') window of the two neighbouring bases. 
df = exp_df[exp_df.pos_idx < react_len]
df = df[df.pos_idx >= 5]

df['reactivity'] = df.apply(lambda row: data.loc[row.RNA_idx].reactivity[row.pos_idx], axis=1)

df['sequence'] = data.loc[df.RNA_idx].set_index(df.index).sequence
df['base_left_motif'] = df.apply(lambda x: x['sequence'][:x.pos_idx][-2:], axis=1)
df['base_right_motif'] = df.apply(lambda x: x['sequence'][x.pos_idx + 1:][:2], axis=1)

df['loop_type'] = data.loc[df.RNA_idx].set_index(df.index).predicted_loop_type
df['loop_left_motif'] = df.apply(lambda x: x['loop_type'][:x.pos_idx][-2:], axis=1)
df['loop_right_motif'] = df.apply(lambda x: x['loop_type'][x.pos_idx + 1:][:2], axis=1)
df.columns
#Train, validation and holdout split
train_idx, test_idx = train_test_split(list(data.index),train_size = 0.7, random_state = 42)

#train_idx, remain_idx = train_test_split(list(data.index),train_size = 0.5, random_state = 42)
#valid_idx, holdout_idx = train_test_split(remain_idx,train_size = 0.5, random_state = 42)

train = df.query('RNA_idx == @train_idx')
test = df.query('RNA_idx == @test_idx')
#holdout = df.query('RNA_idx == @holdout_idx')
train.head()


############################################## Qlattice #############################################
# method1: Training a QGraph using all the features
# Connecting to the QLattice
ql = feyn.connect_qlattice()

# Seeding the QLattice for reproducible results
ql.reset(42)

# Output variable
output = 'reactivity'

# Declaring features
features = ['base', 'loop', 'base_left_motif', 'base_right_motif', 'loop_left_motif', 'loop_right_motif']

# Declaring categorical features
stypes = {}
for f in features:
    if train[f].dtype =='object':
        stypes[f] = 'c'

#use AIC as a selection criterion prior to updating the QLattice with the best graphs.
#This is a regression task, which is default for auto_run
models = ql.auto_run(train[features+[output]], output, stypes=stypes, criterion='aic')
# Select the best Model
model_base = models[0]
models[0]  # loss = 8.41E-02, 5 features exist
model_base.plot_regression(train)
# RMSE
model_base.rmse(train)  # training RMSE = 0.29326600591291035
model_base.rmse(test)  #  test RMSE = 0.2908291770568081


#restrict the graph a bit more to force the QLattice to choose the best features
ql.reset(42)
models = ql.auto_run(train[features+[output]], output, stypes=stypes, max_complexity=7, criterion='aic')
model_constrained = models[0]
# Select the best Model
models[0]  #loss = 8.59E-02, four features exist
model_constrained.plot_regression(train)
model_constrained.plot_residuals(data=train)
# RMSE
model_constrained.rmse(train)  # training RMSE = 0.2943841304442142
model_constrained.rmse(test)  #  test RMSE = 0.2918238980879863


# reduct the number of features to see the RMSE

#3-feature model: remove the least important feature: loop-right-motif
ql.reset(42)
features = ['base', 'loop', 'base_right_motif']

models = ql.auto_run(train[features+[output]], output, stypes=stypes, max_complexity=7, criterion='aic')
model_three_features = models[0]
model_three_features

# RMSE
model_three_features.rmse(train)  # training RMSE = 0.2969779761494834
model_three_features.rmse(test)  #  test RMSE = 0.2950248338576319


# the base doesn't supply much information   
ql.reset(42)
features = ['loop', 'base_right_motif']

# Note we're reducing to max complexity of 3 for two features.
models = ql.auto_run(train[features+[output]], output, stypes=stypes, max_complexity=3, criterion='aic')
model_two_features = models[0]
model_two_features

# RMSE
model_two_features.rmse(train)  # training RMSE = 0.30827234251417823
model_two_features.rmse(test)  #  test RMSE = 0.30587580336067266


# one-feature model by base_right_motif
ql.reset(42)
features = ['base_right_motif']

# Note we're reducing to max complexity of 3 for two features.
models = ql.auto_run(train[features+[output]], output, stypes=stypes, max_complexity=3, criterion='aic')
model_one_features = models[0]
model_one_features

# RMSE
model_one_features.rmse(train)  # training RMSE =0.37027418729615524
model_one_features.rmse(test)  #  test RMSE = 0.3715347305653377


# one-feature model by loop
ql.reset(42)
features = ['loop']

# Note we're reducing to max complexity of 3 for two features.
models = ql.auto_run(train[features+[output]], output, stypes=stypes, max_complexity=3, criterion='aic')
model_one_features = models[0]
model_one_features

# RMSE
model_one_features.rmse(train)  # training RMSE = 0.350748388486994
model_one_features.rmse(test)  #  test RMSE = 0.34658224406863863


# encode varibles
# convert categorical variables into numerical values
model_base.params[4]['categories']
model_base.params[5]['categories']
model_base.params[7]['categories']
model_base.params[8]['categories']

# index = 2, base_left_motif, base_right_motif
# index = 6, loop
# index = 7, loop_left_motif, loop_right_motif
# index = 8, base

df2 = df.copy()

target_enc = {'base':{'A': 0, 'G' : 1, 'C' : 2, 'U' : 3},
              'loop': {'S':0, 'E':1, 'H':2, 'I':3, 'X':4, 'M':5, 'B':6},
              'base_left_motif':{'GG':0,'GA':1,'CU':2,'UC':3,'CG':4,'UA':5,
                                 'UG':6,'UU':7,'AC':8,'AG':9,'CA':10,'CC':11,
                                 'GU':12,'AU':13,'AA':14,'GC':15},
              'base_right_motif':{'GG':0,'GA':1,'CU':2,'UC':3,'CG':4,'UA':5,
                                 'UG':6,'UU':7,'AC':8,'AG':9,'CA':10,'CC':11,
                                 'GU':12,'AU':13,'AA':14,'GC':15},
              'loop_left_motif':{'BS':0,'SH':1,'ES':2,'II':3,'IS':4,'XS':5,
                                 'SS':6,'BB':7,'HH':8,'XX':9,'MM':10,'HS':11,
                                 'SI':12,'EE':13,'SB':14,'SM':15,'SX':16,
                                 'MS':17},
              'loop_right_motif':{'BS':0,'SH':1,'ES':2,'II':3,'IS':4,'XS':5,
                                 'SS':6,'BB':7,'HH':8,'XX':9,'MM':10,'HS':11,
                                 'SI':12,'EE':13,'SB':14,'SM':15,'SX':16,
                                 'MS':17}
              }

df2.replace(target_enc, inplace=True)
df2.head()

#Train and holdout (test) split

train2 = df2.query('RNA_idx == @train_idx')
test2 = df2.query('RNA_idx == @test_idx')

train2.head()

######################################## CNN #######################################################

from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

'''using tensorflow under Anaconda needs to activate 'tensorflow' first.
input command in Anaconda Prompt (anaconda3): conda activate tensorflow '''

# Output variable
output = 'reactivity'
# Declaring features
features = ['base', 'loop', 'base_left_motif', 'base_right_motif', 'loop_left_motif', 'loop_right_motif']
xtrain, xtest, ytrain, ytest = train2[features],test2[features],train2[output],test2[output]

#Defining and fitting the model
#We'll define the Keras sequential model and add a one-dimensional convolutional layer. 
#Input shape becomes as it is defined above (6,1).
#We'll add Flatten and Dense layers and compile it with optimizers.

model = Sequential()
model.add(Conv1D(32, 2, activation="relu", input_shape=(6, 1)))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(1))
model.compile(loss="mse", optimizer="adam")
 
model.summary()

#Next, we'll fit the model with train data.
xtrain = xtrain.to_numpy()
xtest  = xtest.to_numpy()
ytrain = ytrain.to_numpy()
ytest  = ytest.to_numpy()

xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1], 1)
xtest = xtest.reshape(xtest.shape[0], xtest.shape[1], 1)

model.fit(xtrain, ytrain, batch_size=12,epochs=200, verbose=0)

#Predicting and visualizing the results
#Now we can predict the test data with the trained model.

ypred = model.predict(xtest)

#We can evaluate the model, check the mean squared error rate (MSE) of the 
# predicted result, and visualize the result in a plot.

print(model.evaluate(xtrain, ytrain))
# loss = 0.07553846389055252

print("MSE: %.4f" % mean_squared_error(ytrain, model.predict(xtrain)))
#MSE: 0.0755, train RMSE = 0.27477263328068174
 
print("MSE: %.4f" % mean_squared_error(ytest, ypred))
#MSE: 0.0800, test RMSE = 0.282842712474619

x_ax = range(len(ypred))
plt.scatter(x_ax, ytest, s=5, color="blue", label="original")
plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
plt.legend()
plt.show()


#####################################  sparse learning: elastic net ######################################

# Output variable
output = 'reactivity'

# Declaring features
features = ['base', 'loop', 'base_left_motif', 'base_right_motif', 'loop_left_motif', 'loop_right_motif']

xtrain, xtest, ytrain, ytest = train2[features],test2[features],train2[output],test2[output]

xtrain = xtrain.to_numpy()
xtest  = xtest.to_numpy()
ytrain = ytrain.to_numpy()
ytest  = ytest.to_numpy()

# use automatically configured elastic net algorithm
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import RepeatedKFold

# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define model
model = ElasticNetCV(cv=cv, n_jobs=-1)

# fit the model via elastic net
model_elastic = model.fit(xtrain,ytrain)
# summarize chosen configuration
print('alpha: %f' % model_elastic.alpha_)  # alpha: 0.000794
print('l1_ratio_: %f' % model_elastic.l1_ratio_) # l1_ratio_: 0.500000
# make a prediction
yhat = model_elastic.predict(xtest)

# prediction performance
print("MSE: %.4f" % mean_squared_error(ytrain, model_elastic.predict(xtrain)))
#MSE: 0.1324, train RMSE = 0.363868107973205
print("MSE: %.4f" % mean_squared_error(ytest, yhat))
#MSE: 0.1320, test RMSE = 0.363318042491699

#####################################  ensemble learning: xgboost ######################################

from sklearn.model_selection import RepeatedKFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
# load the dataset
model = XGBRegressor()
# fit model
model_XGB = model.fit(xtrain, ytrain)
# make a prediction
yhat = model_XGB.predict(xtest)
# prediction performance
print("MSE: %.4f" % mean_squared_error(ytrain, model_XGB.predict(xtrain)))
#MSE: 0.0614, train RMSE = 0.24779023386727736
print("MSE: %.4f" % mean_squared_error(ytest, yhat))
#MSE: 0.0678, test RMSE = 0.2603843313258307



