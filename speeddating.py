#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 11:26:53 2020

@author: Matt

This project is based on social network analysis. I found a data set for
speed dating on Kaggle...

https://www.kaggle.com/annavictoria/speed-dating-experiment

My initial thought for a social network analysis was to use the Twitter
API and look at followers of a single twitter account. Based on the
followers who followed/intereacted with each other, I was thinking I could
use link prediction to figure out what other followers of that twitter
account may start to follow each other. Since the twitter API is very slow,
I had to think of something different.

The speed dating data is set up in 21 different waves (events). For each
wave, there are males and females who meet, and there is all sorts of data
including gender, prefences, interests, etc. as well as if they matched.

My first goal was to graph the data to get a visual of what it looks like.
Not surprisingly, when you look at all the data at once, the graph shows
21 different circles that are isolated from each other. These represent 
the different waves. When you zoom in and only graph one wave at a time, 
the visual looks much more like a traditional graph.

As far as analysis, I'm thinking that I could build some type of prediction
based on each individual wave. IE. For person1 who was in wave1, I could
use their preferences and matches to predict who they may have matched with 
in waves 2 - 21.

This will create edges between the waves, so they are no longer isolated
from each other. If this doesn't work at the individual level, maybe I
could collectively predict the strength of potential matching of one wave
with another wave?


HOW TO DO MODEL

Can link prediction be used via networkx?
-Answer seems like No. Link Prediction seems to predict what a network will
look like in the future. Since these networks aren't necessarily continuing
to interact, this isn't a true network prediction. However, we can still
visualize the network with a graph!

CORRECTION!...Maybe! Can we use these link prediction scores as features in
a logistic regression model? Would that make sense to do in this case?

Review the link prediction assignment from coursera!

If not, should I build a logistic regression from df_dating looking at
features and using match? If so, I'll use probabilities to set connections.
If so, what is the probability threshold for a potential match?

If I build a logistic regression, how do I then apply that to indivual
people? Do I need to look at the most influential features, and then
see what people have the same preferences for those features?

There are 2 groups of data for this analysis...
1. Data that is filled out before the speed dating event happens
2. Data that is filled out during/after the speed dating event happens

Since we are going to predict whether or not two people may be a good match,
we can only use data that's filled out before the event happens. We don't
want to allow any data during/after the event to influence the prediction,
because we can't factor that in when suggesting people who would meet.

Logistic regression vs. Random forest?

"""

import pandas as pd
import numpy as np
import networkx as nx
import sqlite3
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import validation_curve
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import LogisticRegression

scaler = MinMaxScaler()


# Encoding issue with speeddating file
# https://stackoverflow.com/questions/18171739/unicodedecodeerror-when-reading-csv-file-in-pandas-with-python
df_dating = pd.read_csv('/Users/Matt/Desktop/DataScience/Kaggle/SpeedDating/speeddatingdata.csv', na_values=' ', encoding = "ISO-8859-1" )

# need to rename 'from' column so it works with sqlite3 code later on
df_dating['from_location'] = df_dating['from']
df_dating.drop(['from'], axis=1, inplace=True)

"""
Waves 6-9 are different from the rest because their preferences were scored
differently. Instead of spreading out 100 pts across the preferences, they
scored each preference on a 1-10 scale. The code below converts that by
adding up the total prefence score and calculating the percentage of each
preference. This normalizes the scores across all waves.

After this is completed, there were 5 other questions that were asked.
Based on NaN values, we'll only keep the first 2. 1_1 and 2_1. We'll
drop 3_1 - 5_1 in the code below.
"""
# find column positions to sum up
# df_dating.columns.get_loc('pf_o_att') # 17

# add needed columns for partner preferences
df_dating['pf_o_total'] = df_dating[df_dating.columns[17:23]].sum(axis=1)
df_dating['pf_o_att_new'] = df_dating['pf_o_att'] / df_dating['pf_o_total'] * 100
df_dating['pf_o_sin_new'] = df_dating['pf_o_sin'] / df_dating['pf_o_total'] * 100
df_dating['pf_o_int_new'] = df_dating['pf_o_int'] / df_dating['pf_o_total'] * 100
df_dating['pf_o_fun_new'] = df_dating['pf_o_fun'] / df_dating['pf_o_total'] * 100
df_dating['pf_o_amb_new'] = df_dating['pf_o_amb'] / df_dating['pf_o_total'] * 100
df_dating['pf_o_sha_new'] = df_dating['pf_o_sha'] / df_dating['pf_o_total'] * 100

# drop old columns and column with total since it's not needed
df_dating.drop(['pf_o_total', 'pf_o_att','pf_o_sin','pf_o_int'
                 ,'pf_o_fun','pf_o_amb','pf_o_sha'], axis=1, inplace=True)

df_dating.drop(['attr3_1', 'sinc3_1','intel3_1'
                ,'fun3_1', 'amb3_1'
                ,'attr4_1', 'sinc4_1','intel4_1'
                ,'fun4_1', 'amb4_1', 'shar4_1'
                ,'attr5_1', 'sinc5_1','intel5_1'
                ,'fun5_1', 'amb5_1'
                ], axis=1, inplace=True)



# Create dataframe for graph
df_graph = df_dating[['iid', 'pid', 'gender','match']].copy()

# Drop any NaN values
df_graph.dropna(inplace=True) # HERE OR IN DF_DATING?!?!?!?

# Create color column for matching that can be used as an edge if needed
#df_graph['color'] = df_graph['match'].apply(lambda x: 'green' if x == 1 else 'black')

# convert 'pid' values from float to integer to remove decimals
df_graph['pid'] = df_graph['pid'].apply(lambda x: int(x))

# This will reduce the network down to the first wave of speed dating
#df_graph = df_graph.loc[df_graph['iid'] <= 20] # Remove this later!
#df_graph = df_graph.loc[(df_graph['iid'] <= 20) & (df_graph['match'] == 1)] # Remove this later!
df_graph = df_graph.loc[df_graph['match'] == 1] # Remove this later!

# create nodes for graph
nodes = df_graph['iid'].unique()

# create edges for graph
edges = list(zip(df_graph['iid'], df_graph['pid']))

G = nx.Graph()
G.add_nodes_from(nodes, gender=df_graph['gender'])
G.add_edges_from(edges, colorfordraw='black')

# draw() documentation
# https://networkx.github.io/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html
#nx.draw(G, with_labels=True, edge_color=df_graph['color'])



df_model = df_dating[['gender', 'samerace', 'age_o', 'race_o'
                      ,'age' ,'field', 'race', 'imprace'
                      ,'imprelig', 'from_location', 'goal', 'date', 'go_out'
                      ,'career', 'sports', 'tvsports', 'exercise', 'dining'
                      ,'museums', 'art', 'hiking', 'gaming', 'clubbing'
                      ,'reading', 'tv', 'theater', 'movies', 'concerts'
                      ,'music', 'shopping', 'yoga', 'exphappy'
                      ,'attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1'
                      ,'shar1_1', 'attr2_1', 'sinc2_1', 'intel2_1', 'fun2_1'
                      ,'amb2_1', 'shar2_1', 'pf_o_att_new'
                      ,'pf_o_sin_new', 'pf_o_int_new', 'pf_o_fun_new'
                      ,'pf_o_amb_new', 'pf_o_sha_new', 'match'
                      ]].copy()

df_model.dropna(inplace=True)

y = df_model['match']
X = df_model[['gender', 'samerace', 'age_o', 'race_o'
                      ,'age' ,'field', 'race', 'imprace'
                      ,'imprelig', 'from_location', 'goal', 'date', 'go_out'
                      ,'career', 'sports', 'tvsports', 'exercise', 'dining'
                      ,'museums', 'art', 'hiking', 'gaming', 'clubbing'
                      ,'reading', 'tv', 'theater', 'movies', 'concerts'
                      ,'music', 'shopping', 'yoga', 'exphappy'
                      ,'attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1'
                      ,'shar1_1', 'attr2_1', 'sinc2_1', 'intel2_1', 'fun2_1'
                      ,'amb2_1', 'shar2_1', 'pf_o_att_new'
                      ,'pf_o_sin_new', 'pf_o_int_new', 'pf_o_fun_new'
                      ,'pf_o_amb_new', 'pf_o_sha_new'
                      ]]

# Since some of the features are text, we need to convert the dataframe
# to dummies so we can use it for prediction
X_dummies = pd.get_dummies(X)

# Need to add some fields that are in later prediction group
"""
for xt in X_dummies.columns:
    if xt not in prediction_dummies.columns:
        print(xt)


X_dummies['field_Economics, English'] = 0
X_dummies['field_Speech Languahe Pathology'] = 0
X_dummies['from_location_India, Holland'] = 0
X_dummies['from_location_Woburn, MA'] = 0
X_dummies['career_don\'t know'] = 0
X_dummies['career_entrepeneur'] = 0

This part is weird because the first time I ran this, it seemed like the
column values were uneven and they needed to be adjusted. However, the
model seems to be working now. 

Regardless, the for statement above is a quick way to figure out what
columns in one dataframe may be missing from another dataframe!

It does raise the question for models in the future. If you train/test
with one population, and have a true holdout, how do you easily run the
model on the holdout if one of their features has a text value that
aren't in the train/test sets?
"""

# Create train and test populations
X_train, X_test, y_train, y_test = train_test_split(X_dummies, y, random_state = 0)

# Fit and transform X_train and transform X_test
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create dataframe that runs multiple versions of logistic regression
# based on different C values. We'll then know what's the ideal C value
# based on the test score!
scores_list = []

# function to run multiple logistic regressions
def best_logreg():
    C_list = [0.001, 0.01, 0.1, 1, 10, 100] # Higher C = LESS regularization!
    for c in C_list:
        logreg = LogisticRegression(C=c, max_iter=10000).fit(X_train_scaled, y_train)
        logreg_scaled_score_train = logreg.score(X_train_scaled, y_train)
        logreg_scaled_score_test = logreg.score(X_test_scaled, y_test)
        scores_list.append(['logistic regression', c, np.sum(logreg.coef_ != 0)
                    , logreg_scaled_score_train, logreg_scaled_score_test])

best_logreg()

df_scores = pd.DataFrame(scores_list, columns=['model_type', 'C', 'features_kept'
                                               ,'r2_train', 'r2_test'])

# Since C = 0.1 has the highest test score, that's what we'll use
logreg = LogisticRegression(C=0.1, max_iter=10000).fit(X_train_scaled, y_train)

"""
Below is code to get the top 10 feature importances of the model

Since LogisticRegression() class doesn't have a feature_importance_
function, we are using coef_[0], which was suggested online. It may not
be the correct way, but it seems like it is
"""

# Set coefficient scores for all the features of the logistic regression
fi = logreg.coef_[0]

# Sort coefficient scores in descending order
top_10_values = np.sort(fi)[::-1].tolist()[0:10]

# Get the index values of the coefficient scores in descending order
top_10 = np.argsort(fi)[::-1][0:10]

# Create top 10 list and populate it with feature name
top_10_list = []   
for t in top_10:
    top_10_list.append(X_dummies.columns[t])


"""
Now that we have our model and our most important features, we need
to create a dataframe with new records for IIDs and PIDs that were
in different waves. This may be able to be done via python, but it
definitely CAN be done via a join. For this, try using sqlite3. It's
already imported at the top of the code, but here are 2 links that have
documentation for sqlite3...

https://towardsdatascience.com/starting-with-sql-in-python-948e529586f2
https://docs.python.org/3/library/sqlite3.html

After that, you need to create rows (edges)
that connect people from the different waves and predict the probability
that they would match. 

STEPS...
1. Figure out columns that are needed
2. Create Man table
3. Create Woman table
4. Join with necessary columns but make sure the pairs haven't already met

"""

# Create connection to a new database
sql_connection = sqlite3.connect('dating_for_prediction.db')

# Create cursor object that will be used to execute SQL code
cursor = sql_connection.cursor()

# Create a table in our 'new_connections' database
df_dating.to_sql('df_dating', con=sql_connection)
             
# Commit (save) the changes
# sql_connection.commit()

# Code to drop table if needed
# cursor.execute("DROP TABLE woman_table")

# Create queries to handle steps 2-4 above
query_man_table = "SELECT iid, gender, wave, age, field, race, imprace, imprelig, from_location, goal, date, go_out, career, sports, tvsports, exercise, dining, museums, art, hiking, gaming, clubbing, reading, tv, theater, movies, concerts, music, shopping, yoga, exphappy, attr1_1, sinc1_1, intel1_1, fun1_1, amb1_1, shar1_1, attr2_1, sinc2_1, intel2_1, fun2_1, amb2_1, shar2_1 FROM df_dating WHERE gender = 1"

query_woman_table = "SELECT iid, gender, wave, age, field, race, imprace, imprelig, from_location, goal, date, go_out, career, sports, tvsports, exercise, dining, museums, art, hiking, gaming, clubbing, reading, tv, theater, movies, concerts, music, shopping, yoga, exphappy, attr1_1, sinc1_1, intel1_1, fun1_1, amb1_1, shar1_1, attr2_1, sinc2_1, intel2_1, fun2_1, amb2_1, shar2_1 FROM df_dating WHERE gender = 0"

query_dating_new_m = "SELECT DISTINCT m.iid as person_iid, w.iid as partner_iid, m.gender, case when m.race = w.race then 1 else 0 end as samerace, w.age as age_o, w.race as race_o, m.age, m.field, m.race, m.imprace, m.imprelig, m.from_location, m.goal, m.date, m.go_out, m.career, m.sports, m.tvsports, m.exercise, m.dining, m.museums, m.art, m.hiking, m.gaming, m.clubbing, m.reading, m.tv, m.theater, m.movies, m.concerts, m.music, m.shopping, m.yoga, m.exphappy, m.attr1_1, m.sinc1_1, m.intel1_1, m.fun1_1, m.amb1_1, m.shar1_1, m.attr2_1, m.sinc2_1, m.intel2_1, m.fun2_1, m.amb2_1, m.shar2_1, w.attr1_1 as pf_o_att_new, w.sinc1_1 as pf_o_sin_new, w.intel1_1 as pf_o_int_new, w.fun1_1 as pf_o_fun_new, w.amb1_1 as pf_o_amb_new, w.shar1_1 as pf_o_sha_new FROM man_table m LEFT JOIN woman_table w WHERE m.wave != w.wave"
query_dating_new_w = "SELECT DISTINCT w.iid as person_iid, m.iid as partner_iid, w.gender, case when w.race = m.race then 1 else 0 end as samerace, m.age as age_o, m.race as race_o, w.age, w.field, w.race, w.imprace, w.imprelig, w.from_location, w.goal, w.date, w.go_out, w.career, w.sports, w.tvsports, w.exercise, w.dining, w.museums, w.art, w.hiking, w.gaming, w.clubbing, w.reading, w.tv, w.theater, w.movies, w.concerts, w.music, w.shopping, w.yoga, w.exphappy, w.attr1_1, w.sinc1_1, w.intel1_1, w.fun1_1, w.amb1_1, w.shar1_1, w.attr2_1, w.sinc2_1, w.intel2_1, w.fun2_1, w.amb2_1, w.shar2_1, m.attr1_1 as pf_o_att_new, m.sinc1_1 as pf_o_sin_new, m.intel1_1 as pf_o_int_new, m.fun1_1 as pf_o_fun_new, m.amb1_1 as pf_o_amb_new, m.shar1_1 as pf_o_sha_new FROM woman_table w LEFT JOIN man_table m WHERE w.wave != m.wave"


# If we want to save the results to a dataframe, we can use the below...

# save results of query to dataframe
df_man_table = pd.read_sql_query(query_man_table,sql_connection)

# insert dataframe into a SQL table. Unfortunately, it doesn't seem
# like we can run a query in SQLite3 and have it go directly into a
# table. Maybe possible, but need more SQLite3 knowledge.
df_man_table.to_sql('man_table', con=sql_connection)

df_woman_table = pd.read_sql_query(query_woman_table,sql_connection)
df_woman_table.to_sql('woman_table', con=sql_connection)

df_dating_new_m = pd.read_sql_query(query_dating_new_m,sql_connection)

df_dating_new_w = pd.read_sql_query(query_dating_new_w,sql_connection)

# Combine dataframes for men and women
df_for_prediction = pd.concat([df_dating_new_m, df_dating_new_w])

# drop NaN values
df_for_prediction.dropna(inplace=True)


# Drop all tables in database
cursor.execute("DROP TABLE df_dating")
cursor.execute("DROP TABLE man_table")
cursor.execute("DROP TABLE woman_table")

# Close SQLite3 connection
sql_connection.close()


"""
Now that we have our dataframe with all of the possible edges, we'll
prepare the dataframe for prediction
"""

# Remove IDs from prediction data by creating a seperate dataframe
# specifically for the edge pairs
df_for_prediction_iids = df_for_prediction[df_for_prediction.columns[0:2]].copy()
df_for_prediction_features = df_for_prediction[df_for_prediction.columns[2:]].copy()

# Convert Prediction features to dummy dataframe
prediction_dummies = pd.get_dummies(df_for_prediction_features)
prediction_dummies_scaled = scaler.transform(prediction_dummies)


# Code to get list of probabilities
y_proba_logreg = logreg.predict_proba(prediction_dummies_scaled)[:,1]

# Add Gender and match probability to the edge pairs
df_for_prediction_iids['gender'] = df_for_prediction_features['gender']
df_for_prediction_iids['match_probability'] = y_proba_logreg

# Create dataframe with match probabilities >= .50. This is the
# threshold for matching based on the logistic regression. Ideally,
# this threshold would be higher (maybe .75?), but for the data we had
# the probabilities didn't go that high
df_final = df_for_prediction_iids.loc[df_for_prediction_iids['match_probability'] >= .50]

"""
Here's code to look at the graph of match_potential
# create nodes for graph
nodes = df_final['person_iid'].unique()

# create edges for graph
edges = list(zip(df_final['person_iid'], df_final['partner_iid']))

G2 = nx.Graph()
G2.add_nodes_from(nodes)
G2.add_edges_from(edges)

# draw() documentation
# https://networkx.github.io/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html
nx.draw(G2, node_size = 25)

....HOWEVER, we want to add these edges to our original graph...
"""

# create new edges for graph based on match probability >= .50
edges2 = list(zip(df_final['person_iid'], df_final['partner_iid']))

# add new edges to our original graph G
G.add_edges_from(edges2, colorfordraw='green')

# draw() documentation
# https://networkx.github.io/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html
nx.draw(G, node_size=5)


