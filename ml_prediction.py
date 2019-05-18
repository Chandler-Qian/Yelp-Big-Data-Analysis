
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

df = pd.read_pickle('yelp_dataframe.pkl')

#graph
plt.hist([df[df['is_closed']==0].dropna()['stars'].values,\
          df[df['is_closed']==1].dropna()['stars'].values],\
        label=['Open','Closed'],color=['k','#c41200'])
plt.legend()
plt.title('Yelp Stars Histogram')
plt.xlabel('Yelp Star Rating')
plt.ylabel('Total Number of Restaurants')

df_ml = df_unprocessed[['review_count','stars','price','oldest_review','std_of_stars','reviews_per_week',\
                        'median_of_stars','reactions_per_week','stars_linear_coef','restaurant_density',\
                        'restaurant_similar_density','zreview_count_all','zstar_all','zprice_all','zreview_per_week_all',\
                        'is_claimed','is_chain','is_closed']]

df_ml_clean = df_ml.dropna(axis = 0)

df_ml_features = df_ml_clean.drop('is_closed',axis = 1)
df_ml_target = df_ml_clean['is_closed']

clf = LogisticRegression(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(df_ml_features, df_ml_target, test_size = 0.2, random_state = 10,\
                                                    stratify = df_ml_target)
clf.fit(X_train,list(y_train.values))
y_pred = clf.predict(X_test)
print('Accuracy: ',clf.score(X_test,list(y_test.values)))
print('Precision: ',precision_score(list(y_test.values),y_pred))
print('Recall: ',recall_score(list(y_test.values),y_pred))
print('F1 Score: ',f1_score(list(y_test.values),y_pred))
print('Confusion Matrix: \n',confusion_matrix(list(y_test.values), y_pred))

param_grid = {'penalty' : ['l1','l2'],'C': [0.01,0.1,1.,10.], 'intercept_scaling': [0.0005,0.001,0.005,0.01,0.1,1.,10.]}
scorer = make_scorer(precision_score,pos_label=False)
gscv = GridSearchCV(clf,param_grid,scoring=scorer)
gscv.fit(X_train_scaled,list(y_train_scaled.values))
clf_optimized = gscv.best_estimator_

#after optimization
y_pred_scaled = clf_optimized.predict(X_test_scaled)
print('Accuracy: ',clf_optimized.score(X_test_scaled,list(y_test_scaled.values)))
print('Precision: ',precision_score(list(y_test_scaled.values),y_pred_scaled))
print('Recall: ',recall_score(list(y_test_scaled.values),y_pred_scaled))
print('F1 Score: ',f1_score(list(y_test_scaled.values),y_pred_s	caled))
print('Confusion Matrix: \n',confusion_matrix(list(y_test_scaled.values), y_pred_scaled))

abs(np.std(X_test_scaled,axis=0)*clf_optimized.coef_)
importances_scaled_sign = np.std(X_test_scaled,axis=0)*clf_optimized.coef_[0]

