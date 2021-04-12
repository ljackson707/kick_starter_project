import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#======================================================================================

def logit_model(train, y_train, validate, y_validate):
    
    logit = LogisticRegression(C=1, class_weight={0:1, 1:99},
                               random_state=123)

    #  fit the model on train data
    logit.fit(train, y_train)

    # now use the model to make predictions
    y_pred = logit.predict(train)

    # View raw probabilities (output from the model) (gives proabilities for each observation)
    y_pred_proba = logit.predict_proba(train)
    y_pred_proba = pd.DataFrame(y_pred_proba, columns = ['fail', 'success'])

    print(classification_report(y_train, y_pred ))

    # Test out of sample data
    y_pred_validate = logit.predict(validate)

    print(confusion_matrix(y_validate, y_pred_validate))

    print('Coefficient: \n', logit.coef_)
    print('Intercept: \n', logit.intercept_)

    logit.coef_[0]

    log_coeffs = pd.DataFrame(logit.coef_[0], index = train.columns, columns = ['coeffs']).sort_values(by = 'coeffs', ascending = True)
    odds = np.exp(log_coeffs)

    return odds
    
#======================================================================================

def knn_model(X_train, y_train, X_validate, y_validate, X_test, y_test):
    
    # weights = ['uniform', 'density']
    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_train)
    y_pred_proba = knn.predict_proba(X_train)
    print('Accuracy of KNN classifier on training set: {:.2f}'
         .format(knn.score(X_train, y_train)))
    print(confusion_matrix(y_train, y_pred))
    print(classification_report(y_train, y_pred))

    # Validate Model
    print('Accuracy of KNN classifier on test set: {:.2f}'
         .format(knn.score(X_validate, y_validate)))
    y_pred = knn.predict(X_validate)
    print(classification_report(y_validate, y_pred))

    # Viz Model
    k_range = range(1, 20)
    scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors = k)
        knn.fit(X_train, y_train)
        scores.append(knn.score(X_test, y_test))
    plt.figure()
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.scatter(k_range, scores)
    plt.xticks([0,5,10,15,20])
    plt.show()

   #======================================================================================

def bootstrap_model(X_train, y_train, X_validate, y_validate):
    rf = RandomForestClassifier(bootstrap=True, 
                                class_weight=None, 
                                criterion='gini',
                                min_samples_leaf=3,
                                n_estimators=100,
                                max_depth=3, 
                                random_state=123)
    # Fit the model
    rf.fit(X_train, y_train)
    print(rf.feature_importances_)
    # Make Predictions
    y_pred = rf.predict(X_train)
    # Estimate Probability
    y_pred_proba = rf.predict_proba(X_train)

    # Evaluate Model 
    # Compute the Accuracy
    print('Accuracy of random forest classifier on training set: {:.2f}'
         .format(rf.score(X_train, y_train)))
    # Create a confusion matrix
    print(confusion_matrix(y_train, y_pred))
    print(classification_report(y_train, y_pred))

    # Validate Model
    print('Accuracy of random forest classifier on test set: {:.2f}'
         .format(rf.score(X_validate, y_validate)))
    
#======================================================================================
