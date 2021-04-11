import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

#======================================================================================

def viz_kickstarter(train, kmeans):
    centroids = np.array(train.groupby('cluster')['square_feet', 'logerror'].mean())
    cen_x = [i[0] for i in centroids]
    cen_y = [i[1] for i in centroids]
    # cen_x = [i[0] for i in kmeans.cluster_centers_]
    # cen_y = [i[1] for i in kmeans.cluster_centers_]
    train['cen_x'] = train.cluster.map({0:cen_x[0], 1:cen_x[1], 2:cen_x[2]})
    train['cen_y'] = train.cluster.map({0:cen_y[0], 1:cen_y[1], 2:cen_y[2]})

    colors = ['#DF2020','#2095DF', '#81DF20']
    train['c'] = train.cluster.map({0:colors[0], 1:colors[1], 2:colors[2]})
    #plot scatter chart for Actual species and those predicted by K - Means

    #specify custom palette for sns scatterplot
    colors1 = ['#2095DF','#81DF20' ,'#DF2020']
    customPalette = sns.set_palette(sns.color_palette(colors1))

    #plot the scatterplots

    #Define figure (num of rows, columns and size)
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10,10))

    # plot ax1 
    ax1 = plt.subplot(2,1,1) 
    sns.scatterplot(data = train, x = 'square_feet', y = 'logerror', ax = ax1, palette=customPalette)
    plt.title('County')

    #plot ax2
    ax2 = plt.subplot(2,1,2) 
    ax2.scatter(train.square_feet, train.logerror, c=train.cluster, alpha = 0.6, s=10)
    ax2.set(xlabel = 'square_feet', ylabel = 'logerror', title = 'K - Means')

    # plot centroids on  ax2
    ax2.scatter(cen_x, cen_y, marker='X', c=colors, s=200)
    
    
    train.drop(columns = ['cen_x', 'cen_y', 'c'], inplace = True)
    plt.tight_layout()
    plt.show()
    
#======================================================================================

def OLS_Model(X_train, y_train, X_validate, y_validate):
    # create the model object
    lm = LinearRegression(normalize=True)
    # fit the model to our training data. We must specify the column in y_train,
    # since we have converted it to a dataframe from a series!
    lm.fit(X_train, y_train.logerror)
        # just call y_train.actual_target
    # predict train
    y_train['logerror_pred_lm'] = lm.predict(X_train)
    # evaluate: rmse
    rmse_train_lm = mean_squared_error(y_train.logerror, y_train.logerror_pred_lm)**(1/2)
    # predict validate
    y_validate['logerror_pred_lm'] = lm.predict(X_validate)
    # evaluate: rmse
    rmse_validate_lm = mean_squared_error(y_validate.logerror, y_validate.logerror_pred_lm)**(1/2)
        # make sure you are using x_validate an not x_train
    print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", rmse_train_lm,
          "\nValidation/Out-of-Sample: ", rmse_validate_lm)

#======================================================================================

def ols_actual_vs_predicted(X_train, y_train, X_validate, y_validate):
    # y_validate.head()
    plt.figure(figsize=(16,8))
    plt.plot(y_validate.logerror, y_validate.logerror_pred_median, alpha=.5, color="black", label='_nolegend_')
    plt.annotate("Baseline: Predict Using Median", (16, 9.5))
    plt.plot(y_validate.logerror, y_validate.logerror, alpha=.5, color="black", label='_nolegend_')
    plt.annotate("The Ideal Line: Predicted = Actual", (.5, 3.5), rotation=15.5)
    plt.scatter(y_validate.logerror, y_validate.logerror_pred_lm3,
                alpha=.5, color="darkturquoise", s=100, label="Model 3rd degree Polynomial")
    plt.legend()
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Where are predictions more extreme? More modest?")
    # plt.annotate("The polynomial model appears to overreact to noise", (2.0, -10))
    # plt.annotate("The OLS model (LinearRegression)\n appears to be most consistent", (15.5, 3))
    plt.show()

#======================================================================================

def OLS_hist(X_train, y_train, X_validate, y_validate):
    y_train = pd.DataFrame(y_train)
        # turn it into a single pandas dataframe
    y_validate = pd.DataFrame(y_validate)
        # wrap them as dataframes
    # 1. Predict logerror_pred_mean
        # 2 different aselines of mean and medium
    logerror_pred_mean = y_train['logerror'].mean()
    y_train['logerror_pred_mean'] = logerror_pred_mean
    y_validate['logerror_pred_mean'] = logerror_pred_mean
    # 2. compute logerror_pred_median
        # same process as mean (above)
    logerror_pred_median = y_train['logerror'].median()
    y_train['logerror_pred_median'] = logerror_pred_median
    y_validate['logerror_pred_median'] = logerror_pred_median
    # 3. RMSE of logerror_pred_mean
    rmse_train_mean = mean_squared_error(y_train.logerror, 
                                         y_train.logerror_pred_mean)**(1/2)
        # stick with root mean square error
            # not your only option but that is what we will be using here
                # just because it is eaiest to us and explain
        # remember when you call you it will be your y_true and y_pred
    rmse_validate_mean = mean_squared_error(y_validate.logerror, 
                                            y_validate.logerror_pred_mean)**(1/2)
    lm = LinearRegression(normalize=True)
    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm.fit(X_train, y_train.logerror)
        # just call y_train.actual_target
    # predict train
    y_train['logerror_pred_lm'] = lm.predict(X_train)
    # evaluate: rmse
    rmse_train_lm = mean_squared_error(y_train.logerror, y_train.logerror_pred_lm)**(1/2)
    # predict validate
    y_validate['logerror_pred_lm'] = lm.predict(X_validate)
    # evaluate: rmse
    rmse_validate_lm = mean_squared_error(y_validate.logerror, y_validate.logerror_pred_lm)**(1/2)
    # make sure you are using x_validate an not x_train
    plt.subplots(1, 2, figsize=(8,8), sharey=True)
    sns.set(style="darkgrid")
    plt.title("Comparing the Distribution of appraised_values to Distributions of Predicted appraised_values for the Top Models")
    plt.xlabel("Logerror", size = 15)
    plt.ylabel("appraised_value Count", size = 15)

    plt.subplot(1,2,1)
    plt.hist(y_validate.logerror, color='darkgreen', ec='black', alpha=.5, bins=50)
    plt.title('Actual Logerror', size=15)

    plt.subplot(1,2,2)
    plt.hist(y_validate.logerror, color='mediumblue', alpha=.5,  ec='black', bins=50)
    plt.title('Model: LinearRegression', size=15)
    
#======================================================================================


