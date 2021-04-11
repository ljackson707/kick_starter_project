import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#======================================================================================

def get_kickstarter_data():
    '''
    Grab our data from path and read as dataframe
    '''
    
    df = pd.read_csv('KickStarter.csv')
    
    return df

#======================================================================================

def missing_zero_values_table(df):
    '''This function will look at any data set and report back on zeros and nulls for every column while also giving percentages of total values
        and also the data types. The message prints out the shape of the data frame and also tells you how many columns have nulls '''
    zero_val = (df == 0.00).astype(int).sum(axis=0)
    null_count = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mz_table = pd.concat([zero_val, null_count, mis_val_percent], axis=1)
    mz_table = mz_table.rename(
    columns = {0 : 'Zero Values', 1 : 'null_count', 2 : '% of Total Values'})
    mz_table['Total Zeroes + Null Values'] = mz_table['Zero Values'] + mz_table['null_count']
    mz_table['% Total Zero + Null Values'] = 100 * mz_table['Total Zeroes + Null Values'] / len(df)
    mz_table['Data Type'] = df.dtypes
    mz_table = mz_table[
        mz_table.iloc[:,1] >= 0].sort_values(
        '% of Total Values', ascending=False).round(1)
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns and " + str(df.shape[0]) + " Rows.\n"      
            "There are " +  str((mz_table['null_count'] != 0).sum()) +
          " columns that have NULL values.")
    return mz_table

#======================================================================================

def rename_columns(df):
    return df.rename(columns={'customerID':'customer_id',
                                'SeniorCitizen':'senior_citizen',
                                'Partner':'partner',
                                'Dependents':'dependents',
                                'PhoneService':'phone_service',
                                'MultipleLines':'multiple_lines',
                                'InternetService':'internet_service',
                                'OnlineSecurity':'online_security',
                                'StreamingTV':'streaming_tv',
                                'StreamingMovies':'streaming_movies',
                                'Contract':'contract',
                                'PaperlessBilling':'paperless_billing',
                                'PaymentMethod':'payment_method',
                                'MonthlyCharges':'monthly_charges'})

#======================================================================================

def remove_columns(df):  
    df = df.drop(columns=['pledged', 'usd pledged', 'ID', 'name', 'category'])
    df.drop(df.index[df['state'] == 'canceled'], inplace = True)
    df.drop(df.index[df['state'] == 'live'], inplace = True)
    df.drop(df.index[df['state'] == 'suspended'], inplace = True)
    return df

def handle_missing_values(df, prop_required_column = .5, prop_required_row = .75):
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df

def data_prep(df, cols_to_remove=[], prop_required_column=.5, prop_required_row=.75):
    df = remove_columns(df, cols_to_remove)
    df = handle_missing_values(df, prop_required_column, prop_required_row)
    return df


#======================================================================================

def data_split(df, stratify_by='state_successful'):
    '''
    this function takes in a dataframe and splits it into 3 samples, 
    a test, which is 20% of the entire dataframe, 
    a validate, which is 24% of the entire dataframe,
    and a train, which is 56% of the entire dataframe. 
    It then splits each of the 3 samples into a dataframe with independent variables
    and a series with the dependent, or target variable. 
    The function returns 3 dataframes and 3 series:
    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test. 
    '''
    # split df into test (20%) and train_validate (80%)
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)

    # split train_validate off into train (70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=['state_successful'])
    y_train = pd.DataFrame(train['state_successful'])
    
    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=['state_successful'])
    y_validate = pd.DataFrame(validate['state_successful'])
    
    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=['state_successful'])
    y_test = pd.DataFrame(test['state_successful'])
    
    return train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test

#======================================================================================

def scale_my_data(train, validate, test):
    scale_columns = ["backers", "usd_pledged_real", "usd_goal_real", "currency_type_val", "country_name_val", "number_of_days"]
    scaler = MinMaxScaler()
    scaler.fit(train[scale_columns])

    train_scaled = scaler.transform(train[scale_columns])
    validate_scaled = scaler.transform(validate[scale_columns])
    test_scaled = scaler.transform(test[scale_columns])
    #turn into dataframe
    train_scaled = pd.DataFrame(train_scaled)
    validate_scaled = pd.DataFrame(validate_scaled)
    test_scaled = pd.DataFrame(test_scaled)
    #rename columns
    train_scaled = train_scaled.rename(columns={ 0 : "backers"})
    train_scaled = train_scaled.rename(columns={ 1 : "usd_pledged_real"})
    train_scaled = train_scaled.rename(columns={ 2 : "usd_goal_real"})
    train_scaled = train_scaled.rename(columns={ 3 : "currency_type_val"})
    train_scaled = train_scaled.rename(columns={ 4 : "country_name_val"})
    train_scaled = train_scaled.rename(columns={ 5 : "number_of_days"})
    
    validate_scaled = train_scaled.rename(columns={ 0 : "backers"})
    validate_scaled = train_scaled.rename(columns={ 1 : "usd_pledged_real"})
    validate_scaled = train_scaled.rename(columns={ 2 : "usd_goal_real"})
    validate_scaled = train_scaled.rename(columns={ 3 : "currency_type_val"})
    validate_scaled = train_scaled.rename(columns={ 4 : "country_name_val"})
    validate_scaled = train_scaled.rename(columns={ 5 : "number_of_days"})
    
    test_scaled = train_scaled.rename(columns={ 0 : "backers"})
    test_scaled = train_scaled.rename(columns={ 1 : "usd_pledged_real"})
    test_scaled = train_scaled.rename(columns={ 2 : "usd_goal_real"})
    test_scaled = train_scaled.rename(columns={ 3 : "currency_type_val"})
    test_scaled = train_scaled.rename(columns={ 4 : "country_name_val"})
    test_scaled = train_scaled.rename(columns={ 5 : "number_of_days"})
    
    return train_scaled, validate_scaled, test_scaled

#======================================================================================

def get_dummies(df):
    df = pd.get_dummies(df, columns=['state'], dtype = int)
    return df 

#======================================================================================

def turn_to_time(df):
    df.launched = pd.to_datetime(df.launched)
    df.deadline = pd.to_datetime(df.deadline)
    df1 = pd.DataFrame([int(i.days) for i in (df.deadline - df.launched)])
    df1.columns=['number_of_days']
    df.reset_index(drop = True)
    df = df.join(df1, how="inner")
    return df

#======================================================================================

# Let's use IQR for the entire dataset
    #backers
def handle_outliers_backers(df):
    q1 = df.backers.quantile(.25)
    q3 = df.backers.quantile(.75)

    iqr = q3 - q1

    multiplier = 1.5
    upper_bound = q3 + (multiplier * iqr)
    lower_bound = q1 - (multiplier * iqr)

    # Let's filter out the low outliers
    df = df[df.backers > lower_bound]
    # lets say give us everyhting less than the upper bound
    df = df[df.backers < upper_bound]
    plt.hist(df.backers, 100, range=[0, 100], align='mid')
    plt.show
    return df

#======================================================================================
    
def handle_outliers_usd_pledged_real(df):
    #usd_pledged_real
    q1_u = df.usd_pledged_real.quantile(.25)
    q3_u = df.usd_pledged_real.quantile(.75)

    iqr_u = q3_u - q1_u

    multiplier = 1.5
    upper_bound_u = q3_u + (multiplier * iqr_u)
    lower_bound_u = q1_u - (multiplier * iqr_u)

    # Let's filter out the low outliers
    df = df[df.usd_pledged_real > lower_bound_u]
    # lets say give us everyhting less than the upper bound
    df = df[df.usd_pledged_real < upper_bound_u]
    plt.hist(df.usd_pledged_real, 100, range=[0, 862], align='mid')
    plt.show
    return df

#======================================================================================

def handle_outliers_usd_goal_real(df):
    #usd_goal_real
    q1_s = df.usd_goal_real.quantile(.25)
    q3_s = df.usd_goal_real.quantile(.75)

    iqr_s = q3_s - q1_s

    multiplier = 1.5
    upper_bound_s = q3_s + (multiplier * iqr_s)
    lower_bound_s = q1_s - (multiplier * iqr_s)

    # Let's filter out the low outliers
    df = df[df.usd_goal_real > lower_bound_s]
    # lets say give us everyhting less than the upper bound
    df = df[df.usd_goal_real < upper_bound_s]
    plt.hist(df.usd_goal_real, 100, range=[0, 50000], align='mid')
    plt.show
    return df

#======================================================================================