# kick_starter_project
This is a repo for my individual project over the Kick Starter dataset. I will be creating a classification algorithm predicting whether or not a Kick-Starter Project becomes successful based on funding period, country, main category, and pledge amount.  Link to data set : https://www.kaggle.com/kemical/kickstarter-projects

Kick Starter Project

# <a name="top"></a> Kick Starter Project - README.md
![Zillow Logo](https://github.com/ljackson707/kick_starter_project/raw/main/chart_images/kickstarter_logo.png)
​
***
[[Project Description](#project_description)]
[[Project Planning](#planning)]
[[Key Findings](#findings)]
[[Data Dictionary](#dictionary)]
[[Acquire & Prep](#acquire_and_prep)]
[[Data Exploration](#explore)]
[[Statistical Analysis](#stats)]
[[Modeling](#model)]
[[Conclusion](#conclusion)]
___
​
​
## <a name="project_description"></a>Project Description:
[[Back to top](#top)]

<details>
  <summary>Click to expand!</summary>

### Description
- I want to understand why certain kick starter projects fail or succeed depending on the fourteen available features. I would like to use this data to predict outcomes of future projects. 

### Goals
- My goal is to uncover what features lead to a successful kick starter project.
Utilize clustering models to find these drivers.
Also using dummies to better segment each data column as needed.
Be evaluated through peer review on Monday.

### Where did you get the data?
- Within the Kaggle online database. I downloaded the file as a CSV and ran it through a function called get_kickstarter_data that took in the CSV and read it and produced a pandas data frame for further manipulation.


</details>
    
    
## <a name="planning"></a>Project Planning: 
[[Back to top](#top)]
<details>
  <summary>Click to expand!</summary>

### Projet Outline:
    
- Acquisiton of kick starter data through Kaggle online database. Downloaded as CSV then used get_kickstarter_data to read CSV and convert into Pandas Data Frame for easier manipulation.
- Prepare and clean data with python - Jupyter Labs
- Explore data
    - if value are what the dictionary says they are
    - null values
        - are the fixable or should they just be deleted
    - categorical or continuous values
    - Make graphs that show relations to target variable and distributions of each feature within the data set.
- Run statistical analysis
- Modeling
    - Make multiple models
    - Pick best model
    - Test Data
    - Conclude results
        
### Hypothesis
- Having a successful project depends on if the projects start date. (people may be more willing to invest in projects depending on the month.   
- Having a successful project depends on the USD_pledged to USD_goal ratio.
- Having a successful project depends on which country the project resides in.

### Target variable
- state (if the project was a fail or success)

</details>

    
## <a name="findings"></a>Key Findings:
[[Back to top](#top)]

<details>
  <summary>Click to expand!</summary>

### Explore:

# <a name="top"></a> Correlation Chart - README.md
![Zillow Logo](https://github.com/ljackson707/kick_starter_project/raw/main/chart_images/corr_table.png)

# Findings:
-----------------------------------------------------------------------
- I found that the highest contriubter to the success of a project is the number of backers and pledged usd.
- I want to know how to predict how many backers or the amount of pledged money you will get
- What feature influences the backers or the ammount spent twords a project?
-----------------------------------------------------------------------
- I also found that the there is a slight negative correlation tied to usd_goal_real and state_success. This means that as the goal of projects increase there is a negative impact twords the overall success of the project. 
- From this we could look at where the 'sweet spot' for goal ammounts to increase the liklyhood of getting funded and being successful.
-----------------------------------------------------------------------

# <a name="top"></a> Usd Goal Real vs Usd Pledge Real - README.md
![Zillow Logo](https://github.com/ljackson707/kick_starter_project/raw/main/chart_images/usd_g_real_vs_usd_p_real.png)

# Findings:
-----------------------------------------------------------------------
- I discovered that goal amount has a negative impact on the success of the project. I found that the majority of succeful projects lied between 10 and roughly 8000 dollar range. 
- I have determined to better ones chances at having a successful project they would need to set their project goal at around 5000 dollars.
- This feature does not determin you success on its own, I need to look at the main category as well as tyhe number of backers to see if these features can be accurate identifiers for project success. 
-----------------------------------------------------------------------

# <a name="top"></a> Count Usd Goal Real - README.md
![Zillow Logo](https://github.com/ljackson707/kick_starter_project/raw/main/chart_images/count_usd_goal_real.png)

# Findings:
-----------------------------------------------------------------------
- From this graph we can see that there is a sharp cut off at around 8000 dollars when projects go from having the posibility of being successful to not.
- I also saw a significant spike in projects that are success full in the 5000 dollar range.
- The majority of projects that are successful percentage wise are those between 1000 and 3000 dollar range.     
-----------------------------------------------------------------------
    
# <a name="top"></a> Count Backers - README.md
![Zillow Logo](https://github.com/ljackson707/kick_starter_project/raw/main/chart_images/count_backers.png)

# Findings:
-----------------------------------------------------------------------
- The majority of projects that hade 25 > greater backers where more successful.
-----------------------------------------------------------------------
    
# <a name="top"></a> Count Main Category - README.md
![Zillow Logo](https://github.com/ljackson707/kick_starter_project/raw/main/chart_images/count_main_category.png)

# Findings:
-----------------------------------------------------------------------
- Cat 1 - (film_video) is the most popular propject on the kickstarter website, Their success/fail ratio is average.
- Cat 2 - (Music) is the most successful and secound most popular catagory.
- Cat 5 - (Technology) is the least has the worst success to fail ratio.
- Cat 14 and Cat 15 - (Journalism and Dance) are the least popular categorys with Journalism having terible success/fail ratio. Dance has a good success/fail ratio but not as popular.
-----------------------------------------------------------------------

### Stats
    
- Stat Test backers and state_successful: 
    - **T-Testing**:
        - HO: There is no relationship between backers and state_successful
        - HA: There is a relationship between backers and state_successful
        - t-stat: 242.3599
        - p-value: 0.0
        - Result: Because the p-value: 0.0 is less than the alpha: 0.05, we can reject the null hypothesis
  
- Stats test usd_pledged_real and state_successful:
    - **T-Testing**:
        - HO: There is no relationship between usd_pledged_real and state_successful
        - HA: There is a relationship between usd_pledged_real and state_successful
        - t-stat: 237.4207
        - p-value: 0.0
        - Result: Because the p-value: 0.0 is less than the alpha: 0.05, we can reject the null hypothesis

- Stats test usd_goal_real and state_successful:
    - **T-Testing**:
        - HO: There is no relationship between usd_goal_real and state_successful
        - HA: There is a relationship between usd_goal_real and state_successful
        - t-stat: 290.9375
        - p-value: 0.0
        - Result: Because the p-value: 0.0 is less than the alpha: 0.05, we can reject the null hypothesis

### Modeling:


***

    
</details>

## <a name="dictionary"></a>Data Dictionary  
[[Back to top](#top)]

<details>
  <summary>Click to expand!</summary>

### Data Used
    
| Attribute | Definition | Data Type |
| ----- | ----- | ----- |
| main_category | Main category of project | Object |
| cat_value | Value associated with main category of project | int64 |
| currency | Type of currency used to fund project | Object |
| deadline | Project deadline date | Object |
| launched | Project launch data | Object |
| state* | Current state of project (Fail, Success| Object |
| backers| Total number of backers | int64 |
| country | project origin country | Object |
| usd_pledged_real | Real value of USD pledged for project | float64 |
| usd_goal_real | Real project goal in USD | float64 |

\*  Indicates the target feature in the kickstarter dataset.
    
### Values associated to main_category (main_category_val)
| Category | Value | Data Type |
| ----- | ----- | ----- |
| Film_Video | 1 | int64 |
| Music | 2 | int64 |
| Publishing | 3 | int64 |
| Games | 4 | int64 |
| Technology | 5 | int64 |
| Art | 6 | int64 |
| Design | 7 | int64 |
| Food | 8 | int64 |
| Fashion | 9 | int64 |
| Theater | 10 | int64 |
| Comics | 11 | int64 |
| Photography | 12 | int64 |
| Crafts | 13 | int64 |
| Journalism | 14 | int64 |
| Dance | 15 | int64 |

### Values associated to currency (currency_type_val)
| Currency | Currency Name | Value | Data Type |
| ----- | ----- | ----- |
| USD | US Dollar| 1 | int64 |
| GBP | Pound | 2 | int64 |
| EUR | Euro | 3 | int64 |
| CAD | Canadian Dollar | 4 | int64 |
| AUD | Australian Dollar | 5 | int64 |
| SEK | Swedish Krona | 6 | int64 |
| MXN | Mexican Peso | 7 | int64 |
| NZD | New Zealand Dollar | 8 | int64 |
| DKK | Danish Krone | 9 | int64 |
| CHF | Swiss Franc | 10 | int64 |
| NOK | Norwegian Krone | 11 | int64 |
| HKD | Hong Kong Dollar | 12 | int64 |
| SGD | Singapore Dollar | 13 | int64 |
| JPY | Japanese Yen | 14 | int64 |
    
### Values associated to country (country_name_val)
| Category | Value | Data Type |
| ----- | ----- | ----- |
| Film_Video | 1 | int64 |
| Music | 2 | int64 |
| Publishing | 3 | int64 |
| Games | 4 | int64 |
| Technology | 5 | int64 |
| Art | 6 | int64 |
| Design | 7 | int64 |
| Food | 8 | int64 |
| Fashion | 9 | int64 |
| Theater | 10 | int64 |
| Comics | 11 | int64 |
| Photography | 12 | int64 |
| Crafts | 13 | int64 |
| Journalism | 14 | int64 |
| Dance | 15 | int64 |
***
</details>

## <a name="acquire_and_prep"></a>Acquire & Prep:
[[Back to top](#top)]

<details>
  <summary>Click to expand!</summary>

### Acquire Data:
- Gather Kick Starter project data from Kaggle online database.
    - Code to do this can be found in the wrangle.py file under the `get_kickstarter_data()` function

### Prepare Data
- To clean the data I had to:
    - Drop NULL values
    - Encode features
    - Create new features
    - Drop features
    - Rename features
    - Turn some features into binary features
    - Change some features to int64
    - Handle Outliers
- From here we :
    - Split the data into train, validate, and test
    - Split train, validate, and test into X and y
    - Scaled the data

​
| Function Name | Purpose |
| ----- | ----- |
| acquire_functions | DOCSTRING | 
| prepare_functions | DOCSTRING | 
| wrangle_functions | DOCSTRING |
​
***
​

    
</details>



## <a name="explore"></a>Data Exploration:
[[Back to top](#top)]

<details>
  <summary>Click to expand!</summary>
    
- wrangle.py 

### Findings:
- 
    
    
| Function Name | Definition |
| ------------ | ------------- |
### For Wrangle:
| ------------ | ------------- |
- | missing_zero_values_table | This function will look at any data set and report back on zeros and nulls for every column while also giving percentages of total values and also the data types. The message prints out the shape of the data frame and also tells you how many columns have nulls |
- | handle_missing_values | This function drops all null values within a column and row. The Threshold is determined by prop_required_column and prop_required_row arguments. |
- | remove_columns | This function removes unwanted columns from df, returns new df |
- | data_prep | combines the remove_columns and handle_missing_values functions |
- | train_validate_test_split | This function takes in a dataframe, the target feature as a string, and a seed interger and returns split data: train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test |
- | scale_my_data(train, validate, test) | This function takes in 3 dataframes with the same columns, and fits a min-max scaler to the first dataframe and transforms all 3 dataframes using that scaler. It returns 3 dataframes with the same column names and scaled values. 
- | get_dummies | This fucntion takes in a dataframe and dummifies specific coloumns and returns a df. |
- | turn_to_time | This fucntion takes in a dataframe and changes specified columns to datetime formate for easily manipulation, THen it resets the index and joins new datetime onto original df while droping old datetime, returning a dataframe. |
- | handle_outliers_backers | This fucntion takes in a dataframe and handles any outliers for the specified column using the IQR rule. |
- | handle_outliers_usd_pledged_real | This fucntion takes in a dataframe and handles any outliers for the specified column using the IQR rule. |
- | handle_outliers_usd_goal_real | This fucntion takes in a dataframe and handles any outliers for the specified column using the IQR rule. 
    
| ------------ | ------------- |
### For Explore:
| ------------ | ------------- |
- | explore_univariate_categorical | Takes in a dataframe and a categorical variable and returns a frequency table and barplot of the frequencies. |
- | explore_univariate_quant | takes in a dataframe and a quantitative variable and returns descriptive stats table, histogram, and boxplot of the distributions. |
- | freq_table | for a given categorical variable, compute the frequency count and percent split and return a dataframe of those values along with the different values of column. |
- | explore_bivariate_categorical | takes in categorical variable and binary target variable, returns a crosstab of frequencies, runs a chi-square test for the proportions, and creates a barplot, adding a horizontal line of the overall rate of the target. |
- | explore_bivariate_quant | descriptive stats by each target class. compare means across 2 target groups boxenplot of target x quant swarmplot of target x quant |
- | plot_swarm | Takes in train and target with quant vars and plots swarm plot |
- | plot_boxen | Takes in train and target with quant vars and plots boxen plot |
- | plot_all_continuous_vars |  Melt the dataset to "long-form" representation boxenplot of measurement x value with color representing target. |

| ------------ | ------------- |
### For Stats:
| ------------ | ------------- |
- | t_test | This function takes in 2 populations, and an alpha confidence level and outputs the results of a t-test. 

    # Parameters:
    - population_1: A series that is a subgroup of the total population. 
    - population_2: When sample = 1, population_2 must be a series that is the total population; when sample = 2,  population_2 can be another subgroup of the same population
    - alpha: alpha = 1 - confidence level, 
    - sample: {1 or 2}, default = 1, functions performs 1 or 2 sample t-test.
    - tail: {1 or 2}, default = 2, Need to be used in conjuction with tail_dir. performs a 1 or 2 sample t-test. 
    - tail_dir: {'higher' or 'lower'}, defaul = higher. |
    | chi2 | This function takes in a df, variable, a target variable, and the alpha, and runs a chi squared test. Statistical analysis is printed in the output. |  

| ------------ | ------------- |
### For Modeling:
| ------------ | ------------- |
- | logit_model | This function takes in train, y_train, validate, and y_validate, fits train and y_train on the logit model, predicts on train, and views raw probabilities (output from the model) (gives proabilities for each observation), while also producing a datframe with each features log_coeffs |
- | knn_model | This function takes in X_train, y_train, X_validate, y_validate, X_test, y_test, fits train and y_train on the knn model, predicts, and uses the model on the validate/out of sample data. Lastly this function produces a vizualization that shows the accuracy in realtion to the level of k|
- | bootstrap_model | This function takes in X_train, y_train, X_validate, y_validate, fits train and y_train on the RandomForrestClassifier model, fits X_train and y_train on the rf model, predicts on train, and views raw probabilities (output from the model) (gives proabilities for each observation), then evaluates model on X_validate sample and gernates a confusion matrix.  |

</details>    

## <a name="stats"></a>Statistical Analysis
[[Back to top](#top)]
<details>
  <summary>Click to expand!</summary>


### Stats Test 1:
- What is the test?
  - T-test  
- Why use this test?
  - I want to look at two populations of data with a confidence level of 95  
- What is being compared?
  - backers and state_successful

#### Hypothesis:
- The null hypothesis (H<sub>0</sub>) is...
  - HO: There is no relationship between backers and state_successful
- The alternate hypothesis (H<sub>1</sub>) is ...
  - HA: There is a relationship between backers and state_successful

#### Confidence level and alpha value:
- I established a 95% confidence level
- alpha = 1 - confidence, therefore alpha is 0.05

#### Results:
- Because the p-value: 0.0 is less than the alpha: 0.05, we can reject the null hypothesis

- Summary:
    - t-stat score of:
        - 242.3599
    - P vlaue of:
        - 0.0

### Stats Test 2: 
- What is the test?
  - T-test  
- Why use this test?
  - I want to look at two populations of data with a confidence level of 95  
- What is being compared?
  - usd_pledged_real and state_successful

#### Hypothesis:
- The null hypothesis (H<sub>0</sub>) is...
  - HO: There is no relationship between usd_pledged_real and state_successful
- The alternate hypothesis (H<sub>1</sub>) is ...
  - HA: There is a relationship between usd_pledged_real and state_successful

#### Confidence level and alpha value:
- I established a 95% confidence level
- alpha = 1 - confidence, therefore alpha is 0.05

#### Results:
- Because the p-value: 0.0 is less than the alpha: 0.05, we can reject the null hypothesis
    
- Summary:
    - t-stat score of:
        - 237.4207
    - P vlaue of:
        - 0.0
 
### Stats Test 3:
- What is the test?
  - T-test
- Why use this test?
  - I want to look at two populations of data with a confidence level of 95  
- What is being compared?
  - usd_goal_real and state_successful

#### Hypothesis:
- The null hypothesis (H<sub>0</sub>) is...
  - HO: There is no relationship between usd_goal_real and state_successful
- The alternate hypothesis (H<sub>1</sub>) is ...
  - HA: There is a relationship between usd_goal_real and state_successful

#### Confidence level and alpha value:
- I established a 95% confidence level
- alpha = 1 - confidence, therefore alpha is 0.05

#### Results:
- Because the p-value: 0.0 is less than the alpha: 0.05, we can reject the null hypothesis

- Summary:
    - t-stat score of:
        - 290.9375
    - P vlaue of:
        - 0.0


</details>    

## <a name="model"></a>Modeling:
[[Back to top](#top)]
<details>
  <summary>Click to expand!</summary>

Summary of modeling choices...

### Baseline

- Baseline Results: 
    - Mode In sample = 0
    - Mode Out of sample = 0
        
### Models and R<sup>2</sup> Values:
- Will run the following models:
    - logit
    - KNN
    - bootstrap (RandomForestClassifier)

### Using logit model:
    
Precision: 1.00    
Recal: 1.00 
F1-score: 1.00
    

### Using KNN model:

Accuracy in sample: 1.00
Precision: 1.00      
Recal: 1.00   
F1-score: 1.00 

Accuracy out of sampele: 1.00
Precision:1.00     
Recal: 1.00    
F1-score: 1.00

# <a name="top"></a> KNN Model - README.md
![Zillow Logo](https://github.com/ljackson707/kick_starter_project/raw/main/chart_images/knn_model.png)
   
    
### Using bootstrap (RandomForestClassifier)

Accuracy in sample: 1.00
Precision: 1.00      
Recal: 1.00   
F1-score: 1.00 

Accuracy out of sampele: 1.00
Precision:1.00     
Recal: 1.00    
F1-score: 1.00
    
#### Findings
- Models used with binary target = (logit, KNN, bootstrap models)
- All three models hade an accuracy of 1.00. 
- This is highly unlikly to be true, If i get more time I would like to look into why my accuracy is so high and the features it is testing on.
    

***

</details>  

## <a name="conclusion"></a>Conclusion:
[[Back to top](#top)]
<details>
  <summary>Click to expand!</summary>

- In conclusion I have deatemrined that there are corrlations between the features backers, usd_pledged_real, usd_goal_real with the target of state_successful.

- I found that there was a negative correlation to the amount usd_goal_real and state_successful. From this I discovered that there was a 'sweet spot' for goals around 5000 dollars.

- I also found that the majority of projects that had 25 or greater backers where more successful.

- The most successful and most popular categories werre film and music.

- I was able to reject each of my null hypothesises solidifying the fact that there is a relationship between these features.

- Lastly the modeling came out not as I would have thought. I did not expect to have an accuracy of 1.00 which my baseline model accuary was equal to 0. (This is because i was using a binary target variable.

### If I Had More Time
- I would have looked into using a continious target varible such as backers or usd_pledged_real. Rather than trying to predict projects who will fail or not id like to look at what factors affect the number of backers and the pledge amounts, knowiung that these two features have a relationship with state_sucessful. With this I would be able to utilize stronger models such as the OLS, Tweedie, or LassoLars and find the best one inorder to reproduce these predictions on future kickstarter projects.

</details>  

![Folder Contents](https://github.com/ljackson707/kick_starter_project/raw/main/chart_images/kickstarter_logo.png)


>>>>>>>>>>>>>>>
.
