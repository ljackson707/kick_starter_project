# kick_starter_project
This is a repo for my individual project over the Kick Starter dataset. I will be creating a classification algorithm predicting whether or not a Kick-Starter Project becomes successful based on funding period, country, main category, and pledge amount.  Link to data set : https://www.kaggle.com/kemical/kickstarter-projects

Kick Starter Project

# <a name="top"></a> Kick Starter Project - README.md
![Zillow Logo](https://github.com/ljackson707/kick_starter_project/raw/main/Screen%20Shot%202021-04-09%20at%2011.42.24%20AM.png)
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
- We learned:

### Stats
- Stat Test: 
    - **Anova Test**:
        - Showed that there was a difference between log error of at least one of the n cluster created.
    - **T-Testing**:
        - Showed 
- Stats test:
    - **Anova Test**:
        - Shows that there is a difference between the log error of at least one of the n clusters created.
    - **T-Testing**:
        - Showed 
- Stats test: Quality,
    - **Anova Test**:
        - Shows that there is a difference between the log error of at least one of the n clusters created.
    - **T-Testing**:
        - Showed

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
    -  
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
| missing_zero_values_table | This function will look at any data set and report back on zeros and nulls for every column while also giving percentages of total values and also the data types. The message prints out the shape of the data frame and also tells you how many columns have nulls |
| handle_missing_values | This function drops all null values within a column and row. The Threshold is determined by prop_required_column and prop_required_row arguments. |
| rename_columns | This fucntion renames specific columns with  a specified name. |
| remove_columns | This function removes unwanted columns from df, returns new df |
| data_prep | combines the remove_columns and handle_missing_values functions |
| train_validate_test_split | This function takes in a dataframe, the target feature as a string, and a seed interger and returns split data: train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test |
| scale_my_data(train, validate, test) | This function takes in 3 dataframes with the same columns, and fits a min-max scaler to the first dataframe and transforms all 3 dataframes using that scaler. It returns 3 dataframes with the same column names and scaled values. 
| ------------ | ------------- |
### For Explore:
| ------------ | ------------- |
| explore_univariate_categorical | Takes in a dataframe and a categorical variable and returns a frequency table and barplot of the frequencies. |
| explore_univariate_quant | takes in a dataframe and a quantitative variable and returns descriptive stats table, histogram, and boxplot of the distributions. |
| freq_table | for a given categorical variable, compute the frequency count and percent split and return a dataframe of those values along with the different values of column. |
| explore_bivariate_categorical | takes in categorical variable and binary target variable, returns a crosstab of frequencies, runs a chi-square test for the proportions, and creates a barplot, adding a horizontal line of the overall rate of the target. |
| explore_bivariate_quant | descriptive stats by each target class. compare means across 2 target groups boxenplot of target x quant swarmplot of target x quant |
| plot_swarm | Takes in train and target with quant vars and plots swarm plot |
| plot_boxen | Takes in train and target with quant vars and plots boxen plot |
| plot_all_continuous_vars |  Melt the dataset to "long-form" representation boxenplot of measurement x value with color representing target. |
| ------------ | ------------- |
### For Stats:
| ------------ | ------------- |
| t_test | This function takes in 2 populations, and an alpha confidence level and outputs the results of a t-test. 

Parameters:
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
| viz_kickstarter | Takes in train and kmeans and produces centroids for each cluster, plots multiple scatter subplots shwoing each defined cluster |
| OLS_Model | Takes in X_train, y_train, X_validate, y_validate, then produces the RMSE for OLS using in sample and oiut of sample data. |
| ols_actual_vs_predicted | Takes in X_train, y_train, X_validate, y_validate then produces a bar plot that shows the actual and predicted distribution of target variable to determine which model is more accurate. Model to feed in = (OLS, LassoLars, Tweedie, Ploynomial 1 and 2, etc.) |
| OLS_hist | This function takes in X_train, y_train, X_validate, y_validate and returns subplots that compare the accuracy of each model .|
    
</details>    

## <a name="stats"></a>Statistical Analysis
[[Back to top](#top)]
<details>
  <summary>Click to expand!</summary>


### Stats Test 1:
- What is the test?
    
- Why use this test?
    
- What is being compared?
    
#### Hypothesis:
- The null hypothesis (H<sub>0</sub>) is...
   
- The alternate hypothesis (H<sub>1</sub>) is ...

#### Confidence level and alpha value:
- I established a 95% confidence level
- alpha = 1 - confidence, therefore alpha is 0.05

#### Results:
- 

- Summary:
    - F score of:
        - 
    - P vlaue of:
        - 

### Stats Test 2: 
- What is the test?
    
- Why use this test?

- What is being compared?
    
#### Hypothesis:
- The null hypothesis (H<sub>0</sub>) is...
   
- The alternate hypothesis (H<sub>1</sub>) is ...

#### Confidence level and alpha value:
- I established a 95% confidence level
- alpha = 1 - confidence, therefore alpha is 0.05

#### Results:
-
    
- Summary:
    - F score of:
        - 
    - P vlaue of:
        - 
 
### Stats Test 3:
- What is the test?

- Why use this test?

- What is being compared?

#### Hypothesis:
- The null hypothesis (H<sub>0</sub>) is...
   
- The alternate hypothesis (H<sub>1</sub>) is ...
   
#### Confidence level and alpha value:
- I established a 95% confidence level
- alpha = 1 - confidence, therefore alpha is 0.05

#### Results:
- 

- Summary:
    - F score of:
        - 5.3376
    - P vlaue of:
        - 6.587e-05

### Stats Test 4: 
- What is the test?
    - T Test
- Why use this test?
    - To find statistical differences between the means of 2 or more clusters
- What is being compared?
    - Winning cluster of taxes

#### Results:
 - Homes with low to medium structure tax value and low land tax value affect logerror to some degree.
    

### Stats Test 5:
- What is the test?
    - Anova
- Why use this test?
    - Find out if a cluster has significance to the logerror
- What is being compared?
    - Latitude, Longitude, and House age

#### Hypothesis:
- The null hypothesis (H<sub>0</sub>) is...
    - There is no difference between the log error means of each individual cluster
- The alternate hypothesis (H<sub>1</sub>) is ...
    - There is a difference between the log error means of at least one clusters.


#### Confidence level and alpha value:
- I established a 95% confidence level
- alpha = 1 - confidence, therefore alpha is 0.05

#### Results:
- Reject the null

- Summary:
    - F score of:
        - 
    - P vlaue of:
        - 
    
</details>    

## <a name="model"></a>Modeling:
[[Back to top](#top)]
<details>
  <summary>Click to expand!</summary>

Summary of modeling choices...

### Baseline

- Baseline Results: 
    - Median In sample = 
    - Median Out of sample = 
        
### Models and R<sup>2</sup> Values:
- Will run the following models:
    - Linear regression OLS Model
    - Lasso Lars
    - Tweedie Regressor
    - Polynomail Degree 2
    - Ploynomial Degree 3

- Other indicators of model performance
    - R<sup>2</sup> Baseline Value
        - 
    - R<sup>2</sup> OLS Value 
        - 



### RMSE using Mean
    
Train/In-Sample:  
    
Validate/Out-of-Sample:  
    

### RMSE using Median
Train/In-Sample:  
Validate/Out-of-Sample: 

### RMSE for OLS using LinearRegression
    
Training/In-Sample: 
    
Validation/Out-of-Sample: 
    

### RMSE for Lasso + Lars
    
Training/In-Sample:
    
Validation/Out-of-Sample: 
    

    
### RMSE for GLM using Tweedie, power=0 and alpha=0
    
Training/In-Sample: 
    
Validation/Out-of-Sample:
    

    
### RMSE for Polynomial Model, degrees=2
    
Training/In-Sample:
    
Validation/Out-of-Sample:
    

    
### RMSE for Polynomial Model, degrees=3
    
Training/In-Sample: 
    
Validation/Out-of-Sample:


### Eetc:

## Selecting the Best Model:

### Use Table below as a template for all Modeling results for easy comparison:

| Model | Training/In Sample RMSE | Validation/Out of Sample RMSE | R<sup>2</sup> Value |
| ---- | ----| ---- | ---- |
| Baseline | | | |
| Linear Regression | | | |
| Tweedie Regressor (GLM) | | | n/a |
| Lasso Lars | | | n/a |
| Polynomial Regression D2| | | n/a |
| Polynomial Regression D3| | | n/a |

- Why did you choose this model?
    - It was closer to 0 than our baseline.

## Testing the Model

- Model Testing Results
     - Out-of-Sample Performance:


***

</details>  

## <a name="conclusion"></a>Conclusion:
[[Back to top](#top)]
<details>
  <summary>Click to expand!</summary>

We found that only about 9.36% of log error was inaccurate. Meaning that it was below -0.15 or above 0.15 rendering it inaccurate.

This gave us a small amount to work with. But in the end we were able to create a model to find certain drivers of the inaccurate log error.
Our model performed better than the baseline by a decent amount. With a R baseline of ~-0.0046 and our model performing at ~0.000052. Meaning we were able to get closer to 0 than our baseline.

We found that Ventura, north downtown LA, tax values, home quality, and a homes age affect loerror within their resepective cluster.

With further time we would like to look further into geographical location and tax values to see if there is a more specific reason for log error.

We recommend using our OLS model to be used within the field, in order to establish a closer zestimate score to what the selling price may be, in order to service our custoemrs even better.


    

</details>  

![Folder Contents](https://github.com/ljackson707/kick_starter_project/raw/main/Screen%20Shot%202021-04-09%20at%2011.42.24%20AM.png)


>>>>>>>>>>>>>>>
.
