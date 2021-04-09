import numpy as np 
from numpy import stats


def t_test( population_1, population_2, alpha, sample=1, tail=2, tail_dir="higher"):
    '''
    This function takes in 2 populations, and an alpha confidence level and outputs the results of a t-test. 
    Parameters:
        population_1: A series that is a subgroup of the total population. 
        population_2: When sample = 1, population_2 must be a series that is the total population; when sample = 2,  population_2 can be another subgroup of the same population
        alpha: alpha = 1 - confidence level, 
        sample: {1 or 2}, default = 1, functions performs 1 or 2 sample t-test.
        tail: {1 or 2}, default = 2, Need to be used in conjuction with tail_dir. performs a 1 or 2 sample t-test. 
        tail_dir: {'higher' or 'lower'}, defaul = higher.
    '''
    if sample==1 and tail == 2:
        t, p = stats.ttest_1samp(population_1, population_2.mean())
        print(f't-stat = {round(t,4)}')
        print(f'p     = {round(p,4)}\n')
        if p < alpha:
            print(f'Because the p-value: {round(p, 4)} is less than the alpha: {alpha}, we can reject the null hypothesis')
        else:
            print('There is insufficient evidence to reject the null hypothesis')
    elif sample==1 and tail == 1:
        t, p = stats.ttest_1samp(population_1, population_2.mean())
        print(f't-stat = {round(t,4)}')
        print(f'p     = {round(p,4)}\n')
        if tail_dir == "higher":
            if (p/2) < alpha and t > 0:
                print(f'Because the p-value: {round(p, 4)} is less than the alpha: {alpha}, and the t-stat: {round(t,4)} is greater than 0, we can reject the null hypothesis')
            else:
                print('There is insufficient evidence to reject the null hypothesis')
        elif tail_dir == "lower":
            if (p/2) < alpha and t < 0:
                print(f'Because the p-value: {round(p, 4)} is less than the alpha: {alpha}, and the t-stat: {round(t,4)} is less than 0, we can reject the null hypothesis')
            else:
                print('There is insufficient evidence to reject the null hypothesis')
    elif sample==2 and tail == 2:
        t, p = stats.ttest_ind(population_1, population_2)
        print(f't-stat = {round(t,4)}')
        print(f'p     = {round(p,4)}\n')
        if p < alpha:
            print(f'Because the p-value: {round(p, 4)} is less than the alpha: {alpha}, we reject the null hypothesis')
        else:
            print('There is insufficient evidence to reject the null hypothesis')
    elif sample == 2 and tail == 1:
        t, p = stats.ttest_ind(population_1, population_2)
        print(f't-stat = {round(t,4)}')
        print(f'p     = {round(p,4)}\n')
        if tail_dir == "higher":
            if (p/2) < alpha and t > 0:
                print(f'Because the p-value: {round(p, 4)} is less than alpha: {alpha}, and t-stat: {round(t,4)} is greater than 0, we reject the null hypothesis')
            else:
                print('There is insufficient evidence to reject the null hypothesis')
        elif tail_dir == "lower":
            if (p/2) < alpha and t < 0:
                print(f'Because the p-value: {round(p, 4)} is less than alpha: {alpha} and the t-stat: {round(t,4)} is less than 0, we reject the null hypothesis')
            else:
                print('There is insufficient evidence to reject the null hypothesis')
    else:
        print('sample must be 1 or 2, tail must be 1 or 2, tail_dir must be "higher" or "lower"')


        
def chi2(df, var, target, alpha):
    '''
    This function takes in a df, variable, a target variable, and the alpha, and runs a chi squared test. Statistical analysis is printed in the output.
    '''
    observed = pd.crosstab(df[var], df[target])
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    print('Observed\n')
    print(observed.values)
    print('---\nExpected\n')
    print(expected)
    print('---\n')
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}\n')
    if p < alpha:
        print(f'Becasue the p-value: {round(p, 4)} is less than alpha: {alpha}, we can reject the null hypothesis')
    else:
        print('There is insufficient evidence to reject the null hypothesis')