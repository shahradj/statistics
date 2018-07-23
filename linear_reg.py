from scipy import stats
import numpy as np
from itertools import chain, combinations
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class LinearRegression:
    """exhaustive linear regression that finds models of only significant variables"""
    def __init__(self, df, input_vars, output_var, only_best=False):
        """

        :param df: pandas dataframe that has all the data of interest
        :param input_vars: array of strings that list the columns of the dataframe that are the dependent variables
        :param output_var: string thaat is the column of the dataframe that has the variable to predict
        """
        self.df = df
        assert all(var in df.columns for var in input_vars)
        assert output_var in df.columns
        if 'int' not in str(df[output_var].dtype) or 'float' not in str(df[output_var].dtype):
        	df.loc[:, output_var+'_num'] = LabelEncoder().fit_transform(df[output_var])
        	output_var = output_var + '_num'
        self.df.loc[:, 'Intercept'] = 1.
        self.input_vars = input_vars + ['Intercept']
        self.output_var = output_var
        self.n = len(self.df)
        
    def add_input_variables(self, new_input_variables):
        """adds new input variables, for example after adding transformed columns.

        :param new_input_variables: list of variables that should correspond to the columns in the dataframe
        """
        assert all(var in self.df.columns for var in new_input_variables)
        self.input_vars += new_input_variables
        self.input_vars = list(set(self.input_vars))
        
    def test_all_variable_combinations(self, only_best=True):
        """runs the linear regression on every combination of input variables

        :param only_best: boolean determines whether the analysis should only return the models of significance or all
        :returns: a dictionary with the variable combination as keys and the regression results as values
        """
        results = {}
        for n_variables in range(1, len(self.input_vars)+1):
            for variable_combo in combinations(self.input_vars, n_variables):
                res = self.fit(list(variable_combo))
                if only_best:
                    all_significant = all(x < 0.05 for x in res['coefficients']['p_val'].values)
                    all_non_zero = all(x > 1e-6 for x in res['coefficients']['beta'].values)
                    if all_significant and all_non_zero:
                        results[variable_combo] = res
                else:
                    results[variable_combo] = res
        return results

    def fit(self, input_variables):
        """performs the linear regression on the input_variables"""
        X = self.df[input_variables].values
        Y = self.df[self.output_var].values
        k = len(input_variables)
        
        Qxx = np.dot(X.T, X)
        preSE = np.linalg.inv(Qxx)
        beta = np.dot(np.dot(preSE, X.T), Y)
        hat = np.dot(X, np.dot(preSE, X.T))
        error = Y - np.dot(X, beta)
        ymean = np.mean(Y)
        total = Y - np.repeat(ymean, self.n)
        #return total,error
        s_squared = np.dot(error.T, error) / (self.n-k)
        SE = np.array([
            [np.sqrt(s_squared * preSE[i][i])]
             for i in range(k)
        ])
        L = np.diag(np.ones(self.n)) - np.ones((self.n, self.n))
        # M = np.diag(np.ones(n))-hat
        tstats = [beta[i] / SE[i] for i in range(len(beta))]
        
        deg_freedom = len(self.df) - k
        p_values = [
            [2 * stats.t.cdf(tstat, deg_freedom)[0]]
            if tstat<0 else
            [2 * stats.t.cdf(-tstat, deg_freedom)[0]]
            for tstat in tstats
        ]
        
        SSE = np.dot(error.T, error)
        SST = np.dot(total.T, total)

        TSS = np.dot(Y.T, np.dot(L, Y))

        r_squared = 1 - SSE/SST
        adj_r_sq = 1 - (self.n-1) / (self.n-k) * SSE / SST
        
        AIC = 2 * k - 2 * np.log(beta)
        AICc = AIC +2*(k+1)*(k+2)/(self.n-k-2)
        BIC = np.log(self.n)*k-2*np.log(beta)
        return {
            'coefficients':pd.DataFrame({
                'beta':beta,
                'SE':SE.T[0],
                'tstat':np.array(tstats).T[0],
                'p_val':np.array(p_values).T[0]
            }, input_variables),
            'adj_r_squared':adj_r_sq,
            'r_squared':r_squared,
            'AIC':AIC,
            'AICc':AICc,
            'BIC':BIC
        }

    def add_transforms(self, variables, function_names=['pow2']):
        """adds columns to the dataframe by applying functions to given variables
        
        :param variables_to_transform: array of strings of columns in the dataframe that want to be transformed
        :param function_names: array of strings of functions to apply to the variables
        :returns: the new column names of the transformed variables
        """
        assert all(var in self.df.columns for var in variables)
        functions = {
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan,
            'sqrt': np.sqrt,
            'log': np.log,
            'exp': np.exp,
            'pow2': lambda x:np.power(x, 2)
        }
        assert all(fn_name in functions for fn_name in function_names)
        new_variables = []
        for var in variables:
            for fn_name in function_names:
                new_column = '_'.join(var, fn_name)
                new_variables.append(new_column)
                self.df.loc[:, new_column] = df[var].apply(functions[fn_name])
        return new_variables

