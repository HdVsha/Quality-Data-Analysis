#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox


def IMR(original_df, col_name):
    """Implements the Individual Moving Range (IMR) chart.
    Parameters
    ----------
    original_df : pandas.DataFrame
        A DataFrame containing the data to be plotted.
    col_name : str
        The name of the column to be used for the IMR control chart.
    Returns
    -------
    chart : matplotlib.axes._subplots.AxesSubplot
        The IMR chart.

    df_IMR : pandas.DataFrame with the following additional columns
        - MR: The moving range
        - I_UCL: The upper control limit for the individual
        - I_CL: The center line for the individual
        - I_LCL: The lower control limit for the individual
        - MR_UCL: The upper control limit for the moving range
        - MR_CL: The center line for the moving range
        - MR_LCL: The lower control limit for the moving range
        - I_TEST1: The points that violate the alarm rule for the individual
        - MR_TEST1: The points that violate the alarm rule for the moving range
    """
    # Check if df is a pandas DataFrame
    if not isinstance(original_df, pd.DataFrame):
        raise TypeError('The data must be a pandas DataFrame.')

    # Check if the col_name exists in the DataFrame
    if col_name not in original_df.columns:
        raise ValueError('The column name does not exist in the DataFrame.')

    d2 = 1.128
    D4 = 3.267

    # Create a copy of the original DataFrame
    df = original_df.copy()
    
    # Calculate the moving range
    df['MR'] = df[col_name].diff().abs()
    # Create columns for the upper and lower control limits
    df['I_UCL'] = df[col_name].mean() + (3*df['MR'].mean()/d2)
    df['I_CL'] = df[col_name].mean()
    df['I_LCL'] = df[col_name].mean() - (3*df['MR'].mean()/d2)

    # Define columns for the Western Electric alarm rules
    df['I_TEST1'] = np.where((df[col_name] > df['I_UCL']) | (df[col_name] < df['I_LCL']), df[col_name], np.nan)

    # Plot the I and MR charts
    fig, ax = plt.subplots(2, 1)
    fig.suptitle(('I-MR charts of %s' % col_name))
    ax[0].plot(df[col_name], color='mediumblue', linestyle='--', marker='o')
    ax[0].plot(df['I_UCL'], color='firebrick', linewidth=1)
    ax[0].plot(df['I_CL'], color='g', linewidth=1)
    ax[0].plot(df['I_LCL'], color='firebrick', linewidth=1)
    ax[0].set_ylabel('Individual Value')
    ax[1].set_xlabel('Sample number')
    # add the values of the control limits on the right side of the plot
    ax[0].text(len(df)+.5, df['I_UCL'].iloc[0], 'UCL = {:.2f}'.format(df['I_UCL'].iloc[0]), verticalalignment='center')
    ax[0].text(len(df)+.5, df['I_CL'].iloc[0], 'CL = {:.2f}'.format(df['I_CL'].iloc[0]), verticalalignment='center')
    ax[0].text(len(df)+.5, df['I_LCL'].iloc[0], 'LCL = {:.2f}'.format(df['I_LCL'].iloc[0]), verticalalignment='center')
    # highlight the points that violate the alarm rules
    ax[0].plot(df['I_TEST1'], linestyle='none', marker='s', color='firebrick', markersize=10)

    # Create columns for the upper and lower control limits
    df['MR_UCL'] = D4 * df['MR'].mean()
    df['MR_CL'] = df['MR'].mean()
    df['MR_LCL'] = 0
    # Define columns for the Western Electric alarm rules
    df['MR_TEST1'] = np.where((df['MR'] > df['MR_UCL']) | (df['MR'] < df['MR_LCL']), df['MR'], np.nan)

    ax[1].plot(df['MR'], color='mediumblue', linestyle='--', marker='o')
    ax[1].plot(df['MR_UCL'], color='firebrick', linewidth=1)
    ax[1].plot(df['MR_CL'], color='g', linewidth=1)
    ax[1].plot(df['MR_LCL'], color='firebrick', linewidth=1)
    ax[1].set_ylabel('Moving Range')
    ax[1].set_xlabel('Sample number')
    # add the values of the control limits on the right side of the plot
    ax[1].text(len(df)+.5, df['MR_UCL'].iloc[0], 'UCL = {:.2f}'.format(df['MR_UCL'].iloc[0]), verticalalignment='center')
    ax[1].text(len(df)+.5, df['MR_CL'].iloc[0], 'CL = {:.2f}'.format(df['MR_CL'].iloc[0]), verticalalignment='center')
    ax[1].text(len(df)+.5, df['MR_LCL'].iloc[0], 'LCL = {:.2f}'.format(df['MR_LCL'].iloc[0]), verticalalignment='center')
    # highlight the points that violate the alarm rules
    ax[1].plot(df['MR_TEST1'], linestyle='none', marker='s', color='firebrick', markersize=10)


    plt.show()

    return df

import scipy.integrate as spi
import numpy as np

import numpy as np
import scipy.integrate as spi

class constants:
    @staticmethod
    def getd2(n=None):
        if n is None or n < 2 or abs(n - round(n)) != 0:
            raise ValueError("Invalid sample size ({})".format(n))
        def f(x):
            return stats.studentized_range.sf(x, n, np.inf)
        d2, _ = spi.quad(f, 0, np.inf)
        if _ > 0.001:
            print("The absolute error after numerical integration is greater than 0.001")
        return d2

def summary(results):
    """Prints a summary of the regression results.

    Parameters
    ----------
    results : RegressionResults object
        The results of a regression model.

    Returns
    -------
    None
    """

    # Set the precision of the output
    np.set_printoptions(precision=4, suppress=True)
    pd.options.display.precision = 4

    # Extract information from the result object
    terms = results.model.exog_names
    coefficients = results.params
    std_errors = results.bse
    t_values = results.tvalues
    p_values = results.pvalues
    r_squared = results.rsquared
    adjusted_r_squared = results.rsquared_adj

    # Print the regression equation
    print("REGRESSION EQUATION")
    print("-------------------")
    equation = ("%s = " % results.model.endog_names)
    for i in range(len(coefficients)):
        if results.model.exog_names[i] == 'Intercept':
            equation += "%.3f" % coefficients[i]
        else:
            if coefficients[i] > 0:
                equation += " + %.3f %s" % (coefficients[i], terms[i])
            else:
                equation += " %.3f %s" % (coefficients[i], terms[i])
    print(equation)

    # Print the information in a similar format to Minitab
    print("\nCOEFFICIENTS")
    print("------------")
    # make a dataframe to store the results
    
    df_coefficients = pd.DataFrame({'Term': terms, 'Coef': coefficients, 'SE Coef': std_errors, 'T-Value': t_values, 'P-Value': p_values})
    df_coefficients.style.format({'Coef': '{:.3f}', 'SE Coef': '{:.3f}', 'T-Value': '{:.3f}', 'P-Value': '{:.3f}'})
    print(df_coefficients.to_string(index=False))

    # Print the R-squared and adjusted R-squared
    print("\nMODEL SUMMARY")
    print("-------------")
    # compute the standard deviation of the distance between the data values and the fitted values
    S = np.std(results.resid, ddof=len(terms))
    # make a dataframe to store the results
    df_model_summary = pd.DataFrame({'S': [S], 'R-sq': [r_squared], 'R-sq(adj)': [adjusted_r_squared]})
    print(df_model_summary.to_string(index=False))

    # Print the ANOVA table
    print("\nANALYSIS OF VARIANCE")
    print("---------------------")
    # make a dataframe with the column names and no data
    df_anova = pd.DataFrame(columns=['Source', 'DF', 'Adj SS', 'Adj MS', 'F-Value', 'P-Value'])
    # add the rows of data
    df_anova.loc[0] = ['Regression', results.df_model, results.mse_model * results.df_model, results.mse_model, results.fvalue, results.f_pvalue]
    jj = 1
    for term in terms:
        if term != 'Intercept':
            # perform the f-test for the term
            f_test = results.f_test(term + '= 0')
            df_anova.loc[jj] = [term, f_test.df_num, f_test.fvalue * results.mse_resid * f_test.df_num, f_test.fvalue * results.mse_resid, f_test.fvalue, f_test.pvalue]
            jj += 1

    df_anova.loc[jj] = ['Error', results.df_resid, results.mse_resid * results.df_resid, results.mse_resid, np.nan, np.nan]

    '''
    # Lack-of-fit
    # compute the number of levels in the independent variables 
    n_levels = results.df_resid
    for term in terms:
        if term != 'Intercept':
            n_levels = np.minimum(n_levels, len(data[term].unique())

    if n_levels < results.df_resid:
        dof_lof = n_levels - len(terms)
        dof_pe = results.df_resid - n_levels
        # compute the SSE for the pure error term
        for 


        df_anova.loc[jj + 1] = ['Lack-of-fit', n_levels - len(terms), np.nan, np.nan, np.nan, np.nan]
    '''

    df_anova.loc[jj + 1] = ['Total', results.df_model + results.df_resid, results.mse_total * (results.df_model + results.df_resid), np.nan, np.nan, np.nan]

    print(df_anova.to_string(index=False))



