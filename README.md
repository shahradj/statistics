# linear_reg.py
A multivariate linear regression class built from first principles.

In R, the multivariate linear regression is an incredibly simple function, where one can also see the significance of each variable within the model.

The class here applies the same linear regression and outputs the same results including:
- the estimated coefficients (beta),
- the standard error (SE),
- the t-statistic for the SE (tstat), and
- the corresponding p-value(p_val) for each input variable.

To filter out the variables that are not contributing to the predictor variable, we can also apply the regression over every combination of input variables to output the models that only have significant input variables. This method should be used with caution as the quality of the model is not determined just by the significance of variables but also by the R squared and AIC/ BIC measures.

To achieve a quadratic regression, one would simply add the quadratic values of an input variable as an extra input variable, then run the linear regression on all variables. That is, instead of calculating coefficients for b_1 + b_2* X, we would calculate b_1 + b_2*X + b_3*X^2.
Given this ease of applying quadratic regression, I included the addition of such transforms into the class, so that new variables can be created to apply exponential, logarithmic, snusoidal, etc, regressions.

