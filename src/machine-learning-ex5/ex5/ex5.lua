---
--- Created by shuieryin.
--- DateTime: 24/12/2017 9:29 PM
---

require "../../../lib/util"

require "optim"
require "util"
require "ex5_func"

-- Machine Learning Online Class
--  Exercise 5 | Regularized Linear Regression and Bias-Variance
--  Instructions
--  ------------

--  This file contains code that helps you get started on the
--  exercise. You will need to complete the following functions:

require 'linearRegCostFunction'
require 'learningCurve'
require 'validationCurve'
require 'trainLinearReg'
require 'polyFeatures'
require 'featureNormalize'
require 'plotFit'

--  For this exercise, you will not need to change any code in this file,
--  or any other files other than those mentioned above.

-- =========== Part 1: Loading and Visualizing Data =============
--  We start the exercise by first loading and visualizing the dataset.
--  The following code will load the dataset into your environment and plot
--  the data.

-- Load Training Data
print('Loading and Visualizing Data ...')

-- Load from ex5data1:
-- You will have X, y, Xval, yval, Xtest, ytest in your environment
local X, y, Xval, yval, Xtest, ytest, plot = loadData()

-- m = Number of examples
local m = X:size(1)

pause()

-- =========== Part 2: Regularized Linear Regression Cost =============
--  You should now implement the cost function for regularized linear
--  regression.

local theta = torch.ones(2, 1)
local Xtrain = torch.ones(m, 1):cat(X, 2)
local J, grad = linearRegCostFunction(Xtrain, y, theta, 1)

print('Cost at theta = [1 ; 1] should be about 303.993192)')
print(J)

pause()

-- =========== Part 3: Regularized Linear Regression Gradient =============
--  You should now implement the gradient for regularized linear
--  regression.

print('Gradient at theta = [1 ; 1] should be about [-15.303016; 598.250744])')
print(grad)

pause()

-- =========== Part 4: Train Linear Regression =============
--  Once you have implemented the cost and gradient correctly, the
--  trainLinearReg function will use your cost function to train
--  regularized linear regression.
--
--  Write Up Note: The data is non-linear, so this will not give a great
--                 fit.

--  Train linear regression with lambda = 0
local lambda = 0
theta = trainLinearReg(Xtrain, y, lambda)

--  Plot fit over the data
local plot_x = torch.Tensor({ { X:min() - 2, X:max() + 8 } })
local plot_y = theta[2]:sum() * plot_x + theta[1]:sum()
plot:line(plot_x:totable(), plot_y:totable(), 'blue', 'Decision boundary'):legend(true):title('Line Plot Demo'):redraw()
itorchHtml(plot, 'decision_boundary.html')

-- =========== Part 5: Learning Curve for Linear Regression =============
--  Next, you should implement the learningCurve function.
--  Write Up Note: Since the model is underfitting the data, we expect to
--                 see a graph with "high bias" -- Figure 3 in ex5.pdf

lambda = 0
local error_train, error_val = learningCurve(torch.ones(m, 1):cat(X, 2), y, torch.ones(Xval:size(1), 1):cat(Xval, 2), yval, lambda)

-- plot error
plotTable({
    [1] = {
        data = error_train:totable(),
        desc = 'Train error',
        color = 'blue'
    },

    [2] = {
        data = error_val:totable(),
        desc = 'Cv error',
        color = 'green'
    }
}, 'Linear regression learning curve', 'Number of training examples', 'Error', 'errorHistory')

print('# Training Examples\tTrain Error\tCross Validation Error\n')
for i = 1, m do
    print(i, error_train[i]:sum(), error_val[i]:sum())
end

-- =========== Part 6: Feature Mapping for Polynomial Regression =============
--  One solution to this is to use polynomial regression. You should now
--  complete polyFeatures to map each example into its powers

local p = 8

-- Map X onto Polynomial Features and Normalize
local X_poly = polyFeatures(X, p)

local mu, sigma
X_poly, mu, sigma = featureNormalize(X_poly)  -- Normalize
X_poly = torch.ones(m, 1):cat(X_poly, 2)                   -- Add Ones

-- Map X_poly_test and normalize (using mu and sigma)
local X_poly_test = polyFeatures(Xtest, p)
X_poly_test = bsxfun(minus, X_poly_test, mu)
X_poly_test = bsxfun(rdivide, X_poly_test, sigma)
X_poly_test = torch.ones(X_poly_test:size(1), 1):cat(X_poly_test, 2)         -- Add Ones

-- Map X_poly_val and normalize (using mu and sigma)
local X_poly_val = polyFeatures(Xval, p)
X_poly_val = bsxfun(minus, X_poly_val, mu)
X_poly_val = bsxfun(rdivide, X_poly_val, sigma)
X_poly_val = torch.ones(X_poly_val:size(1), 1):cat(X_poly_val, 2)           -- Add Ones

print('Normalized Training Example 1:')
print(X_poly[1])

-- =========== Part 7: Learning Curve for Polynomial Regression =============
--  Now, you will get to experiment with polynomial regression with multiple
--  values of lambda. The code below runs polynomial regression with
--  lambda = 0. You should try running the code with different values of
--  lambda to see how the fit and learning curve change.

lambda = 0
theta = trainLinearReg(X_poly, y, lambda)

-- Plot training data and fit
-- figure 4
plotFit(X:min(), X:max(), mu, sigma, theta, p)

-- figure 5
error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, lambda)
plotTable({
    [1] = {
        data = error_train:totable(),
        desc = 'Train error',
        color = 'blue'
    },

    [2] = {
        data = error_val:totable(),
        desc = 'Cv error',
        color = 'green'
    }
}, 'Figure 5: Polynomial learning curve, λ = 0', 'Number of training examples', 'Error', 'figure5')

print('Polynomial Regression (lambda = ' .. lambda .. ')')
print('# Training Examples', 'Train Error', 'Cross Validation Error')
for i = 1, m do
    print( i, error_train[i][1], error_val[i][1])
end

pause()

-- =========== Part 8: Validation for Selecting Lambda =============
--  You will now implement validationCurve to test various values of
--  lambda on a validation set. You will then use this to select the
--  "best" lambda value.

local lambda_vec
lambda_vec, error_train, error_val = validationCurve(X_poly, y, X_poly_val, yval)

plotTable({
    [1] = {
        data = error_train:totable(),
        desc = 'Train error',
        color = 'blue'
    },

    [2] = {
        data = error_val:totable(),
        desc = 'Cv error',
        color = 'green'
    }
}, 'Figure 9: Selecting λ using a cross validation set', 'lambda', 'Error', 'figure6')

print('lambda', 'Train Error', 'Validation Error')
for i = 1, lambda_vec:size(1) do
    print(lambda_vec[i], error_train[i], error_val[i])
end

pause()