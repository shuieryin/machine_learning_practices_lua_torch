---
--- Created by shuieryin.
--- DateTime: 11/12/2017 9:41 PM
---

package.path = package.path .. ";../../../lib/?.lua"

require "util"
require "optim"
require "ex2_func"

-- Machine Learning Online Class - Exercise 2: Logistic Regression
--  Instructions
--  ------------
--
--  This file contains code that helps you get started on the second part
--  of the exercise which covers regularization with logistic regression.
--
--  You will need to complete the following functions in this exericse:

require 'plotData'
require 'sigmoid'
require 'costFunction'
require 'predict'
require 'costFunctionReg'
require 'mapFeature'
require 'plotDecisionBoundary'

--  For this exercise, you will not need to change any code in this file,
--  or any other files other than those mentioned above.

-- Load Data
--  The first two columns contains the X values and the third column
--  contains the label (y).

local X, y, init_theta = loadData("ex2data2.txt")

-- Find Indices of Positive and Negative Examples
plotData(X, y)
pause()

-- =========== Part 1: Regularized Logistic Regression ============
--  In this part, you are given a dataset with data points that are not
--  linearly separable. However, you would still like to use logistic
--  regression to classify the data points.
--
--  To do so, you introduce more features to use -- in particular, you add
--  polynomial features to our data matrix (similar to polynomial
--  regression).

-- Add Polynomial Features

-- Note that mapFeature also adds a column of ones for us, so the intercept term is handled
X = mapFeature(X[{ {}, 2 }], X[{ {}, 3 }])
init_theta = torch.zeros(X:size(2), 1)

-- Set regularization parameter lambda to 1
local lambda = 1

-- Compute and display initial cost and gradient for regularized logistic regression
local cost, grad = costFunctionReg(init_theta, X, y, lambda)

print('Cost at initial theta (zeros):', cost)
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros) - first five values only:')
print(grad[{ { 1, 5 }, {} }])
print('Expected gradients (approx) - first five values only:\n')
print(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n')
pause()

-- Compute and display cost and gradient with all-ones theta and lambda = 10
local testCost, testGrad = costFunctionReg(init_theta:fill(1), X, y, 10)

print('\nCost at test theta (with lambda = 10):')
print(testCost)
print('Expected cost (approx): 3.16\n')
print('Gradient at test theta - first five values only:')
print(testGrad[{ { 1, 5 }, {} }])
print('Expected gradients (approx) - first five values only:')
print(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n')
pause()

-- ============= Part 2: Regularization and Accuracies =============
--  Optional Exercise:
--  In this part, you will get to try different values of lambda and
--  see how regularization affects the decision coundart
--
--  Try the following values of lambda (0, 1, 10, 100).
--
--  How does the decision boundary change when you vary lambda? How does
--  the training set accuracy vary?

-- Optimize
local finalTheta, costHistory = optim.cg( function(theta)
    return costFunctionReg(theta, X, y, lambda)
end, init_theta:zero(), { maxIter = 400 })

plotDecisionBoundary(finalTheta, X, y)
pause()

-- Compute accuracy on our training set
local p = predict(finalTheta, X)
print('Train Accuracy: ', p:eq(y):sum() / length(p))
print('Expected accuracy (with lambda = 1): 83.1 (approx)\n')