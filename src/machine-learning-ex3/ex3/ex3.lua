---
--- Created by shuieryin.
--- DateTime: 13/12/2017 11:07 PM
---

package.path = package.path .. ";../../../lib/?.lua"

require "util"
require "torch"
require "ex3_func"

-- Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all
--  Instructions
--  ------------
--  This file contains code that helps you get started on the
--  linear exercise. You will need to complete the following functions
--  in this exericse:

require 'lrCostFunction' --(logistic regression cost function)
require 'oneVsAll'
require 'predictOneVsAll'
require 'predict'
require 'displayData'

--  For this exercise, you will not need to change any code in this file,
--  or any other files other than those mentioned above.

-- Setup the parameters you will use for this part of the exercise
local num_labels = 10          -- 10 labels, from 1 to 10
-- (note that we have mapped "0" to label 10)

-- =========== Part 1: Loading and Visualizing Data =============
--  We start the exercise by first loading and visualizing the dataset.
--  You will be working with a dataset that contains handwritten digits.

local X, y = loadData()
pause()

-- ============ Part 2a: Vectorize Logistic Regression ============
--  In this part of the exercise, you will reuse your logistic regression
--  code from the last exercise. You task here is to make sure that your
--  regularized logistic regression implementation is vectorized. After
--  that, you will implement one-vs-all classification for the handwritten
--  digit dataset.

-- Test case for lrCostFunction
print('Testing lrCostFunction() with regularization')

local theta_t = torch.Tensor({ -2, -1, 1, 2 })
local X_t = torch.zeros(15)
for i = 1, X_t:size(1) do
    X_t[i] = i / 10
end
X_t = torch.ones(5):cat(torch.reshape(X_t, 3, 5):t(), 2)
local y_t = torch.Tensor({ 1, 0, 1, 0, 1 })
local lambda_t = 3
local J, grad = lrCostFunction(theta_t, X_t, y_t, lambda_t)

print('\nCost:', J)
print('Expected cost: 2.534819')
print('\nGradients:')
print(grad)
print('Expected gradients:')
print('0.146561\n -0.548558\n 0.724722\n 1.398003\n')
pause()

-- ============ Part 2b: One-vs-All Training ============
print('Training One-vs-All Logistic Regression...')

local lambda = 0.1
local all_theta = oneVsAll(X, y, num_labels, lambda, 1)
pause()

-- ================ Part 3: Predict for One-Vs-All ================
local p = predictOneVsAll(all_theta, X)
print('Training Set Accuracy: ', p:eq(y):sum() / length(p) * 100)