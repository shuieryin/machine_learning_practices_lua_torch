---
--- Created by shuieryin.
--- DateTime: 07/12/2017 10:37 PM
---

package.path = package.path .. ";../../../lib/?.lua"

require "util"
require "optim"
require "ex2_func"

-- Machine Learning Online Class - Exercise 2: Logistic Regression
--
--  Instructions
--  ------------
--
--  This file contains code that helps you get started on the logistic
--  regression exercise. You will need to complete the following functions
--  in this exericse:

require 'plotData'
require 'sigmoid'
require 'costFunction'
require 'predict'
require 'costFunctionReg'

--  For this exercise, you will not need to change any code in this file,
--  or any other files other than those mentioned above.

local X, y, init_theta = loadData("ex2data1.txt")

-- ==================== Part 1: Plotting ====================
--  We start the exercise by first plotting the data to understand the
--  the problem we are working with.

print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.')

-- ============ Part 2: Compute Cost and Gradient ============
--  In this part of the exercise, you will implement the cost and gradient
--  for logistic regression. You need to complete the code in
--  costFunction.m

local plot = plotData(X, y)
pause()

local J, grad = costFunction(X, y, init_theta)
print("\nJ: " )
print(J)
print("\ngrad: ")
print(grad)

print('Cost at initial theta (zeros): ')
print(J)
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros):')
print(grad)
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')
pause()

-- ============= Part 3: Optimizing using fminunc  =============
--  In this exercise, you will use a built-in function (fminunc) to find the
--  optimal parameters theta.

local finalTheta, cost_history = optim.cg(function(theta)
    return costFunction(X, y, theta)
end, init_theta, { maxIter = 400 })
print("\nCost at theta found by fminunc: ")
print(cost_history[#cost_history])
print("Expected cost (approx): 0.203")
print("\ntheta:")
print(finalTheta)
print("Expected theta (approx):")
print(' -25.161\n 0.206\n 0.201\n')

local XColTwo = X[{ {}, 2 }]
local plot_x = torch.Tensor({ { XColTwo:min() - 2, XColTwo:max() + 2 } })
local x0 = (-1 / finalTheta[3]:sum())
if x0 == -math.huge then
    x0 = 0
end
local plot_y = x0 * (finalTheta[2]:sum() * plot_x + finalTheta[1]:sum())
plot:line(plot_x:totable(), plot_y:totable(), 'blue', 'Decision boundary'):legend(true):title('Line Plot Demo'):redraw()
itorchHtml(plot, 'decision_boundary.html')
pause()

-- ============== Part 4: Predict and Accuracies ==============
--  After learning the parameters, you'll like to use it to predict the outcomes
--  on unseen data. In this part, you will use the logistic regression model
--  to predict the probability that a student with score 45 on exam 1 and 
--  score 85 on exam 2 will be admitted.

--  Furthermore, you will compute the training and test set accuracies of 
--  our model.

--  Your task is to complete the code in predict.m

--  Predict probability for a student with score 45 on exam 1 
--  and score 85 on exam 2 

local prob = sigmoid(torch.Tensor({ { 1, 45, 85 } }) * finalTheta)
print("\nFor a student with scores 45 and 85, we predict an admission probability of ")
print(prob[1][1])
print('Expected value: 0.775 +/- 0.002\n')

-- Compute accuracy on our training set
local predict = sigmoid(X * finalTheta)
print('Train Accuracy: ', predict[y:eq(1)]:mean() * 100)
print('Expected accuracy (approx): 89.0\n')