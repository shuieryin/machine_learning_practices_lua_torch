---
--- Created by shuieryin.
--- DateTime: 03/12/2017 5:07 PM
---

package.path = package.path .. ";../../../lib/?.lua"

require "util"
require "optim"
require "ex1_func"

-- Machine Learning Online Class - Exercise 1: Linear Regression
--  Instructions
--  ------------
--
--  This file contains code that helps you get started on the
--  linear exercise. You will need to complete the following functions
--  in this exericse:

require 'warmUpExercise'
require 'plotData'
require 'gradientDescent'
require 'computeCost'
require 'gradientDescentMulti'
require 'computeCostMulti'
require 'featureNormalize'
require 'normalEqn'

--  For this exercise, you will not need to change any code in this file,
--  or any other files other than those mentioned above.
--
-- x refers to the population size in 10,000s
-- y refers to the profit in $10,000s

-- ==================== Part 1: Basic Function ====================
-- Complete warmUpExercise.lua
print('Running warmUpExercise ...')
print('5x5 Identity Matrix:')
print(warmUpExercise())
pause()

-- ======================= Part 2: Plotting =======================
print('Plotting Data ...')
local X, y, theta = loadData("ex1data1.txt")
local XColTwo = X[{ {}, 2 }]

local plot = plotData(XColTwo, y)
pause()

-- =================== Part 3: Gradient descent ===================
print('Running Gradient Descent ...')
local cost = computeCost(X, y, theta)
print("cost:", cost)

local iterations = 1500
local alpha = 0.01

local finalTheta, history = gradientDescent(X, y, theta, alpha, iterations)

-- print theta to screen
print('Theta found by gradient descent:')
print(finalTheta)
pause()

local predict1 = torch.Tensor({ { 1, 3.5 } }) * finalTheta
local predict2 = torch.Tensor({ { 1, 7 } }) * finalTheta
print('For population = 35,000, we predict a profit of', predict1[1][1] * 10000)
print('For population = 70,000, we predict a profit of', predict2[1][1] * 10000)
pause()

-- ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...')

local plot_x = torch.Tensor({ { XColTwo:min(), y:max() } })
local plot_y = finalTheta[2]:sum() * plot_x + finalTheta[1]:sum()
plot:line(plot_x:totable(), plot_y:totable(), 'blue', 'Decision boundary'):legend(true):title('Line Plot Demo'):draw()
itorchHtml(plot, 'decision_boundary.html')