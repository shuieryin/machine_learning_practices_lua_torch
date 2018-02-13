---
--- Created by shuieryin.
--- DateTime: 27/12/2017 2:49 PM
---

require "../../../lib/util"

require "optim"
require "util"
require "ex6_func"

-- Machine Learning Online Class
--  Exercise 6 | Support Vector Machines
--  Instructions
--  ------------

--  This file contains code that helps you get started on the
--  exercise. You will need to complete the following functions:

require 'gaussianKernel'
require 'dataset3Params'
require 'processEmail'
require 'emailFeatures'
require 'svmTrain'
require 'visualizeBoundaryLinear'
require 'visualizeBoundary'

--  For this exercise, you will not need to change any code in this file,
--  or any other files other than those mentioned above.

-- =============== Part 1: Loading and Visualizing Data ================
--  We start the exercise by first loading and visualizing the dataset.
--  The following code will load the dataset into your environment and plot
--  the data.

print('Loading and Visualizing Data ...\n')

-- Load from ex6data1:
-- You will have X, y in your environment
local X, y, plot = loadData('ex6data1.mat')

pause()

-- ==================== Part 2: Training Linear SVM ====================
--  The following code will train a linear SVM on the dataset and plot the
--  decision boundary learned.

print('Training Linear SVM ...')

-- You should try to change the C value below and see how the decision
-- boundary varies (e.g., try C = 1000)
local C = 1
local model = svmTrain(X, y, C, { name = "linearKernel" }, 1e-3, 20)
visualizeBoundaryLinear(X, y, model, plot)

pause()

-- =============== Part 3: Implementing Gaussian Kernel ===============
--  You will now implement the Gaussian kernel to use
--  with the SVM. You should complete the code in gaussianKernel.m

print('\nEvaluating the Gaussian Kernel ...')

local x1 = torch.Tensor({ { 1, 2, 1 } })
local x2 = torch.Tensor({ { 0, 4, -1 } })
local sigma = 2
local sim = gaussianKernel(x1, x2, sigma)

print('Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = :' .. sigma)
print(sim)
print('(for sigma = 2, this value should be about 0.324652)')

pause()

-- =============== Part 4: Visualizing Dataset 2 ================
--  The following code will load the next dataset into your environment and
--  plot the data.

print('Loading and Visualizing Data ...')

-- Load from ex6data2:
-- You will have X, y in your environment
X, y, plot = loadData('ex6data2.mat')

pause()

-- ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
--  After you have implemented the kernel, we can now use it to train the
--  SVM classifier.
print('Training SVM with RBF Kernel (this may take 1 to 2 minutes) ...')

-- SVM Parameters
C = 1
sigma = 0.1

-- We set the tolerance and max_passes lower here so that the code will run
-- faster. However, in practice, you will want to run the training to
-- convergence.
model = svmTrain(X, y, C, {
    name = "gaussianKernel",
    func = function(x1, x2)
        return gaussianKernel(x1, x2, sigma)
    end,
    gaussian = function(x1, x2)
        return gaussian(x1, x2, sigma)
    end
})
visualizeBoundary(X, y, model, plot)

pause()

-- =============== Part 6: Visualizing Dataset 3 ================
--  The following code will load the next dataset into your environment and
--  plot the data.

print('Loading and Visualizing Data ...')
-- Load from ex6data3:
-- You will have X, y in your environment
local Xval, yval
X, y, Xval, yval, plot = loadData2('ex6data3.mat')

pause()

-- ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========
--  This is a different dataset that you can use to experiment with. Try
--  different values of C and sigma here.

-- Try different SVM Parameters here
C, sigma = dataset3Params(X, y, Xval, yval)

-- Train the SVM
model = svmTrain(X, y, C, {
    name = "gaussianKernel",
    func = function(x1, x2)
        return gaussianKernel(x1, x2, sigma)
    end,
    gaussian = function(x1, x2)
        return gaussian(x1, x2, sigma)
    end
})

visualizeBoundary(X, y, model)
pause()