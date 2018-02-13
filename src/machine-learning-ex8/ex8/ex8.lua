---
--- Created by shuieryin.
--- DateTime: 08/01/2018 1:48 PM
---

require "../../../lib/util"

require "util"
require "image"
require "ex8_func"
require "distributions"
require "torchx"

-- Machine Learning Online Class
--  Exercise 8 | Anomaly Detection and Collaborative Filtering
--  Instructions
--  ------------

--  This file contains code that helps you get started on the
--  exercise. You will need to complete the following functions:

require 'estimateGaussian'
require 'selectThreshold'
require 'cofiCostFunc'
require 'multivariateGaussian'
require 'visualizeFit'

--  For this exercise, you will not need to change any code in this file,
--  or any other files other than those mentioned above.

-- ================== Part 1: Load Example Dataset  ===================
--  We start this exercise by using a small dataset that is easy to
--  visualize.

--  Our example case consists of 2 network server statistics across
--  several machines: the latency and throughput of each machine.
--  This exercise will help us find possibly faulty (or very fast) machines.

print('Visualizing example dataset for outlier detection.')

--  The following command loads the dataset. You should now have the
--  variables X, Xval, yval in your environment
local X, Xval, yval, plot = loadData('ex8data1.mat', 'Figure 1: The first dataset')

pause()

-- ================== Part 2: Estimate the dataset statistics ===================
--  For this exercise, we assume a Gaussian distribution for the dataset.

--  We first estimate the parameters of our assumed Gaussian distribution,
--  then compute the probabilities for each of the points and then visualize
--  both the overall distribution and where each of the points falls in
--  terms of that distribution.

print('Visualizing Gaussian fit.')

--  Estimate my and sigma2
local mu, sigma2 = estimateGaussian(X)

--  Returns the density of the multivariate normal at each data point (row) of X
local p = multivariateGaussian(X, mu, sigma2)
--local p = distributions.mvn.logpdf(X, mu, diag(sigma2))

--  Visualize the fit
visualizeFit(X, mu, sigma2)
--xlabel('Latency (ms)');
--ylabel('Throughput (mb/s)');

pause()

-- ================== Part 3: Find Outliers ===================
--  Now you will find a good epsilon threshold using a cross-validation set
--  probabilities given the estimated Gaussian distribution

local pval = multivariateGaussian(Xval, mu, sigma2)

local epsilon, F1 = selectThreshold(yval, pval)
print('Best epsilon found using cross-validation:', epsilon)
print('Best F1 on Cross Validation Set:', F1)
print('   (you should see a value epsilon of about 8.99e-05)')
print('   (you should see a Best F1 value of  0.875000)')

--  Find the outliers in the training set and plot the
local outliers = torch.find(p:lt(epsilon), 1)
local plot_x = {}
local plot_y = {}
for i = 1, #outliers do
    local outlier = X[outliers[i]]
    plot_x[#plot_x + 1] = outlier[1]
    plot_y[#plot_y + 1] = outlier[2]
end

--  Draw a red circle around those outliers
if #plot_x > 0 and #plot_y > 0 then
    plot:circle(plot_x, plot_y, 'red', 'outlier'):redraw()
    itorchHtml(plot, 'outliers.html')
    print("see 'outliers.html'")
    pause()
end

-- ================== Part 4: Multidimensional Outliers ===================
--  We will now use the code from the previous part and apply it to a
--  harder problem in which more features describe each datapoint and only
--  some features indicate whether a point is an outlier.

--  Loads the second dataset. You should now have the
--  variables X, Xval, yval in your environment
X, Xval, yval, plot = loadData('ex8data2.mat')

--  Apply the same steps to the larger dataset
mu, sigma2 = estimateGaussian(X)

--  Training set
p = multivariateGaussian(X, mu, sigma2)

--  Cross-validation set
pval = multivariateGaussian(Xval, mu, sigma2)

--  Find the best threshold
epsilon, F1 = selectThreshold(yval, pval)

print('Best epsilon found using cross-validation:', epsilon)
print('Best F1 on Cross Validation Set:', F1)
print('   (you should see a value epsilon of about 1.38e-18)')
--- the diff of epsilon may be caused by for each step implementation of lua, just a guess :)
print('   (you should see a Best F1 value of 0.615385)')
print('# Outliers found:', p:lt(epsilon):sum())