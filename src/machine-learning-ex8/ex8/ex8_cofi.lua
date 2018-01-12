---
--- Generated by EmmyLua(https://github.com/EmmyLua)
--- Created by shuieryin.
--- DateTime: 09/01/2018 5:21 PM
---

package.path = package.path .. ";../../../lib/?.lua"

require "util"
require "image"
require "ex8_func"
require "distributions"
require "torchx"
require "gnuplot"
require "optim"

--%% Machine Learning Online Class
--%  Exercise 8 | Anomaly Detection and Collaborative Filtering
--%  Instructions
--%  ------------

--%  This file contains code that helps you get started on the
--%  exercise. You will need to complete the following functions:

require 'estimateGaussian'
require 'selectThreshold'
require 'cofiCostFunc'
require 'checkCostFunction'
require 'loadMovieList'
require 'normalizeRatings'

--%  For this exercise, you will not need to change any code in this file,
--%  or any other files other than those mentioned above.

-- =============== Part 1: Loading movie ratings dataset ================
--  You will start by loading the movie ratings dataset to understand the
--  structure of the data.

print('Loading movie ratings dataset.')

--  Load data
local Y, R = loadData2('ex8_movies.mat')

--  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on
--  943 users

--  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
--  rating to movie i

--  From the matrix, we can compute statistics like average rating.
print('Average rating for movie 1 (Toy Story) out of 5:', Y[1][R[1]]:mean())

--  We can "visualize" the ratings matrix by plotting it with imagesc
--gnuplot.imagesc(Y, 'color')

pause()

-- ============ Part 2: Collaborative Filtering Cost Function ===========
--  You will now implement the cost function for collaborative filtering.
--  To help you debug your cost function, we have included set of weights
--  that we trained on that. Specifically, you should complete the code in
--  cofiCostFunc.m to return J.

--  Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
local X, Theta = loadData3('ex8_movieParams.mat')

--  Reduce the data set size so that this runs faster
local num_users = 4
local num_movies = 5
local num_features = 3
X = X[{ { 1, num_movies }, { 1, num_features } }]
Theta = Theta[{ { 1, num_users }, { 1, num_features } }]
Y = Y[{ { 1, num_movies }, { 1, num_users } }]
R = R[{ { 1, num_movies }, { 1, num_users } }]

--  Evaluate cost function
local params = torch.reshape(X, X:numel()):cat(torch.reshape(Theta, Theta:numel()))
local J = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, 0)

print('Cost at loaded parameters: (this value should be about 22.22)', J)

pause()

-- ============== Part 3: Collaborative Filtering Gradient ==============
--  Once your cost function matches up with ours, you should now implement
--  the collaborative filtering gradient function. Specifically, you should
--  complete the code in cofiCostFunc.m to return the grad argument.

print('Checking Gradients (without regularization) ... ')

--  Check gradients by running checkNNGradients
checkCostFunction()

pause()

-- ========= Part 4: Collaborative Filtering Cost Regularization ========
--  Now, you should implement regularization for the cost function for
--  collaborative filtering. You can implement it by adding the cost of
--  regularization to the original cost computation.

--  Evaluate cost function
J = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, 1.5)

print('Cost at loaded parameters (lambda = 1.5)\n(this value should be about 31.34)', J)
pause()

-- ======= Part 5: Collaborative Filtering Gradient Regularization ======
--  Once your cost matches up with ours, you should proceed to implement
--  regularization for the gradient.

print('Checking Gradients (with regularization) ... ')

--  Check gradients by running checkNNGradients
checkCostFunction(1.5)

pause()

-- ============== Part 6: Entering ratings for a new user ===============
--  Before we will train the collaborative filtering model, we will first
--  add ratings that correspond to a new user that we just observed. This
--  part of the code will also allow you to put in your own ratings for the
--  movies in our dataset!

local movieList = loadMovieList()

--  Initialize my ratings
local my_ratings = torch.zeros(#movieList, 1)

-- Check the file movie_idx.txt for id of each movie in our dataset
-- For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
my_ratings[1] = 4

-- Or suppose did not enjoy Silence of the Lambs (1991), you can set
my_ratings[98] = 2

-- We have selected a few movies we liked / did not like and the ratings we
-- gave are as follows:
my_ratings[7] = 3
my_ratings[12] = 5
my_ratings[54] = 4
my_ratings[64] = 5
my_ratings[66] = 3
my_ratings[69] = 5
my_ratings[183] = 4
my_ratings[226] = 5
my_ratings[355] = 5

print('New user ratings:')
for i = 1, length(my_ratings) do
    if my_ratings[i][1] > 0 then
        print('Rated ' .. my_ratings[i][1] .. ' for ' .. movieList[i])
    end
end

pause()

-- ================== Part 7: Learning Movie Ratings ====================
--  Now, you will train the collaborative filtering model on a movie rating
--  dataset of 1682 movies and 943 users

print('Training collaborative filtering...')

--  Load data
Y, R = loadData2('ex8_movies.mat')

--  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by
--  943 users
--  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
--  rating to movie i

--  Add our own ratings to the data matrix
Y = my_ratings:cat(Y)
R = my_ratings:ne(0):cat(R)

--  Normalize Ratings
local Ynorm, Ymean = normalizeRatings(Y, R)

--  Useful Values
num_users = Y:size(2)
num_movies = Y:size(1)
num_features = 10

-- Set Initial Parameters (Theta, X)
X = torch.randn(num_movies, num_features)
Theta = torch.randn(num_users, num_features)

local initial_parameters = torch.reshape(X, X:numel()):cat(torch.reshape(Theta, Theta:numel()))

-- Set Regularization
local lambda = 10
local last_str = ''
local index = 1
local theta, costHist = optim.cg( function(theta)
    local curJ, curGrad = cofiCostFunc(theta, Ynorm, R, num_users, num_movies, num_features, lambda)
    io.write(('\b'):rep(#last_str))
    local str = 'Iteration    ' .. index .. ' | Cost: ' .. curJ
    io.write(str)
    io.flush()
    index = index + 1
    last_str = str
    return curJ, curGrad
end, initial_parameters, { maxIter = 100 })

-- Unfold the returned theta back into U and W
X = torch.reshape(theta[{ { 1, num_movies * num_features } }], num_movies, num_features)
Theta = torch.reshape(theta[{ { num_movies * num_features + 1, length(theta) } }], num_users, num_features)

print('Recommender system learning completed.\n')
pause()

-- ================== Part 8: Recommendation for you ====================
--  After training the model, you can now make recommendations by computing
--  the predictions matrix.

local p = X * Theta:t()
local my_predictions = p[{ {}, 1 }] + Ymean

local r, ix = torch.sort(my_predictions, 1, true)
print('Top recommendations for you:')
for i = 1, 10 do
    local j = ix[i]
    print('Predicting rating ' .. my_predictions[j] .. ' for movie ' .. movieList[j])
end

print('\nOriginal ratings provided:')
for i = 1, length(my_ratings) do
    if my_ratings[i][1] > 0 then
        print('Rated ' .. my_ratings[i][1] .. ' for ' .. movieList[i])
    end
end