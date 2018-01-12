---
--- Created by shuieryin.
--- DateTime: 03/01/2018 2:46 PM
---

package.path = package.path .. ";../../../lib/?.lua"

require "util"
require "image"
require "ex7_func"
require "unsup"

-- ================= Part 1: Find Closest Centroids ====================
--  To help you implement K-Means, we have divided the learning algorithm
--  into two functions -- findClosestCentroids and computeCentroids. In this
--  part, you should complete the code in the findClosestCentroids function.

print('Finding closest centroids.')

-- Load an example dataset that we will be using
local X = loadData('ex7data2.mat', 'black', 'Figure 1: The expected output.')

-- Select an initial set of centroids
local K = 3 -- 3 Centroids
local initial_centroids = torch.Tensor({ { 3, 3 }, { 6, 2 }, { 8, 5 } })

-- Find the closest centroids for the examples using the
-- initial_centroids
local idx = findClosestCentroids(X, initial_centroids)

print('Closest centroids for the first 3 examples:')
print(idx[{ { 1, 3 } }])
print('(the closest centroids should be 1, 3, 2 respectively)')

pause()

-- ===================== Part 2: Compute Means =========================
--  After implementing the closest centroids function, you should now
--  complete the computeCentroids function.

print('Computing centroids means.')

--  Compute means based on the closest centroids found in the previous part.
local centroids = computeCentroids(X, idx, K)

print('Centroids computed after initial finding of closest centroids: ')
print(centroids)
print('(the centroids should be')
print('   [ 2.428301 3.157924 ]')
print('   [ 5.813503 2.633656 ]')
print('   [ 7.119387 3.616684 ]')

pause()

-- =================== Part 3: K-Means Clustering ======================
--  After you have completed the two functions computeCentroids and
--  findClosestCentroids, you have all the necessary pieces to run the
--  kMeans algorithm. In this part, you will run the K-Means algorithm on
--  the example dataset we have provided.

print('\nRunning K-Means clustering on example dataset.')

-- Settings for running K-Means
K = 3
local max_iters = 10

-- For consistency, here we set centroids to specific values
-- but in practice you want to generate them automatically, such as by
-- settings them to be random examples (as can be seen in
-- kMeansInitCentroids).

-- Run K-Means algorithm. The 'true' at the end tells our function to plot
-- the progress of K-Means
centroids = unsup.kmeans(X, K, max_iters)
print('K-Means Done.')

pause()

-- ============= Part 4: K-Means Clustering on Pixels ===============
--  In this exercise, you will use K-Means to compress an image. To do this,
--  you will first run K-Means on the colors of the pixels in the image and
--  then you will map each pixel onto its closest centroid.
--
--  You should now complete the code in kMeansInitCentroids.m

print('Running K-Means clustering on pixels from an image.')
local A = image.load('bird_small.png')

A = A / 255

-- Size of the image
local img_size = A:size()

-- Reshape the image into an Nx3 matrix where N = number of pixels.
-- Each row will contain the Red, Green and Blue pixel values
-- This gives us our dataset matrix X that we will use K-Means on.
X = torch.reshape(A, img_size[2] * img_size[3], img_size[1])

-- Run your K-Means algorithm on this data
-- You should try different values of K and max_iters here
K = 16
max_iters = 10

-- When using K-Means, it is important the initialize the centroids
-- randomly.

-- Run K-Means
centroids = unsup.kmeans(X, K, max_iters)

pause()

-- ================= Part 5: Image Compression ======================
--  In this part of the exercise, you will use the clusters of K-Means to
--  compress an image. To do this, we first find the closest clusters for
--  each example. After that, we

print('Applying K-Means to compress an image.')

-- Find closest cluster members
idx = findClosestCentroids(X, centroids)

-- Essentially, now we have represented the image X as in terms of the
-- indices in idx.

-- We can now recover the image from the indices (idx) by mapping each pixel
-- (specified by its index in idx) to the centroid value
local X_recovered = torch.zeros(X:size())
for i = 1, idx:numel() do
    X_recovered[i] = centroids[idx[i][1]]
end

-- Reshape the recovered image into proper dimensions
X_recovered = torch.reshape(X_recovered, img_size)

image.save("data.png", image.toDisplayTensor {
    input = X_recovered
})
os.execute(openCmd .. ' "' .. paths.cwd() .. '/data.png"')
pause()