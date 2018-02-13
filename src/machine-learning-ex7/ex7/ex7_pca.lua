---
--- Created by shuieryin.
--- DateTime: 04/01/2018 2:43 PM
---

require "../../../lib/util"

require "util"
require "image"
require "ex7_func"
require "gnuplot"
local matio = require 'matio'
local Plot = require 'itorch.Plot'

-- Machine Learning Online Class
--  Exercise 7 | Principle Component Analysis and K-Means Clustering
--  Instructions
--  ------------

--  This file contains code that helps you get started on the
--  exercise. You will need to complete the following functions:

require 'pca'
require 'projectData'
require 'recoverData'
require 'computeCentroids'
require 'findClosestCentroids'
require 'kMeansInitCentroids'
require 'featureNormalize'
require 'displayData'
require 'runkMeans'

--  For this exercise, you will not need to change any code in this file,
--  or any other files other than those mentioned above.

-- ================== Part 1: Load Example Dataset  ===================
--  We start this exercise by using a small dataset that is easily to
--  visualize

print('Visualizing example dataset for PCA.')

--  The following command loads the dataset. You should now have the
--  variable X in your environment
local X, plot = loadData('ex7data1.mat', 'blue', 'Figure 5: Computed eigenvectors of the dataset')

pause()

-- =============== Part 2: Principal Component Analysis ===============
--  You should now implement PCA, a dimension reduction technique. You
--  should complete the code in pca.m

print('Running PCA on example dataset.')

--  Before running PCA, it is important to first normalize X
local X_norm, mu, sigma = featureNormalize(X)

--  Run PCA
local U, S = pca(X_norm)

--  Compute mu, the mean of the each feature

--  Draw the eigenvectors centered at mean of data. These lines show the
--  directions of maximum variations in the dataset.
local plot_x = mu[{ {}, 1 }]
local plot_y = mu[{ {}, 2 }]
local nextMu = mu + U[{ {}, 1 }] * 1.5 * S[{ { 1, 1 } }][1]
plot_x = plot_x:cat(nextMu[{ {}, 1 }], 1)
plot_y = plot_y:cat(nextMu[{ {}, 2 }], 1)
plot:line(plot_x, plot_y, 'black', 'towards')

plot_x = mu[{ {}, 1 }]
plot_y = mu[{ {}, 2 }]
nextMu = mu + U[{ {}, 2 }] * 1.5 * S[{ { 2, 2 } }][1]
plot_x = plot_x:cat(nextMu[{ {}, 1 }], 1)
plot_y = plot_y:cat(nextMu[{ {}, 2 }], 1)
plot:line(plot_x, plot_y, 'black', 'towards'):redraw()
itorchHtml(plot, 'ex7data1.html')

print('Top eigenvector:')
print(' U(:,1) = ', U[{ 1, 1 }], U[{ 2, 1 }])
print('(you should expect to see -0.707107 -0.707107)')

pause()

-- =================== Part 3: Dimension Reduction ===================
--  You should now implement the projection step to map the data onto the
--  first k eigenvectors. The code will then plot the data in this reduced
--  dimensional space.  This will show you what the data looks like when
--  using only the corresponding eigenvectors to reconstruct it.

--  You should complete the code in projectData.m

print('Dimension reduction on example dataset.')

--  Plot the normalized dataset (returned from pca)
local drPlot = Plot():circle(X_norm[{ {}, 1 }], X_norm[{ {}, 2 }], 'blue', 'Data')
drPlot:title('Dimension reduction')
drPlot:xaxis('X1'):yaxis('X2'):draw()
itorchHtml(drPlot, 'DimensionReduction.html')

--  Project the data onto K = 1 dimension
local K = 1
local Z = projectData(X_norm, U, K)
print('Projection of the first example:', Z[1][1])
print('(this value should be about 1.481274)')
pause()

local X_rec = recoverData(Z, U, K)
print('Approximation of the first example:', X_rec[{ 1, 1 }], X_rec[{ 1, 2 }])
print('(this value should be about  -1.047419 -1.047419)')

--  Draw lines connecting the projected points to the original points
drPlot:circle(X_rec[{ {}, 1 }], X_rec[{ {}, 2 }], 'red', 'Rec')
for i = 1, X_norm:size(1) do
    local cutPlotX = {}
    local cutPlotY = {}
    local curXnormI = X_norm[i]
    local curXrecI = X_rec[i]
    cutPlotX[1] = curXnormI[1]
    cutPlotY[1] = curXnormI[2]
    cutPlotX[2] = curXrecI[1]
    cutPlotY[2] = curXrecI[2]
    drPlot:line(cutPlotX, cutPlotY, 'black', 'towards')
end
drPlot:redraw()
itorchHtml(drPlot, 'DimensionReduction.html')

pause()

-- =============== Part 4: Loading and Visualizing Face Data =============
--  We start the exercise by first loading and visualizing the dataset.
--  The following code will load the dataset into your environment

print('Loading face dataset.')

--  Load Face dataset
local tensors = matio.load('ex7faces.mat', { 'X' })
X = tensors['X']

--  Display the first 100 faces in the dataset
displayData(X[{ { 1, 100 } }], "data_original_face.png")

pause()

-- =========== Part 5: PCA on Face Data: Eigenfaces  ===================
--  Run PCA and visualize the eigenvectors which are in this case eigenfaces
--  We display the first 36 eigenfaces.

print('Running PCA on face dataset.\n(this might take a minute or two ...)')

--  Before running PCA, it is important to first normalize X by subtracting
--  the mean value from each feature
X_norm, mu, sigma = featureNormalize(X)

--  Run PCA
U, S = pca(X_norm)

--  Visualize the top 36 eigenvectors found
displayData(U[{ {}, { 1, 36 } }]:t(), "data_pca_face.png")

pause()

-- ============= Part 6: Dimension Reduction for Faces =================
--  Project images to the eigen space using the top k eigenvectors
--  If you are applying a machine learning algorithm
print('Dimension reduction for face dataset.')

K = 100
Z = projectData(X_norm, U, K)

print('The projected data Z has a size of: ')
print(Z:size())

pause()

-- ==== Part 7: Visualization of Faces after PCA Dimension Reduction ====
--  Project images to the eigen space using the top K eigen vectors and
--  visualize only using those K dimensions
--  Compare to the original input, which is also displayed

print('Visualizing the projected (reduced dimension) faces.')

K = 100
X_rec = recoverData(Z, U, K)

-- Display reconstructed data from only k eigenfaces
displayData(X_rec[{ { 1, 100 } }], 'data_recovered_face.png')

pause()

-- === Part 8(a): Optional (ungraded) Exercise: PCA for Visualization ===
--  One useful application of PCA is to use it to visualize high-dimensional
--  data. In the last K-Means exercise you ran K-Means on 3-dimensional
--  pixel colors of an image. We first visualize this output in 3D, and then
--  apply PCA to obtain a visualization in 2D.

-- Reload the image from the previous exercise and run K-Means on it
-- For this to work, you need to complete the K-Means assignment first

local A = image.load('bird_small.png')
A = A / 255
local img_size = A:size()
X = torch.reshape(A, img_size[2] * img_size[3], img_size[1])
K = 16
local max_iters = 10
local initial_centroids = kMeansInitCentroids(X, K)
local centroids, idx = runkMeans(X, initial_centroids, max_iters)

--  Sample 1000 random indexes (since working with all the data is
--  too expensive. If you have a fast computer, you may increase this.
local sel = torch.floor(torch.rand(1000, 1) * X:size(1)) + 1

--  Setup Color Palette
--local palette = hsv(K)
--local colors = palette(idx(sel), :)

--  Visualize the data and centroid memberships in 3D
plot_x = torch.zeros(sel:size(1))
plot_y = torch.zeros(sel:size(1))
local plot_z = torch.zeros(sel:size(1))

for i = 1, sel:size(1) do
    plot_x[i] = X[{ sel[i][1], 1 }]
    plot_y[i] = X[{ sel[i][1], 2 }]
    plot_z[i] = X[{ sel[i][1], 3 }]
end

--- don't know how to plot color, please suppliment if you know how
gnuplot.scatter3(plot_x, plot_y, plot_z)

pause()

-- === Part 8(b): Optional (ungraded) Exercise: PCA for Visualization ===
-- Use PCA to project this cloud to 2D for visualization

-- Subtract the mean to use PCA
X_norm, mu, sigma = featureNormalize(X)

-- PCA and project the data to 2D
U, S = pca(X_norm)
Z = projectData(X_norm, U, 2)

-- Plot in 2D
plot_x = torch.Tensor(sel:size(1), Z:size(2))
for i = 1, sel:size(1) do
    plot_x[i] = Z[i]
end

--- don't know hot to plot color, please suppliment if you know how
plot = Plot():circle(X[{ {}, 1 }], X[{ {}, 2 }], 'blue')
plot:title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction')
plot:xaxis('X1'):yaxis('X2'):draw()
itorchHtml(plot, 'face_2d_reduction.html')
pause()