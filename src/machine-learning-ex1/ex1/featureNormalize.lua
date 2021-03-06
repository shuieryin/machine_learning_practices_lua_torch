---
--- Generated by EmmyLua(https://github.com/EmmyLua)
--- Created by shuieryin.
--- DateTime: 12/01/2018 4:21 PM
---

function featureNormalize(X)
    --FEATURENORMALIZE Normalizes the features in X
    --   FEATURENORMALIZE(X) returns a normalized version of X where
    --   the mean value of each feature is 0 and the standard deviation
    --   is 1. This is often a good preprocessing step to do when
    --   working with learning algorithms.

    -- You need to set these values correctly
    local X_norm = X:clone()
    local colSize = X:size(2)
    local mu = torch.zeros(1, colSize)
    local sigma = torch.zeros(mu:size())

    -- ====================== YOUR CODE HERE ======================
    -- Instructions: First, for each feature dimension, compute the mean
    --               of the feature and subtract it from the dataset,
    --               storing the mean value in mu. Next, compute the
    --               standard deviation of each feature and divide
    --               each feature by it's standard deviation, storing
    --               the standard deviation in sigma.
    --
    --               Note that X is a matrix where each column is a
    --               feature and each row is an example. You need
    --               to perform the normalization separately for
    --               each feature.
    --
    -- Hint: You might find the 'mean' and 'std' functions useful.


    -- ============================================================

    return X_norm, mu, sigma
end