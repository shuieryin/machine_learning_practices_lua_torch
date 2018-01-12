---
--- Generated by EmmyLua(https://github.com/EmmyLua)
--- Created by shuieryin.
--- DateTime: 12/01/2018 10:44 PM
---

function normalizeRatings(Y, R)
    --NORMALIZERATINGS Preprocess data by subtracting mean rating for every
    --movie (every row)
    --   [Ynorm, Ymean] = NORMALIZERATINGS(Y, R) normalized Y so that each movie
    --   has a rating of 0 on average, and returns the mean rating in Ymean.

    local m = Y:size(1)
    local Ymean = torch.zeros(m, 1)
    local Ynorm = torch.zeros(Y:size())
    for i = 1, m do
        local idx = R[i]
        local yidx = Y[{ i, idx }]
        Ymean[i] = yidx:sum() / idx:sum()
        Ynorm[{ i, idx }] = yidx - Ymean[i][1]
    end
    return Ynorm, Ymean
end