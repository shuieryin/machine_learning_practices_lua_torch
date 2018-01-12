---
--- Generated by EmmyLua(https://github.com/EmmyLua)
--- Created by shuieryin.
--- DateTime: 12/01/2018 5:42 PM
---

require 'gnuplot'

function plotDecisionBoundary(theta, X, y)
    print("Plotting decision boundary")
    --    %PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
    --    %the decision boundary defined by theta
    --    %   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the
    --    %   positive examples and o for the negative examples. X is assumed to be
    --    %   a either
    --    %   1) Mx3 matrix, where the first column is an all-ones column for the
    --      intercept.
    --   2) MxN, N>3 matrix, where the first column is all-ones

    -- Plot Data

    if X:size(2) <= 3 then
        -- Only need 2 points to define a line, so choose two endpoints
        local plot_x = (X[{ {}, 2 }]:min() - 2):cat(X[{ {}, 2 }]:max() + 2, 2)

        -- Calculate the decision boundary line
        local plot_y = torch.cmul(torch.cdiv(torch.Tensor(theta[3]:size()):fill(-1), theta[3]), torch.cmul(theta[2], plot_x) + theta[1])

        -- Plot, and adjust axes for better viewing
        plot:line(plot_x:totable(), plot_y:totable(), 'blue', 'Decision boundary')
    else
        -- Here is the grid range
        local u = torch.linspace(-1, 1.5, 50)
        local v = torch.linspace(-1, 1.5, 50)

        local z = torch.zeros(u:numel(), v:numel())
        -- Evaluate z = theta*x over the grid
        for i = 1, u:numel() do
            for j = 1, v:numel() do
                z[{ i, j }] = (mapFeature(u[i], v[j]) * theta)[1][1]
            end
        end
        z = z:t() -- important to transpose z before calling contour

        -- Plot z = 0
        -- Notice you need to specify the range [0, 0]
        --contour(u, v, z, [0, 0], 'LineWidth', 2)
        --- don't know hot to plot contour, please suppliment if you know how
        gnuplot.imagesc(z:gt(0), 'color')
    end
end