---
--- Generated by EmmyLua(https://github.com/EmmyLua)
--- Created by shuieryin.
--- DateTime: 12/01/2018 5:42 PM
---

function costFunctionReg(theta, X, y, lambda)
    --COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
    --   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
    --   theta as the parameter for regularized logistic regression and the
    --   gradient of the cost w.r.t. to the parameters.

    -- Initialize some useful values
    local m = length(y) -- number of training examples

    -- You need to return the following variables correctly
    local J = 0
    local grad = torch.zeros(theta:size())

    -- ====================== YOUR CODE HERE ======================
    -- Instructions: Compute the cost of a particular choice of theta.
    --               You should set J to the cost.
    --               Compute the partial derivatives and set grad to the partial
    --               derivatives of the cost w.r.t. each parameter in theta


    -- =============================================================

    return J, grad
end