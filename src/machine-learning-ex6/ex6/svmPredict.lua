---
--- Generated by EmmyLua(https://github.com/EmmyLua)
--- Created by shuieryin.
--- DateTime: 12/01/2018 9:29 PM
---

function svmPredict(model, X)
    --SVMPREDICT returns a vector of predictions using a trained SVM model
    --(svmTrain).
    --   pred = SVMPREDICT(model, X) returns a vector of predictions using a
    --   trained SVM model (svmTrain). X is a mxn matrix where there each
    --   example is a row. model is a svm model returned from svmTrain.
    --   predictions pred is a m x 1 column of predictions of {0, 1} values.

    -- Check if we are getting a column vector, if so, then assume that we only
    -- need to do prediction for a single example
    if X:size(2) == 1 then
        -- Examples should be in rows
        X = X:t()
    end

    -- Dataset
    local m = X:size(1)
    local p = torch.zeros(m, 1)
    local pred = torch.zeros(m, 1)

    if model.kernelFunction.name == 'linearKernel' then
        -- We can use the weights and bias directly if working with the
        -- linear kernel
        p = X * model.w + model.b
    elseif model.kernelFunction.name == 'gaussianKernel' then
        -- Vectorized RBF Kernel
        -- This is equivalent to computing the kernel on every pair of examples
        local X1 = torch.pow(X, 2):sum(2)
        local X2 = torch.pow(model.X, 2):sum(2):t()
        local K = bsxfun(plus, X1, bsxfun(plus, X2, -2 * X * model.X:t()))
        K = torch.cpow(torch.Tensor(K:size()):fill(model.kernelFunction.gaussian(1, 0)), K)
        K = bsxfun(times, torch.reshape(model.y, 1, model.y:size(1)), K)
        K = bsxfun(times, torch.reshape(model.alphas, 1, model.alphas:size(1)), K)
        p = K:sum(2)
    else
        -- Other Non-linear kernel
        for i = 1, m do
            local prediction = 0
            for j = 1, model.X:size(1) do
                prediction = prediction + model.alphas[j][1] * model.y[j][1] * model.kernelFunction.func(X[i]:t(), model.X[j]:t())
            end
            p[i] = prediction + model.b
        end
    end

    -- Convert predictions into 0 / 1
    pred[p:gt(-1)] = 1
    pred[p:lt(0)] = 0

    return pred
end