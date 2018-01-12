---
--- Created by shuieryin.
--- DateTime: 01/01/2018 9:58 PM
---

function loadData(filename)
    local data = torch.Tensor(matrixLoad(filename))
    local colSize = data:size()[2]
    local columnNum = {}
    for i = 1, colSize - 1 do
        columnNum[i] = i
    end
    local X_ori = data:index(2, torch.LongTensor(columnNum))
    local X = torch.Tensor(data:size()[1], 1):fill(1):cat(X_ori, 2)
    local y = data:select(2, colSize)
    local theta = torch.Tensor(colSize, 1):fill(0)
    return X, y, theta, X_ori
end