---
--- Generated by EmmyLua(https://github.com/EmmyLua)
--- Created by shuieryin.
--- DateTime: 12/01/2018 8:38 PM
---

function sigmoid(z)
    --SIGMOID Compute sigmoid functoon
    --   J = SIGMOID(z) computes the sigmoid of z.
    return torch.pow(1 + (-z):exp(), -1)
end