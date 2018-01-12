---
--- Created by shuieryin.
--- DateTime: 03/12/2017 5:07 PM
---

local Plot = require 'itorch.Plot'
local paths = require "paths"
openCmd = "start"
if package.config:sub(1, 1):match("^/") then
    openCmd = "open"
end

function trim(s)
    if not s then
        return ""
    end
    return (s:gsub("^%s*(.-)%s*$", "%1"))
end

function dirname(str)
    if str:match(".-/.-") then
        local name = string.gsub(str, "(.*/)(.*)", "%1")
        return name
    else
        return ''
    end
end

function basename(str)
    local name = string.gsub(str, "(.*/)(.*)", "%2")
    return name
end

function splitFilename(str)
    local name = string.gsub(str, '(.*)%.(.*)', "%1")
    return name
end

function typeof(var)
    local _type = type(var)
    if (_type ~= "table" and _type ~= "userdata") then
        return _type
    end
    local _meta = getmetatable(var)
    if (_meta ~= nil and _meta._NAME ~= nil) then
        return _meta._NAME
    else
        return _type
    end
end

function table_dump(obj)
    local getIndent, quoteStr, wrapKey, wrapVal, dumpObj
    getIndent = function(level)
        return string.rep("\t", level)
    end
    quoteStr = function(str)
        return '"' .. string.gsub(str, '"', '\\"') .. '"'
    end
    wrapKey = function(val)
        if type(val) == "number" then
            return "[" .. val .. "]"
        elseif type(val) == "string" then
            return "[" .. quoteStr(val) .. "]"
        else
            return "[" .. tostring(val) .. "]"
        end
    end
    wrapVal = function(val, level)
        if type(val) == "table" then
            return dumpObj(val, level)
        elseif type(val) == "number" then
            return val
        elseif type(val) == "string" then
            return quoteStr(val)
        else
            return tostring(val)
        end
    end
    dumpObj = function(obj, level)
        if type(obj) ~= "table" then
            return wrapVal(obj)
        end
        level = level + 1
        local tokens = {}
        tokens[#tokens + 1] = "{"
        for k, v in pairs(obj) do
            tokens[#tokens + 1] = getIndent(level) .. wrapKey(k) .. " = " .. wrapVal(v, level) .. ","
        end
        tokens[#tokens + 1] = getIndent(level - 1) .. "}"
        return table.concat(tokens, "\n")
    end
    return dumpObj(obj, 0)
end

function file_exists(file)
    local f = io.open(file, "rb")
    if f then
        f:close()
    end
    return f ~= nil
end

function lines_from(file)
    if not file_exists(file) then
        return {}
    end
    lines = {}
    for line in io.lines(file) do
        lines[#lines + 1] = line
    end
    return lines
end

function print_r(obj)
    print(table_dump(obj))
    --util.tablePrint(obj)
end

local function pr(obj)
    if obj == nil then
        return "nil    "
    elseif type(obj) == "table" then
        print_r(obj)
    else
        return tostring(obj) .. "    "
    end
end

function p(...)
    local t = table.pack(...)
    local str = ""
    for i = 1, t.n do
        local t1 = pr(t[i])
        if t1 then
            str = str .. t1
        end
    end
    print(str)
end

function deepcopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in next, orig, nil do
            copy[deepcopy(orig_key)] = deepcopy(orig_value)
        end
        setmetatable(copy, deepcopy(getmetatable(orig)))
    else
        -- number, string, boolean, etc
        copy = orig
    end
    return copy
end

function matrixLoad(file)
    if not file_exists(file) then
        return {}
    end
    lines = {}
    for line in io.lines(file) do
        local curLine = {}
        for word in string.gmatch(line, '([^,]+)') do
            curLine[#curLine + 1] = tonumber(word)
        end
        lines[#lines + 1] = curLine
    end
    return lines
end

function getColumns(table, startColNum, numOfCols)
    local returnTable = {}
    if type(table) ~= "table" then
        return returnTable
    end

    local endColNum = startColNum + numOfCols - 1
    for i = 1, #table do
        local curLine = {}
        for j = startColNum, endColNum do
            curLine[#curLine + 1] = table[i][j]
        end
        returnTable[#returnTable + 1] = curLine
    end

    return returnTable
end

function eyeMatrix(size)
    local mtx = {}
    for i = 1, size do
        local curLine = {}
        for j = 1, size do
            if j == i then
                curLine[j] = 1
            else
                curLine[j] = 0
            end
        end
        mtx[i] = curLine
    end
    return mtx
end

local itorchDir = dirname(package.searchpath("torch", package.path)) .. "../itorch"
local base_template = [[
<script type="text/javascript">
$(function() {
    if (typeof (window._bokeh_onload_callbacks) === "undefined"){
  window._bokeh_onload_callbacks = [];
    }
    function load_lib(url, callback){
  window._bokeh_onload_callbacks.push(callback);
  if (window._bokeh_is_loading){
      console.log("Bokeh: BokehJS is being loaded, scheduling callback at", new Date());
      return null;
  }
  console.log("Bokeh: BokehJS not loaded, scheduling load and callback at", new Date());
  window._bokeh_is_loading = true;
  var s = document.createElement('script');
  s.src = url;
  s.async = true;
  s.onreadystatechange = s.onload = function(){
      Bokeh.embed.inject_css("]] .. itorchDir .. [[/bokeh-0.7.0.min.css");
      window._bokeh_onload_callbacks.forEach(function(callback){callback()});
  };
  s.onerror = function(){
      console.warn("failed to load library " + url);
  };
  document.getElementsByTagName("head")[0].appendChild(s);
    }

    bokehjs_url = "]] .. itorchDir .. [[/bokeh-0.7.0.min.js"

    var elt = document.getElementById("${window_id}");
    if(elt==null) {
  console.log("Bokeh: ERROR: autoload.js configured with elementid '${window_id}'"
        + "but no matching script tag was found. ")
  return false;
    }

    if(typeof(Bokeh) !== "undefined") {
  console.log("Bokeh: BokehJS loaded, going straight to plotting");
  var modelid = "${model_id}";
  var modeltype = "Plot";
  var all_models = ${all_models};
  Bokeh.load_models(all_models);
  var model = Bokeh.Collections(modeltype).get(modelid);
  model.attributes.plot_height = 1500;
  console.log(model.attributes);
  $("#${window_id}").html(''); // clear any previous plot in window_id
  var view = new model.default_view({model: model, el: "#${window_id}"});
    } else {
  load_lib(bokehjs_url, function() {
      console.log("Bokeh: BokehJS plotting callback run at", new Date())
      var modelid = "${model_id}";
      var modeltype = "Plot";
      var all_models = ${all_models};
      Bokeh.load_models(all_models);
      var model = Bokeh.Collections(modeltype).get(modelid);
      $("#${window_id}").html(''); // clear any previous plot in window_id
      var view = new model.default_view({model: model, el: "#${window_id}"});
  });
    }
});
</script>
]]

local html_template = [[
<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="utf-8">
        <link rel="stylesheet" href="]] .. itorchDir .. [[/bokeh-0.7.0.min.css" type="text/css" />
        <script type="text/javascript" src="]] .. itorchDir .. [[/bokeh-0.7.0.min.js"></script>
]] .. base_template .. [[
    </head>
    <body>
        <div class="plotdiv" id="${div_id}"></div>
    </body>
</html>
]]

function itorchHtml(plot, filename)
    assert(filename and not paths.dirp(filename), 'filename has to be provided and should not be a directory')
    local html = plot:toTemplate(html_template)
    local f = assert(io.open(filename, 'w'), 'filename cannot be opened in write mode')
    f:write(html)
    f:close()
    os.execute(openCmd .. ' "' .. paths.cwd() .. '/' .. filename .. '"')
    return plot
end

function plotTable(datas, title, xTitle, yTitle, filename)
    local plot = Plot()
    for i = 1, #datas do
        local curData = datas[i]
        plot:line(torch.range(1, #curData.data):totable(), curData.data, curData.color, curData.desc)
    end
    plot:title(title)
    plot:xaxis(xTitle):yaxis(yTitle)
    plot:legend(true):draw()
    itorchHtml(plot, filename .. '.html')
end

function shuffle_tensor(tensor)
    local shuffle_indexes = torch.randperm(tensor:size(1))
    local tensor_shuffled = torch.Tensor(tensor:size())
    for i = 1, tensor:size(1), 1 do
        tensor_shuffled[i] = tensor[shuffle_indexes[i]]
    end
    return tensor_shuffled
end

function minus(a, b)
    return a - b
end

function plus(a, b)
    return a + b
end

function rdivide(a, b)
    return torch.cdiv(a, b)
end

function times(a, b)
    return torch.cmul(a, b)
end

function bsxfun(oper, a, b)
    if type(a) == "number" then
        a = torch.Tensor(b:size()):fill(a)
    elseif a:dim() == 1 then
        a = torch.reshape(a, a:numel(), 1)
    end

    if type(b) == "number" then
        b = torch.Tensor(a:size()):fill(b)
    elseif b:dim() == 1 then
        b = torch.reshape(b, b:numel(), 1)
    end

    if a:dim() == 1 and b:dim() == 1 and a:size(1) == b:size(1) then
        return oper(a:expand(a:size(1), b:size(2)), b:expand(a:size(1), b:size(2)))
    elseif a:size(1) == b:size(1) and a:size(2) > b:size(2) or a:size(2) == b:size(2) and a:size(1) > b:size(1) then
        return oper(a, b:expand(a:size()))
    elseif a:size(1) == b:size(1) and a:size(2) < b:size(2) or a:size(2) == b:size(2) and a:size(1) < b:size(1) then
        return oper(a:expand(b:size()), b)
    end

    return oper(a, b)
end

function meshgrid(x, y)
    local row = x:size(1)
    local col = y:size(1)
    local xx = torch.repeatTensor(x, col, 1)
    local yy = torch.repeatTensor(y:view(-1, 1), 1, row)
    return torch.reshape(xx, row, col), torch.reshape(yy, row, col)
end

-- see if the file exists
function file_exists(file)
    local f = io.open(file, "rb")
    if f then
        f:close()
    end
    return f ~= nil
end

-- get all lines from a file, returns an empty
-- list/table if the file does not exist
function lines_from(file)
    if not file_exists(file) then
        return {}
    end
    lines = {}
    for line in io.lines(file) do
        lines[#lines + 1] = line
    end
    return lines
end

function pause()
    print('Program paused. Press enter to continue.')
    io.read()
end

function colors()
    local colors = {}
    for k, v in pairs(sys.COLORS) do
        if k:match("^[A-Z]%w+$") then
            colors[#colors + 1] = k:lower()
        end
    end
    return colors
end

function hsv(K)
    local colors = colors()
    local palette = {}
    for i = 1, K do
        palette[i] = colors[i % #colors]
    end
    return palette
end

function length(X)
    local length = X:size(1)
    local dim = X:dim()
    if dim > 1 then
        for i = 1, X:dim() do
            local curLen = X:size(i)
            if curLen > length then
                length = curLen
                break
            end
        end
    end
    return length
end

function det(X)
    local m = X:size(1)
    local det = 0
    for i = 1, m do
        local curDet = X[{ 1, i }]
        local forward = 0
        local backward = 0
        for j = i, m + i - 2 do
            local curY = j - i + 2

            local curFowardX = j + 1
            if curFowardX > m then
                curFowardX = curFowardX - m
            end

            local curBackwardX = m - j + i * 2 - 1
            if curBackwardX > m then
                curBackwardX = curBackwardX - m
            end

            if curFowardX == curBackwardX and j == 1 then
                forward = curDet * X[{ curY, curFowardX }]
            elseif curFowardX == curBackwardX and i == m + i - 2 then
                backward = curDet * X[{ curY, curBackwardX }]
            else
                forward = curDet * X[{ curY, curFowardX }]
                backward = curDet * X[{ curY, curBackwardX }]
            end
        end
        det = det + forward
        det = det - backward
    end
    return det
end

function diag(X)
    local src = torch.reshape(X, X:numel())
    local len = length(src)
    local diagMatrix = torch.eye(len)
    for i = 1, len do
        diagMatrix[{ i, i }] = src[i]
    end
    return diagMatrix
end