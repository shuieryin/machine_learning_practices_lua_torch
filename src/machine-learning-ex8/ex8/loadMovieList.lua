---
--- Generated by EmmyLua(https://github.com/EmmyLua)
--- Created by shuieryin.
--- DateTime: 12/01/2018 10:44 PM
---

function loadMovieList()
    --GETMOVIELIST reads the fixed movie list in movie.txt and returns a
    --cell array of the words
    --   movieList = GETMOVIELIST() reads the fixed movie list in movie.txt
    --   and returns a cell array of the words in movieList.


    -- Read the fixed movieulary list
    local movieList = {}
    for line in io.lines('movie_ids.txt') do
        movieList[#movieList + 1] = line:gsub("^%w+%s(.*)$", "%1")
    end

    return movieList
end