-- Allow inline Python/path text to wrap in the PDF without changing source text.
local escapes = {
  ["\\"] = "\\textbackslash{}", ["{"] = "\\{", ["}"] = "\\}",
  ["%"] = "\\%", ["$"] = "\\$", ["&"] = "\\&", ["#"] = "\\#",
  ["_"] = "\\_", ["^"] = "\\textasciicircum{}", ["~"] = "\\textasciitilde{}"
}

function Code(element)
  if FORMAT:match("latex") then
    local words = {}
    for word in element.text:gmatch("%S+") do
      local text = word:gsub("[\\{}%%$&#_^~]", function(char) return escapes[char] end)
      -- Keep short tokens intact so examples such as stocks and .all(axis=1)
      -- do not split awkwardly. Long paths/chains may break within a token.
      if #word > 35 then text = "\\seqsplit{" .. text .. "}" end
      table.insert(words, text)
    end
    return pandoc.RawInline("latex", "\\texttt{" .. table.concat(words, " ") .. "}")
  end
end
