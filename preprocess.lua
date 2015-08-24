require 'data'
require 'pl.path'

local args = lapp [[
    Gets the data in the right format
    <data_file> (string) set to be preprocessed
    ]]

print(args)
local name, ext = path.splitext(args.data_file)
print(name, ext)
local new_name = name .. 'prep' .. ext
print(new_name)
d = torch.load(args.data_file)
d = preprocess(d)
torch.save(new_name, d)
