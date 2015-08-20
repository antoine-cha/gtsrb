--Custom imports
local learn = require 'learn'
local path = require 'pl.path'
local dir = require 'pl.dir'

local args = lapp [[
    Gets the data in the right format
    -r, --lr (number) learning rate
    -m, --momentum (number)
    --lrdecay  (number) learning rate decay
    --weightdecay (number) weight decay
    --model (string) model name
]]
local params = {
      orig_file = '',
      dataPath = './dataset-43c-allex.t7',
      modelDir = './models/',
      filename = args.model or 'model.t7',
      only2classes = false,
      cudaOn = true,
      nb_epochs = 10,
      freq_save = 500,
      -- SGD parameters
      batchSize = 100,
      learningRate = args.lr or 1,
      learningRateDecay = args.lrdecay or 1e-4,
      weightDecay = args.weightdecay or 0,
      momentum = args.momentum or 0.5,
      dampening = 0,
      nesterov = 1,
}

params.filename = path.join(params.modelDir, params.filename)
if not path.isdir(params.modelDir) then
  dir.makepath(params.modelDir)
end

if path.exists(params.filename) then
  local name, ext = path.splitext(params.filename) 
  local n = 1
  while path.exists((name .. '-' .. '%.2i' .. ext):format(n)) do
    print((name .. '%.2i' .. ext):format(n))
    n = n + 1
  end
  params.filename = (name .. '-' .. '%.2i' .. ext):format(n)
  params.paramsFile = (name .. '-' .. '%.2i' .. params .. ext):format(n)
end

local mlp_ = nn.Sequential()
if params.orig_file ~= '' then
  local network = torch.load(params.orig_file)
  print('Using ' .. params.orig_file .. ' as starting network')
  mlp_:add(network)
else
  local network = learn.createNetwork() 
  mlp_:add(network)
end

print(params)
learn.trainNetwork(mlp_, params)
torch.save(params.paramsFile,params)
torch.save(params.filename, mlp_)
