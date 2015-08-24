--Custom imports
local learn = require 'learn'
local path = require 'pl.path'
local dir = require 'pl.dir'

local args = lapp [[
    Gets the data in the right format
    -r, --lr (default 0.1) learning rate
    -m, --momentum (default 0.5)
    -b, --batch (default 64) batch size
    --lrdecay  (default 1e-4) learning rate decay
    --weightdecay (default 1e-4) weight decay
    --model (default model.t7) model name
    --retrain (default '') network to retrain
]]
local params = {
      retrain = args.retrain,
      dataPath = './dataset-43c-allex-3fa-prep.t7',
      modelDir = './models/',
      filename = args.model or 'model.t7',
      only2classes = false,
      cudaOn = true,
      nb_epochs = 7,
      freq_save = 500,
      -- SGD parameters
      batchSize = args.batch,
      learningRate = args.lr,
      learningRateDecay = args.lrdecay,
      weightDecay = args.weightdecay,
      momentum = args.momentum,
      --dampening = 0,
      --nesterov = 1,
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
  params.paramsFile = (name .. '-' .. '%.2i-params' .. ext):format(n)
end

local mlp_ = nn.Sequential()
if params.retrain ~= '' then
  local network = torch.load(params.retrain)
  print('Using ' .. params.retrain .. ' as starting network')
  mlp_:add(network)
  local params_file = ''
else
  local network = learn.createMultiscaleNetwork() 
  mlp_:add(network)
end

print(params)
learn.trainNetwork(mlp_, params)
torch.save(params.paramsFile,params)
torch.save(params.filename, mlp_)
