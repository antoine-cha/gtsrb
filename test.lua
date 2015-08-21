require 'optim'
require 'nn'
require 'image'
require 'cunn'
require 'cutorch'
require 'sys'
require 'xlua'
local lapp = require 'pl.lapp'


local args = lapp [[
    Test the given model
    <model> (string) path to the model
    <dataset> (string) path to the dataset
    -b, --batch (default 100) batch size
    -p, --preprocess (default 1.0) preprocessing
]]
local comp = require 'pl.comprehension' . new()
print(args)
if not args.model then
  print('Give the model path as argument')
  os.exit()
end
-- Get a network and test it 
function show_conf(network, dataset_file, batchSize, preprocess)
  -- Show the confusion matrix

  local dataset = torch.load(dataset_file)
  local conf = optim.ConfusionMatrix(43)
  conf:zero()

  local indices = torch.range(1,
      dataset.data:size(1)):long():split(batchSize)
  -- remove last so that all batches have same size
  indices[#indices] = nil
  local base_size_ = dataset.data[1]:size()
  local targets = torch.CudaTensor(batchSize)
  local preds = torch.CudaTensor(batchSize)
  local samples = torch.CudaTensor(batchSize, 
                base_size_[1], base_size_[2], base_size_[3])
  network = network:cuda()
  -- limit testing for large datasets
  local N = math.min(#indices, 1000)

  for i=1, N do
    xlua.progress(i, N)
    samples:copy(dataset.data:index(1,indices[i]))
    -- Do the same preprocessing as training data
    if preprocess then
      local mean_ = samples:mean(2):mean(3):mean(4)
      samples = samples - mean_:expand(samples:size())
    end
    preds = network:forward(samples)
    targets:copy(dataset.labels:index(1, indices[i]))
    conf:batchAdd(preds, targets)
  end

  conf:updateValids()
  io.write(('Test accuracy: '..'%.2f \n'):format(
            conf.totalValid * 100))
end


local network = torch.load(args.model)
print(torch.type(network))

print('Compute the confusion matrix')
show_conf(network, args.dataset, args.batch, args.preprocess)

