require 'nn'
require 'optim'
require 'cunn'

-- define the dataset
local function getDataset(file, only2classes)
  -- Get the dataset object from the file.
  -- Format it if only 2 classes (change to 1, -1)
  -- Return a Tensor 
  -- -------------------------
  -- file (string)
  --  relative path to the t7 file 
  -- only2classes(bool)
  --  specify whether the dataset only contains 2 classes

  local dataset = torch.load(file)
  -- Modify dataset to get Tensors as classes
  if only2classes then
    for i, c in ipairs(dataset) do
      if dataset[i][2] == 2 then
        dataset[i][2] = torch.Tensor(1)
        dataset[i][2][1] = -1
      else
        dataset[i][2] = torch.Tensor(1)
        dataset[i][2][1] = 1
      end
    end
  end
  -- Needed by the algorithm
  function dataset:size() return #table[dataset] end
  return dataset
end


local function createNetwork(nb_classes)
  -- Define the network as in the paper
  local nb_classes = nb_classes or 43
  -- Enables to test the output sizes

  local mlp = nn.Sequential()
  kW = 3; kH=3;
  local conv1 = nn.SpatialConvolutionMM(3, 108, kW, kH)
  local subnor1 = nn.SpatialSubtractiveNormalization(108)
  local divnor1 = nn.SpatialDivisiveNormalization(108)
  local max1 = nn.SpatialMaxPooling(4, 4, 4, 4)
  -- Branching now
  -- Warning : Concat along 2nd axis for batch training !
  local ways = nn.Concat(2)
  -- First branch with 2 layers of convolutions
  local way1 = nn.Sequential()
  local conv2 = nn.SpatialConvolutionMM(108, 108, kW, kH)
  local subnor2 = nn.SpatialSubtractiveNormalization(108)
  local divnor2 = nn.SpatialDivisiveNormalization(108)
  local max2 = nn.SpatialMaxPooling(4, 4, 4, 4)
  local conv3 = nn.SpatialConvolutionMM(108, 50, 1, 1) 
  way1:add(conv2)
  way1:add(nn.ReLU())
  way1:add(subnor2)
  way1:add(divnor2)
  way1:add(max2)
  way1:add(conv3)
  way1:add(nn.ReLU())

  local way2 = nn.Sequential()
  way2:add(nn.SpatialConvolution(108,50,7,7))
  way2:add(nn.Reshape(50,1)) 
  ways:add(way1)
  ways:add(way2)


  mlp:add(conv1)
  mlp:add(nn.ReLU())
  mlp:add(subnor1)
  mlp:add(divnor1)
  mlp:add(max1)
  -- Branching 
  mlp:add(ways)

  mlp:add(nn.Reshape(100))

  mlp:add(nn.Linear(100,nb_classes))
  mlp:add(nn.SoftMax())
  
  return mlp
end



local function trainNetwork(network, dataset_file, params)
  -- Train a network given as input, with the given dataset and batchSize
  -- --------------------------
  -- Returns nothing :
  --  the network object is changed by the function
  -- --------------------------
  -- network (mlp)
  --  network to be trained
  -- dataset_file (path)
  --  relative path to the dataset file
  -- only2classes(bool)
  --  specifies the task to format the dataset
  -- batchSize(int)
  local dataset = getDataset(dataset_file, params.only2classes)
  dataset.data  = dataset.data:float()
  dataset.labels  = dataset.labels:float()
  local cudaOn = params.cudaOn
 
  ---- Training Criterion defined by the task
  if params.only2classes then
    criterion = nn.MSECriterion()
    print('MSE Criterion')
  else
    criterion = nn.CrossEntropyCriterion()
    print('Cross-entropy Criterion')
  end

  if cudaOn then
    criterion:cuda()
    network:cuda()
  end
  -- Some layers may have different behaviours during training
  network:training()
  local conf = optim.ConfusionMatrix(43)

  f, df = network:getParameters()
  local base_size_ = dataset.data[1]:size()
  local targets = torch.CudaTensor(params.batchSize)
  local inputs = torch.Tensor(params.batchSize, 
                base_size_[1], base_size_[2], base_size_[3])

  local indices = torch.range(1,
  dataset.data:size(1)):long():split(params.batchSize)
  -- remove last so that all batches have same size
  indices[#indices] = nil
  local _ind_ = 1

  -- Define the function to be optimized for SGD
  local feval = function(f_new)

    -- set x to x_new, if different
    -- (in this simple example, x_new will typically always point to x,
    -- so the copy is really useless)
    if f ~= f_new then
      f:copy(f_new)
    end
    -- reset gradients (gradients are always accumulated, to accomodate 
    -- batch methods)
    df:zero()
    --Get the indices of the next batch
    _ind_ = _ind_ + 1 
    if _ind_ == #indices then _ind_ = 1 end
    batch = indices[_ind_]

    local inputs = dataset.data:index(1, batch):cuda()
    targets:copy(dataset.labels:index(1, batch)):cuda()

    if cudaOn then
      inputs:cuda()
      targets:cuda()
    end

    local outputs = network:forward(inputs)
    --evaluate the loss function and its derivative wrt x,
    local loss_x = criterion:forward(outputs, targets)
    local df_do = criterion:backward(outputs, targets)
    network:backward(inputs, df_do)
    conf:batchAdd(outputs, targets)
    -- return loss(x) and dloss/dx
    return loss_x, df
  end


  -- Start training
  for i=1, params.nb_epochs do
    _, fs = optim.sgd(feval, f, params)
    local current_loss = fs[1] / params.batchSize
    conf:updateValids()
    io.write((i ..'th iteration, Current loss :' .. 
              ' %.6f // '):format(current_loss))
    io.write(('Train accuracy: '..'%.2f \n'):format(
            conf.totalValid * 100))
    conf:zero()
    if i % params.freq_save == 0 then 
      torch.save(params.filename, network)
      print('Model saved at ' .. params.filename)
    end
  end
end

return {getDataset = getDataset,
        createNetwork = createNetwork,
        trainNetwork = trainNetwork
      }
