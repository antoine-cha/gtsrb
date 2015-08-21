require 'nn'
require 'optim'
require 'cunn'
require 'sys'

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

local function createSimpleNetwork(nb_classes)
  local mlp = nn.Sequential()
  kW = 32; kH=32;
  local conv1 = nn.SpatialConvolutionMM(3, 43, kW, kH)
  mlp:add(conv1)
  mlp:add(nn.Reshape(43))
  mlp:add(nn.LogSoftMax())

  return mlp
end

local function createNetwork(nb_classes)
  local nb_classes = nb_classes or 43
  local x = torch.Tensor(1,3,32,32)
  -- Enables to test the output sizes

  local mlp = nn.Sequential()
  local units_1 = 32
  local units_2 = 32
  --1st layer
  local conv1 = nn.SpatialConvolutionMM(3, units_1, 3, 3)
  local divnor1 = nn.SpatialDivisiveNormalization(units_1)
  local subnor1 = nn.SpatialSubtractiveNormalization(units_1)
  local max1 = nn.SpatialMaxPooling(2, 2, 2, 2)
  mlp:add(conv1)
  mlp:add(nn.Tanh())
  mlp:add(nn.Abs())
  mlp:add(subnor1)
  mlp:add(divnor1)
  mlp:add(max1)
  print(mlp:forward(x):size())

  --2nd layer
  local conv2 = nn.SpatialConvolutionMM(units_1, units_2, 12, 12)
  local max2 = nn.SpatialMaxPooling(2, 2, 2, 2)
  local divnor2 = nn.SpatialDivisiveNormalization(units_2)
  local subnor2 = nn.SpatialSubtractiveNormalization(units_2)
  mlp:add(conv2)
  mlp:add(nn.Tanh())
  mlp:add(nn.Abs())
  mlp:add(subnor2)
  mlp:add(divnor2)
  mlp:add(max2)
  print(mlp:forward(x):size())

  mlp:add(nn.Reshape(4*units_2))
  mlp:add(nn.Linear(4*units_2, nb_classes))
  print(mlp:forward(x):size())
  mlp:add(nn.LogSoftMax())
  
  return mlp
end
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
  way2:add(nn.ReLU())
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

  mlp:add(nn.Linear(100,100))
  mlp:add(nn.ReLU())
  mlp:add(nn.Linear(100,50))
  mlp:add(nn.ReLU())
  mlp:add(nn.Linear(50,nb_classes))
  mlp:add(nn.SoftMax())
  
  return mlp
end



local function trainNetwork(network, params)
  -- Train a network given as input, with the given dataset and batchSize
  -- --------------------------
  -- Returns nothing :
  --  the network object is changed by the function
  -- --------------------------
  -- network (mlp)
  --  network to be trained
  -- params(table)
  --  contains the parameters
  local dataset = getDataset(params.dataPath, params.only2classes)
  dataset.data  = dataset.data:float()
  dataset.labels  = dataset.labels:float()
  local cudaOn = params.cudaOn
 
  ---- Training Criterion defined by the task
  if params.only2classes then
    criterion = nn.MSECriterion()
  else
    criterion = nn.ClassNLLCriterion()
  end
  print(criterion)

  if cudaOn then
    criterion:cuda()
    network:cuda()
  end
  local time = sys.clock()
  local time_0 = sys.clock()
  -- Some layers may have different behaviours during training
  network:training()
  local conf = optim.ConfusionMatrix(43)

  f, df = network:getParameters()
  local base_size_ = dataset.data[1]:size()
  local targets = torch.CudaTensor(params.batchSize)
  local inputs = torch.CudaTensor(params.batchSize, 
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

    inputs:copy(dataset.data:index(1, batch))
    targets:copy(dataset.labels:index(1, batch))

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
  local current_loss = 0.0
  local max_batches = params.nb_epochs * #indices
  for i=1, max_batches do
    _, fs = optim.sgd(feval, f, params)
    current_loss = current_loss + fs[1]
    conf:updateValids()
    if i%10 == 0 then
      current_loss = current_loss / params.batchSize
      io.write(('%.2f / ' .. params.nb_epochs.. ' epochs , Average loss :' .. 
                ' %.6f // '):format(i/#indices, current_loss))
      io.write(('Train acc: '..'%.2f '):format(
              conf.totalValid * 100))
      io.write(('t=%.1f s \n'):format(sys.clock() - time))
      time = sys.clock()
      current_loss = 0.0
      conf:zero()
    end
    if i % params.freq_save == 0 then 
      torch.save(params.filename, network)
      print('Model saved at ' .. params.filename)
    end
  end
  print(('Total experience time : %.1f s'):format(sys.clock() - time_0))
end

return {getDataset = getDataset,
        createNetwork = createNetwork,
        trainNetwork = trainNetwork,
        createSimpleNetwork = createSimpleNetwork,
        createMultiscaleNetwork = createMultiscaleNetwork
      }
