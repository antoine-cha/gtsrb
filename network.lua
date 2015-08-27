require 'nn'
require 'cudnn'


local function SimpleNetwork(nb_classes)
  local mlp = nn.Sequential()
  kW = 32; kH=32;
  local conv1 = nn.SpatialConvolutionMM(3, 43, kW, kH)
  mlp:add(conv1)
  mlp:add(nn.Reshape(43))
  mlp:add(nn.LogSoftMax())
  return mlp
end


local function ParallelNetwork(nb_classes)
  local nb = 2
  local nfeats = 3
  kW = 4; kH=4;
  local mlp = nn.Sequential()
  local map_3 = torch.Tensor(1*nfeats, 2):zero()
  for i=1,map_3:size(1) do
    map_3[i][1] = 2
    map_3[i][2] = i
  end
  print(map_3)
  --print(nn.tables.full(3,4))

  local conv1 = nn.SpatialConvolutionMap(map_3, kW, kH)
  mlp:add(conv1)
  local x = torch.Tensor(nb, 3, kW, kH):zero()

  print('changing 1st channel-------------------')
  x[{{}, 1, {}, {}}]:fill(1)
  print(torch.all(torch.eq(x[1], x[2])))
  local y = mlp:forward(x)
  print(torch.all(torch.eq(y[1], y[2])))
  print(y:reshape(nb,nfeats))

  print('changing 2nd channel----------------')
  x[{{}, 2, {}, {}}]:fill(1)
  print(torch.all(torch.eq(x[1], x[2])))
  y = mlp:forward(x)
  print(y:reshape(nb,nfeats))
  print(torch.all(torch.eq(y[1], y[2])))

  print('changing 3rd channel---------------------')
  x[{ {}, 3, {}, {}}]:fill(4)
  print(torch.all(torch.eq(x[1], x[2])))
  y = mlp:forward(x)
  print(y:reshape(nb,nfeats))
  print(torch.all(torch.eq(y[1], y[2])))
  return mlp
end

local function MediumNetwork(nb_classes)
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


local function MultiscaleNetwork()
  local nb_classes = nb_classes or 43

  -- Enables to test the output sizes
  local x = torch.Tensor(1,3,32,32)

  local units_1 = 108
  local units_2 = 108
  local units_3 = 108
  local units_class_1 = 100
  local units_class_2 = 50

  local mlp = nn.Sequential()
  -- 1st layer
  local conv1 = cudnn.SpatialConvolution(3, units_1, 3, 3)
  local divnor1 = nn.SpatialDivisiveNormalization(units_1)
  local subnor1 = nn.SpatialSubtractiveNormalization(units_1)
  local max1 = cudnn.SpatialMaxPooling(2, 2, 2, 2)
  mlp:add(conv1)
  mlp:add(nn.Tanh())
  mlp:add(nn.Abs())
  mlp:add(divnor1)
  mlp:add(max1)
  --local y = mlp:forward(x)

  -- Branching now
  -- Warning : Concat along 2nd axis for batch training !
  local ways = nn.Concat(2)
  local way1 = nn.Sequential()
  local way2 = nn.Sequential()

  -- 1st branch with 2 layers of convolutions
  local conv2 = cudnn.SpatialConvolution(units_1, units_2, 3, 3)
  local subnor2 = nn.SpatialSubtractiveNormalization(units_2)
  local divnor2 = nn.SpatialDivisiveNormalization(units_2)
  local max2 = cudnn.SpatialMaxPooling(2, 2, 2, 2)
  way1:add(conv2)
  way1:add(nn.Tanh())
  way1:add(nn.Abs())
  way1:add(subnor2)
  way1:add(divnor2)
  way1:add(max2)

  local conv3 = cudnn.SpatialConvolution(units_2, units_class_1/2, 6, 6) 
  local subnor3 = nn.SpatialSubtractiveNormalization(units_class_1/2)
  local divnor3 = nn.SpatialDivisiveNormalization(units_class_1/2)
  way1:add(conv3)
  way1:add(nn.Tanh())
  way1:add(nn.Abs())

  -- 2nd branch : 1 conv
  local max_1 = cudnn.SpatialMaxPooling(2, 2, 2, 2)
  local conv_ = cudnn.SpatialConvolution(units_1, units_class_1/2, 7, 7) 
  local subnor_ = nn.SpatialSubtractiveNormalization(units_class_1/2)
  local divnor_ = nn.SpatialDivisiveNormalization(units_class_1/2)
  way2:add(max_1)
  way2:add(conv_)
  way2:add(nn.Tanh())
  way2:add(nn.Abs())

  ways:add(way1)
  ways:add(way2)
  mlp:add(ways)
  mlp:add(nn.Reshape(units_class_1))

  --Classifier
  mlp:add(nn.Linear(units_class_1, units_class_2))
  way2:add(nn.Tanh())
  way2:add(nn.Abs())
  mlp:add(nn.Linear(units_class_2, 43))
  mlp:add(nn.LogSoftMax())
  return mlp
end

return {SimpleNetwork = SimpleNetwork,
        MediumNetwork = MediumNetwork,
        MultiscaleNetwork = MultiscaleNetwork,
        ParallelNetwork = ParallelNetwork
      }
