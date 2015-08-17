require 'nn'


local mlp = nn.Concat(1)
local mlp2 = nn.Sequential()
local mlp3 = nn.Sequential()

mlp3:add(nn.Linear(10,5))
mlp2:add(nn.Linear(10,2))
mlp2:add(nn.Linear(2,1))


local x = torch.Tensor(10)

print(mlp2:forward(x):size())
print(mlp3:forward(x):size())

mlp:add(mlp2)
mlp:add(mlp3)
print(mlp:forward(x):size())
