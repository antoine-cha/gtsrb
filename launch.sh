# Use to launch different exp during one night
th main.lua --lr 0.005 --m 0.9 --lrdecay 0.001 --weightdecay 0.005
th main.lua --lr 0.005 --m 0.8 --lrdecay 0.001 --weightdecay 0.005
th main.lua --lr 0.005 --m 0.8 --lrdecay 0.001 --weightdecay 0.001
th main.lua --lr 0.001 --m 0.9 --lrdecay 0.001 --weightdecay 0.001
th main.lua --lr 0.0005 --m 0.9 --lrdecay 0.001 --weightdecay 0.001
th main.lua --lr 0.0005 --m 0.9 --lrdecay 0.0001 --weightdecay 0.001



