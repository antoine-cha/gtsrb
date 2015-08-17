--[[ This script extracts images to tensors in .t7 files.
--  The images will be sorted by classes, 1 file per class
--
--]]

-- Load packages 
local image = require('image')
local csv = require('csv')
local path = require('pl.path')
local dir = require('pl.dir')
local comp = require 'pl.comprehension' . new()
print('Packages loaded')

-- Paths
local data_path = './train/Final_Training/Images/'
local img_path = '00000'
local meta_file = 'GT-00000.csv'

-- Functions
--
function shuffle(t)
    local n = #t
    while n >= 2 do
        --n is now the last pertinent index
        local k = math.random(n) -- 1 <= k <= n
        -- Quick swap
        t[n], t[k] = t[k], t[n]
        n = n - 1
    end
    return t
end

function csvToTable(meta_file)
    -- Returns a table (dict) of the metadata indexed by filenames
    -----------------------------------------
    -- meta_file (string)
    --      relative path to the file
    local f = csv.open(meta_file, {separator=";"})
    meta_data = {}
    for fields in f:lines() do
        -- Remove header
        if fields[1] ~= "Filename" then
            meta_data[fields[1]] = {width=fields[2],
                                height=fields[3],
                                x1=fields[4],
                                y1=fields[5],
                                x2=fields[6],
                                y2=fields[7]
                                }
        end
    end
    return meta_data
end



function imagesToTensorFiles(origDir, destDir, size)
    -- extract all the examples and write them per class
    -- The t7 files will be written in the current dir
    ------------------------------------
    -- origDir (string)
    --      relative path to the directory containing the image files
    -- destDir (string)
    --      relative path to the destination directory, will contain the t7 files
    -- size (int) / OPTIONAL - default=32
    --      the size to which the images are rescaled
    
    size = size or 32

    for root, dirs, files in dir.walk(origDir) do
        -- Only check one level of recursion for now
        
        if root == origDir then
            print(" Dirs :")
            print(table.getn(dirs))
            if (table.getn(dirs) ~= 0) then
                for i,f in ipairs(dirs) do
                    io.write(f .. ', ')
                end
                io.write('\n')
            end
            print(" Files :")
            print(table.getn(files))
            if (table.getn(files) ~= 0) then
                for i,f in ipairs(files) do
                    io.write(f .. ', ')
                end
                io.write('\n')
            end
        else
            -- Here we will process a given class
            local class_ = path.splitext(path.basename(root))
            local meta_file = 'GT-' .. class_ .. '.csv'
            local meta_file = path.join(origDir, meta_file)
            local meta_data = csvToTable(meta_file)
            local dataset = {}
            local c = tonumber(class_)
            print("Extracting class " .. c)

            for i, f in ipairs(files) do
                local file_  = path.join(root, f)
                -- load image
                local img = image.load(file_)
                img = image.scale(img, size..'x'..size)
                img = image.rgb2yuv(img)
                dataset[i] = img
            end
            local filename = path.join(destDir, "class_"..c..".t7")
            torch.save(filename, dataset)
            print(filename .. " saved")
        end
    end
    return dataset
end

function datasetFromClasses(origDir, ex_per_class, classes)
    -- Extract a given number of examples in the selected classes to create a dataset object.
    -- The dataset will be saved to a .t7 file. 
    -- WARNING : the classes are 0-indexed
    ---------------------------------------------------
    -- classes (table)
    --      array of the selected classes
    -- ex_per_class (int)
    --      number of examples to be extracted from each class
    -- origDir (string)
    --      path to the directory containing the examples as tensors, 1 file per class
    local classes = classes or comp 'x for x=0, 42' ()
    local ex_per_class = ex_per_class or 500
    local dataset = {}
    local current_i = 1

    for i_c, c in ipairs(classes) do
        --open file and take the first *ex_per_class* examples
        local examples = torch.load(path.join(origDir,  "class_"..c..".t7"))
        print(c, table.getn(examples))
        local nb = math.min(table.getn(examples), ex_per_class)
        for i=1, nb do
            dataset[current_i] = {}
            dataset[current_i][1] = examples[i] 
            dataset[current_i][2] = c
            current_i = current_i + 1 
        end
    end
    shuffle(dataset)
    return dataset
end

----------------------------------------------------------------------------
----------------------------------------------------------------------------
--metadata = csvToTable(data_path .. meta_file)
local orig = './train/Final_Training/Images'
local dest = './train/class_files'

extraction = false
creation = true
if extraction then
    dataset = imagesToTensorFiles(orig, dest, 32)
end

if creation then
    dataset = datasetFromClasses('./train/class_files', 500, {1,2,3})
    torch.save('1k2classes.t7', dataset)
end




