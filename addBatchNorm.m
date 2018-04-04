function net = addBatchNorm(net, layerIndex)
%ADDBATCHNORM adds a batch norm layer
%    ADDBATCHNORM adds a batch layer to the network
%    to a network at the index given by `layerIndex`

% pair inputs and outputs to ensure a valid network
inputs = net.layers(layerIndex).outputs;

% find the number of channels produced by the previous layer
numChannels = net.layers(layerIndex).block.size(4);

outputs = sprintf('xbn%d',layerIndex);

% Define the name and parameters for the new layer
name = sprintf('bn%d', layerIndex);

block = dagnn.BatchNorm();
paramNames = {sprintf('%sm', name) ...
              sprintf('%sb', name) ...
              sprintf('%sx', name) };

% add new layer to the network          
net.addLayer(...
    name, ...
    block, ...
    inputs, ...
    outputs, ...
    paramNames) ;


params{1} = ones(numChannels,1,'single') ;
params{2} = ones(numChannels,1,'single') ;
params{3} = ones(numChannels,2,'single') ;
% params = cellfun(@gather, params, 'UniformOutput', false) ;    
 
% set mu (gain parameter)
mIdx = net.getParamIndex(paramNames{1});
net.params(mIdx).value = params{1};
net.params(mIdx).learningRate = 1;
net.params(mIdx).weightDecay = 1;
% 
% set beta (bias parameter)
bIdx = net.getParamIndex(paramNames{2});
net.params(bIdx).value = params{2};
net.params(bIdx).learningRate = 1;
net.params(bIdx).weightDecay = 1;

% set moments parameter
xIdx = net.getParamIndex(paramNames{3});
net.params(xIdx).value = params{3};
net.params(xIdx).learningRate = 0.1;
net.params(xIdx).weightDecay = 1;

% 
% % % modify the next layer to take the new inputs
% % net.layers(layerIndex + 1).inputs = {outputs};