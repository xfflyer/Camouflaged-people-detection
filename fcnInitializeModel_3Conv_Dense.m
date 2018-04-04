function net = fcnInitializeModel_3Conv_Dense(varargin)
%FCNINITIALIZEMODEL8S Initialize the FCN-8S model from FCN-16S

opts.sourceModelUrl = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat' ;
opts.sourceModelPath = 'data/models/imagenet-vgg-verydeep-16.mat' ;
opts = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
%                    Load & download the source model if needed (VGG VD 16)
% -------------------------------------------------------------------------
if ~exist(opts.sourceModelPath)
  fprintf('%s: downloading %s\n', opts.sourceModelUrl) ;
  mkdir(fileparts(opts.sourceModelPath)) ;
  urlwrite('http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat', opts.sourceModelPath) ;
end
net = vl_simplenn_tidy(load(opts.sourceModelPath)) ;

% for convt (deconv) layers, cuDNN seems to be slower?
net.meta.cudnnOpts = {'cudnnworkspacelimit', 512 * 1024^3} ;
%net.meta.cudnnOpts = {'nocudnn'} ;

% -------------------------------------------------------------------------
%                                  Edit the model to create the FCN version
% -------------------------------------------------------------------------

% Add dropout to the fully-connected layers in the source model
drop1 = struct('name', 'dropout1', 'type', 'dropout', 'rate' , 0.5) ;
drop2 = struct('name', 'dropout2', 'type', 'dropout', 'rate' , 0.5) ;
net.layers = [net.layers(1:33) drop1 net.layers(34:35) drop2 net.layers(36:end)] ;

% Convert the model from SimpleNN to DagNN
net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;

% Add more padding to the input layer
%net.layers(1).block.pad = 100 ;
net.layers(5).block.pad = [0 1 0 1] ;
net.layers(10).block.pad = [0 1 0 1] ;
net.layers(17).block.pad = [0 1 0 1] ;
net.layers(24).block.pad = [0 1 0 1] ;
net.layers(31).block.pad = [0 1 0 1] ;
net.layers(32).block.pad = [3 3 3 3] ;
% ^-- we could do [2 3 2 3] but that would not use CuDNN

% Modify the bias learning rate for all layers
for i = 1:numel(net.layers)-1
  if (isa(net.layers(i).block, 'dagnn.Conv') && net.layers(i).block.hasBias)
    filt = net.getParamIndex(net.layers(i).params{1}) ;
    bias = net.getParamIndex(net.layers(i).params{2}) ;
    net.params(bias).learningRate = 2 * net.params(filt).learningRate ;
  end
end

% Modify the last fully-connected layer to have 21 output classes
% Initialize the new filters to zero
for i = [1 2]
  p = net.getParamIndex(net.layers(end-1).params{i}) ;
  if i == 1
    sz = size(net.params(p).value) ;
    sz(end) = 2 ;
  else
    sz = [2 1] ;
  end
  net.params(p).value = zeros(sz, 'single') ;
end
net.layers(end-1).block.size = size(...
  net.params(net.getParamIndex(net.layers(end-1).params{1})).value) ;

% Remove the last loss layer
net.removeLayer('prob') ;
net.setLayerOutputs('fc8', {'x38'}) ;


%% Add skip layers on top of pool5
net.addLayer('skip5', ...
     dagnn.Conv('size', [1 1 512 2], 'pad', 0), ...
     'x30', 'x39', {'skip5f','skip5b'});
f = net.getParamIndex('skip5f') ;
net.params(f).value = zeros(1, 1, 512, 2, 'single') ;
net.params(f).learningRate = 0.1 ;
net.params(f).weightDecay = 1 ;

f = net.getParamIndex('skip5b') ;
net.params(f).value = zeros(1, 1, 2, 'single') ;
net.params(f).learningRate = 2 ;
net.params(f).weightDecay = 1 ;

%% Add skip layers on top of pool4
net.addLayer('skip4', ...
     dagnn.Conv('size', [1 1 512 2], 'pad', 0), ...
     'x23', 'x40', {'skip4f','skip4b'});
 f = net.getParamIndex('skip4f') ;
net.params(f).value = zeros(1, 1, 512, 2, 'single') ;
net.params(f).learningRate = 0.1 ;
net.params(f).weightDecay = 1 ;

f = net.getParamIndex('skip4b') ;
net.params(f).value = zeros(1, 1, 2, 'single') ;
net.params(f).learningRate = 2 ;
net.params(f).weightDecay = 1 ;

%% Add skip layers on top of pool3
net.addLayer('skip3', ...
     dagnn.Conv('size', [1 1 256 2]), ...
     'x16', 'x41', {'skip3f','skip3b'});

f = net.getParamIndex('skip3f') ;
net.params(f).value = zeros(1, 1, 256, 2, 'single') ;
net.params(f).learningRate = 0.01 ;
net.params(f).weightDecay = 1 ;

f = net.getParamIndex('skip3b') ;
net.params(f).value = zeros(1, 1, 2, 'single') ;
net.params(f).learningRate = 2 ;
net.params(f).weightDecay = 1 ;


%%%%%%%%%C8反卷积%%%%%%
filters_8_f = single(bilinear_u(64, 1, 2)) ;
net.addLayer('deconv_c8_full', ...
  dagnn.ConvTranspose(...
  'size', size(filters_8_f), ...
  'upsample', 32, ...
  'crop', [16 16 16 16], ...
  'hasBias', false, ...
    'opts', net.meta.cudnnOpts), ...
  'x38', 'x42', 'deconv_c8_full') ;

f = net.getParamIndex('deconv_c8_full') ;
net.params(f).value = filters_8_f ;
net.params(f).learningRate = 0 ;
net.params(f).weightDecay = 1 ;

filters_8_5 = single(bilinear_u(4, 2, 2)) ;
net.addLayer('deconv_c8_dc5', ...
  dagnn.ConvTranspose(...
  'size', size(filters_8_5), ...
  'upsample', 2, ...
  'crop', 1, ...
  'numGroups', 2, ...
  'hasBias', false, ... 
   'opts', net.meta.cudnnOpts), ...
  'x38', 'x43', 'deconv_c8_dc5') ;

f = net.getParamIndex('deconv_c8_dc5') ;
net.params(f).value = filters_8_5 ;
net.params(f).learningRate = 0 ;
net.params(f).weightDecay = 1 ;

filters_8_4 = single(bilinear_u(8, 2, 2)) ;
net.addLayer('deconv_c8_dc4', ...
  dagnn.ConvTranspose(...
  'size', size(filters_8_4), ...
  'upsample', 4, ...
  'crop', 2, ...
  'numGroups', 2, ...
  'hasBias', false, ...
    'opts', net.meta.cudnnOpts), ...
  'x38', 'x44', 'deconv_c8_dc4') ;

f = net.getParamIndex('deconv_c8_dc4') ;
net.params(f).value = filters_8_4 ;
net.params(f).learningRate = 0 ;
net.params(f).weightDecay = 1 ;

filters_8_3 = single(bilinear_u(16, 2, 2)) ;
net.addLayer('deconv_c8_dc3', ...
  dagnn.ConvTranspose(...
  'size', size(filters_8_3), ...
  'upsample', 8, ...
  'crop', 4, ...
  'numGroups', 2, ...
  'hasBias', false, ...
    'opts', net.meta.cudnnOpts), ...
  'x38', 'x45', 'deconv_c8_dc3') ;

f = net.getParamIndex('deconv_c8_dc3') ;
net.params(f).value = filters_8_3 ;
net.params(f).learningRate = 0 ;
net.params(f).weightDecay = 1 ;

%%%%%%%%%C5反卷积%%%%%%
filters_5_4 = single(bilinear_u(4, 2, 2)) ;
net.addLayer('deconv_c5_dc4', ...
  dagnn.ConvTranspose(...
  'size', size(filters_5_4), ...
  'upsample', 2, ...
  'crop', 1, ...
  'numGroups', 2, ...
  'hasBias', false, ...
    'opts', net.meta.cudnnOpts), ...
  'x39', 'x46', 'deconv_c5_dc4') ;

f = net.getParamIndex('deconv_c5_dc4') ;
net.params(f).value = filters_5_4 ;
net.params(f).learningRate = 0 ;
net.params(f).weightDecay = 1 ;

filters_5_3 = single(bilinear_u(8, 2, 2)) ;
net.addLayer('deconv_c5_dc3', ...
  dagnn.ConvTranspose(...
  'size', size(filters_5_3), ...
  'upsample', 4, ...
  'crop', 2, ...
  'numGroups', 2, ...
  'hasBias', false, ...
    'opts', net.meta.cudnnOpts), ...
  'x39', 'x47', 'deconv_c5_dc3') ;

f = net.getParamIndex('deconv_c5_dc3') ;
net.params(f).value = filters_5_3 ;
net.params(f).learningRate = 0 ;
net.params(f).weightDecay = 1 ;

net.addLayer('sum1', dagnn.Sum(), {'x43', 'x39'}, 'x48') ;

filters_5_f = single(bilinear_u(32, 1, 2)) ;
net.addLayer('deconv_c5_full', ...
  dagnn.ConvTranspose(...
  'size', size(filters_5_f), ...
  'upsample', 16, ...
  'crop', [8 8 8 8], ...
  'hasBias', false, ...
    'opts', net.meta.cudnnOpts), ...
  'x48', 'x49', 'deconv_c5_full') ;

f = net.getParamIndex('deconv_c5_full') ;
net.params(f).value = filters_5_f ;
net.params(f).learningRate = 0 ;
net.params(f).weightDecay = 1 ;
%%%%%%%%%C4反卷积%%%%%%
filters_4_3 = single(bilinear_u(4, 2, 2)) ;
net.addLayer('deconv_c4_dc3', ...
  dagnn.ConvTranspose(...
  'size', size(filters_4_3), ...
  'upsample', 2, ...
  'crop', 1, ...
  'numGroups', 2, ...
  'hasBias', false, ...
    'opts', net.meta.cudnnOpts), ...
  'x40', 'x50', 'deconv_c4_dc3') ;

f = net.getParamIndex('deconv_c4_dc3') ;
net.params(f).value = filters_4_3 ;
net.params(f).learningRate = 0 ;
net.params(f).weightDecay = 1 ;

net.addLayer('sum2', dagnn.Sum(), {'x46', 'x44','x40'}, 'x51') ;

filters_4_f = single(bilinear_u(16, 1, 2)) ;
net.addLayer('deconv_c4_full', ...
  dagnn.ConvTranspose(...
  'size', size(filters_4_f), ...
  'upsample', 8, ...
  'crop', [4 4 4 4], ...
  'hasBias', false, ...
    'opts', net.meta.cudnnOpts), ...
  'x51', 'x52', 'deconv_c4_full') ;

f = net.getParamIndex('deconv_c4_full') ;
net.params(f).value = filters_4_f ;
net.params(f).learningRate = 0 ;
net.params(f).weightDecay = 1 ;

%%%%%%%%%C3卷积融合%%%%%%
net.addLayer('sum3', dagnn.Sum(), {'x50','x47','x45','x41'}, 'x53') ;

filters_3_f = single(bilinear_u(8, 1, 2)) ;
net.addLayer('deconv_c3_full', ...
  dagnn.ConvTranspose(...
  'size', size(filters_3_f), ...
  'upsample', 4, ...
  'crop', [2 2 2 2], ...
  'hasBias', false, ...
    'opts', net.meta.cudnnOpts), ...
  'x53', 'x54', 'deconv_c3_full') ;

f = net.getParamIndex('deconv_c3_full') ;
net.params(f).value = filters_3_f ;
net.params(f).learningRate = 0 ;
net.params(f).weightDecay = 1 ;

%%%%%%%%%C3-C8卷积融合%%%%%%
net.addLayer('sumdeconv', dagnn.Sum(), {'x54','x52','x49','x42'}, 'prediction') ;

net.addLayer('objective', ...
  SegmentationLoss('loss', 'softmaxlog'), ...
  {'prediction', 'label'}, 'objective') ;

% Add accuracy layer
net.addLayer('accuracy', ...
  SegmentationAccuracy(), ...
  {'prediction', 'label'}, 'accuracy') ;
% Make the output of the bilinear interpolator is not discared for
% visualization purposes
net.vars(net.getVarIndex('prediction')).precious = 1 ;

end
