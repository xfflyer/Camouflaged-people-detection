function [ out ] = normalize( img )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
out=(img-min(img(:)))/(max(img(:))-min(img(:)));
%out=uint8(out.*255);

end

