clear all;clc;
run matconvnet/matlab/vl_setupnn ;
addpath ('matconvnet/examples') ;
addpath ('RegularFunction') ;
path1='.\models\DDCN-4C.mat';
path=path1;
snet = (load(path)) ;
snet2=snet.net;
net = dagnn.DagNN.loadobj(snet2) ;
predVar = net.getVarIndex('prediction') ;
inputVar = 'data' ;
rgbPath='E:\Camouflage Dataset\CamouflageData\img\';


semanicimg_savePath='result\DDCN\';
semspopt_savePath='result\Final\';
if ~exist(semanicimg_savePath, 'dir')
    mkdir(semanicimg_savePath);
end
if ~exist(semspopt_savePath, 'dir')
    mkdir(semspopt_savePath);
end

images = dir([rgbPath, '*.jpg']);
imagenum = length(images);
parfor i=1:imagenum
    
    imagename=[rgbPath,images(i).name];
    rgb = single(imread(imagename));    
    I=imread(imagename);
    h = size(rgb,1) ;
    w = size(rgb,2) ;
    pixNumInSP = 200;                           %pixels in each superpixel
    spnumber = round( h * w / pixNumInSP );     %super-pixel number for current image
    [idxImg, adjcMatrix, pixelList] = SLIC_Split(I, spnumber);

    spNum=max(idxImg(:));
    bdIds = GetBndPatchIds(idxImg);
    meanRgbCol = GetMeanColor(I, pixelList);
    meanLabCol = colorspace('Lab<-', double(meanRgbCol));
    meanPos = GetNormedMeanPos(pixelList, h, w);
    colDistM = GetDistanceMatrix(meanLabCol);
    posDistM = GetDistanceMatrix(meanPos);
    [clipVal, geoSigma, neiSigma] = EstimateDynamicParas(adjcMatrix, colDistM);
    [bgProb, bdCon, bgWeight] = EstimateBgProb(colDistM, adjcMatrix, bdIds, clipVal, geoSigma);
    
    
    rgb1=rgb(:,:,1);rgb2=rgb(:,:,2);rgb3=rgb(:,:,3);
    crgb1(:,:,1)=rgb1(1:384,1:384);
    crgb1(:,:,2)=rgb2(1:384,1:384);
    crgb1(:,:,3)=rgb3(1:384,1:384);    
    crgb2(:,:,1)=rgb1(1:384,284:284+384-1);
    crgb2(:,:,2)=rgb2(1:384,284:284+384-1);
    crgb2(:,:,3)=rgb3(1:384,284:284+384-1);
    crgb3(:,:,1)=rgb1(1:384,471:854);
    crgb3(:,:,2)=rgb2(1:384,471:854);
    crgb3(:,:,3)=rgb3(1:384,471:854);
    
    crgb4(:,:,1)=rgb1(97:480,1:384);
    crgb4(:,:,2)=rgb2(97:480,1:384);
    crgb4(:,:,3)=rgb3(97:480,1:384);    
    crgb5(:,:,1)=rgb1(97:480,284:284+384-1);
    crgb5(:,:,2)=rgb2(97:480,284:284+384-1);
    crgb5(:,:,3)=rgb3(97:480,284:284+384-1);
    crgb6(:,:,1)=rgb1(97:480,471:854);
    crgb6(:,:,2)=rgb2(97:480,471:854);
    crgb6(:,:,3)=rgb3(97:480,471:854);       

    rgbmean(1,1,1)=123.680;rgbmean(1,1,2)=116.779;rgbmean(1,1,3)=103.9390;
    net.mode = 'test' ;
   
    [output1,pre1]=imforwardpre(crgb1,rgbmean,net,predVar);
    output1(output1==1)=0; output1(output1==2)=1;
    
    [output2,pre2]=imforwardpre(crgb2,rgbmean,net,predVar);
    output2(output2==1)=0; output2(output2==2)=1;
    
    [output3,pre3]=imforwardpre(crgb3,rgbmean,net,predVar);
    output3(output3==1)=0; output3(output3==2)=1;
    
    [output4,pre4]=imforwardpre(crgb4,rgbmean,net,predVar);
    output4(output4==1)=0; output4(output4==2)=1;
    
    [output5,pre5]=imforwardpre(crgb5,rgbmean,net,predVar);
    output5(output5==1)=0; output5(output5==2)=1;
    
    [output6,pre6]=imforwardpre(crgb6,rgbmean,net,predVar);
    output6(output6==1)=0; output6(output6==2)=1;

    output=zeros(h,w);
    padcoutput1=zeros(h,w);padcoutput1(1:384,1:384)=output1;
    padcoutput2=zeros(h,w);padcoutput2(1:384,284:284+384-1)=output2;
    padcoutput3=zeros(h,w);padcoutput3(1:384,471:854)=output3;
    padcoutput4=zeros(h,w);padcoutput4(97:480,1:384)=output4;
    padcoutput5=zeros(h,w);padcoutput5(97:480,284:284+384-1)=output5;
    padcoutput6=zeros(h,w);padcoutput6(97:480,471:854)=output6;    
    output=padcoutput1+padcoutput2+padcoutput3+padcoutput4+padcoutput5+padcoutput6;
    output(output>=1)=1; output(output<1)=0;

    
    labelseg=bwlabel(output);
    semanticregion_pixel=cell(1,max(labelseg(:)));
    removeindex=[];
    for jj=1:max(labelseg(:))
        semanticregion_pixel{jj}=find(labelseg==jj);
        if length(semanticregion_pixel{jj})<200
            output(find(labelseg==jj))=0;
        end
    end
    labelseg2=bwlabel(output);uindex=unique(labelseg2);
    bdimg=zeros(h,w);bdimg(2:end-1,2:end-1)=1;bdimg=1-bdimg;
    bdcon=zeros(1,max(uindex));
    for jj=1:max(uindex)
        imgtem=(labelseg2==jj);
        len=sum(sum((imgtem.*bdimg)));
        area=sum(imgtem(:));
        bdcon(jj)=len/sqrt(area);
        if bdcon(jj)>0.9
            output(labelseg2==jj)=0;
        end
    end
    labelseg2=bwlabel(output);uindex=unique(labelseg2);
    if length(uindex)>1
        label=GetMeanColor(output, pixelList);
        label=normalize(label(:,1));
        spfgimg=zeros(h,w);
        spbgimg=zeros(h,w);
        for j=1:spNum
            spfgimg(find(idxImg==j))=label(j);
        end

        optwCtr=SaliencyOptimization(adjcMatrix, bdIds, colDistM, neiSigma, 1-label, label)
        salimg=zeros(h,w);
        for j=1:spNum
            salimg(find(idxImg==j))=optwCtr(j);
        end  
    else
        spfgimg=output;
        salimg=output;
    end
    
    savename=images(i).name;
    bdindex=find(savename=='.');
    eindex=bdindex(end);
    savename=savename(1:eindex-1);
    imwrite(mat2gray(output), [semanicimg_savePath,savename,'.png'], 'png');
    imwrite(mat2gray(salimg), [semspopt_savePath,savename,'.png'], 'png');    
end

