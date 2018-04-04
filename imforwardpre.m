function [output,pre]=imforwardpre(imrgb,meanrgb,net,predVar)
    im = bsxfun(@minus, single(imrgb), (meanrgb)) ;
    h = size(imrgb,1) ;
    w = size(imrgb,2) ;
    inputdata=single(zeros(h,w,3,1));
    inputdata(:,:,:,1)=(im);
    inputVar = 'input' ;
    net.eval({inputVar,inputdata}) ;
    scores_ = gather(net.vars(predVar).value) ;%Ô¤²â
    [ss,pred_] = max(scores_,[],3) ;    
    output=pred_;
    pre = vl_nnsoftmax(scores_);
end