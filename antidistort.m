% /// unbend and untile image for bended vessel
function [img3d] = antidistort(img3d,vessel_axis,Inty_threshold_tilt,Inty_threshold_bend,poly_cir_mix_fac)
    % identify orientation of images
    if (vessel_axis=='N')                                                   % rotate to V, following a back-rotation later
        imgbuf = zeros(size(img3d,2),size(img3d,1),size(img3d,3));
        parfor nn = 1:size(img3d,3)
            imgbuf(:,:,nn) = img3d(:,:,nn)';
        end
        img3d = imgbuf; clear imgbuf;
    end   
    dimx = size(img3d,1); dimy = size(img3d,2); dimz = size(img3d,3);
    
    
    % specify intersection image for fitting
    imgtilt_origin = reshape( img3d(:,round(0.5*dimy),:),[dimx dimz] );
    imgbend_origin = reshape( img3d(round(0.5*dimx),:,:),[dimy dimz] );
    
    
    % untilt vessel
    [ibnd1,jbnd1] = pickrange(imgtilt_origin,'untilting image');                              % pick image range for data fitting                
    imgtilt = imgtilt_origin( jbnd1(1):jbnd1(2), ibnd1(1):ibnd1(2) );
    [x y] = find(imgtilt > prctile(reshape(imgtilt,[size(imgtilt,1)*size(imgtilt,2) 1]),Inty_threshold_tilt) );
    x = x+jbnd1(1)-1; y = y+ibnd1(1)-1;                                     % back to the coordinate of full image
    p1 = polyfit(x,y,1);                                                    % 1-order polyfit for tilt
    [xi,yi,zi]=meshgrid(1:dimy, 1:dimx, 1:dimz);    zo=zi;                  % index of image pixels
    parfor nn = 1:dimz
        zo(:,:,nn)=zo(:,:,nn)+p1(1)*yi(:,:,nn);                             % correct z-index for untilt
    end
    img3d = interp3(xi,yi,zi,img3d,xi,yi,zo);                               % untilt image by 3-dim. interpolation 
    index = find(isnan(img3d)==1); img3d(index) = 0;
    imgtilt_corr = reshape( img3d(:,round(0.5*dimy),:),[dimx dimz] );
    % show numerical result for polyfitting
    yfit = p1(1)*x+p1(2);
    figure(2); plot(x,y,'.g',x,yfit,'.r'); xlabel(' 1st index of pixel'); ylabel(' 2nd index of pixel');
    legend(['pixel with intensity > ' num2str(Inty_threshold_tilt) '% of maximum'], 'fitting curve'); 
    title('[antidistor.m]: determine orientation of tilted vessel');
    clear xi yi zi zo;
    
    
    % unbend vessel 
    [ibnd2,jbnd2] = pickrange(imgbend_origin, 'unbending image');                              % pick image range for data fitting
    imgbend = imgbend_origin( jbnd2(1):jbnd2(2), ibnd2(1):ibnd2(2) );
    [x y] = find(imgbend > prctile(reshape(imgbend,[size(imgbend,1)*size(imgbend,2) 1]),Inty_threshold_bend) );
    x = x+jbnd2(1)-1; y = y+ibnd2(1)-1;                                     % back to the coordinate of full image
    % first fit to y=a*x^2+b*x+c
    p2 = polyfit(x,y,2);                                                    % 2-order polyfit for bend
    r1 = 1/2/(p2(1)); xc1 = -r1*p2(2); yc1 = p2(3)+r1-xc1^2/2/r1;
    yfit1 = p2(1)*x.^2+p2(2)*x+p2(3);
    % with given (xc,yc,r) guess, then fit into a circule
    % sl = lsqnonlin(@(f)(x-f(1)).^2+(y-f(2)).^2-f(3)^2,[xc yc r]); xc = sl(1); yc= sl(2); r = sl(3); % not stable
    s1 = [x y ones(size(x))]\[-(x.^2+y.^2)]; xc2 = -0.5*s1(1); yc2 = -0.5*s1(2); r2 =  sqrt((s1(1)^2+s1(2)^2)/4-s1(3));
    yfit2 = real(yc2-sqrt( r2^2-(x-xc2).^2 ));
    % mix poly-fit and circle-fit
    r = poly_cir_mix_fac*r2+(1-poly_cir_mix_fac)*r1; xc = poly_cir_mix_fac*xc2+(1-poly_cir_mix_fac)*xc1; yc = poly_cir_mix_fac*yc2+(1-poly_cir_mix_fac)*yc1;
    yfitm = real(yc-sqrt( r^2-(x-xc).^2 ));
    % correct curvature of vessel
    [xi,yi,zi] = meshgrid(1:size(img3d,2), 1:size(img3d,1), 1:size(img3d,3));
    xo = xi; zo = zi;
    parfor nn = 1:size(img3d,3)
        xo(:,:,nn)=xc+(yc-nn)*sin( (xo(:,:,nn)-xc)/(yc-nn) );
        zo(:,:,nn)=yc-(yc-nn)*cos( (xo(:,:,nn)-xc)/(yc-nn) );
    end
    img3d = interp3(xi,yi,zi,img3d,xo,yi,zo);                               % untilt image by 3-dim. interpolation
    index = find(isnan(img3d)==1); img3d(index) = 0;
    imgbend_corr = reshape( img3d(round(0.5*dimx),:,:),[dimy,dimz] );
    
    % show numerical result for fitting functions
    figure(3); plot(x,y,'.g',x,yfit1,'.r',x,yfit2,'.b',x,yfitm,'.k'); xlabel(' 1st index of pixel'); ylabel(' 2nd index of pixel');
    legend(['pixel with intensity > ' num2str(Inty_threshold_bend) '% of maximum'], 'poly-fit curve', 'circle-fit curve', 'mixed-fit curve'); 
    title('[antidistor.m]: determine curvature of bended vessel');
    clear xi yi zi xo zo theta;
    
    
    % compare images before and after correction
    figure(4);
    subplot(2,2,1); imagesc(imgtilt_origin); box off; title('[antidistor.m]: intersection of tilted vessel');
    subplot(2,2,2); imagesc(imgbend_origin); box off; title('[antidistor.m]: intersection of bended vessel');
    subplot(2,2,3); imagesc(imgtilt_corr); box off; title('[antidistor.m]: intersection of untilted vessel');
    subplot(2,2,4); imagesc(imgbend_corr); box off; title('[antidistor.m]: intersection of unbended vessel'); drawnow;
    
    
    % back-rotate if necessary
    if (vessel_axis=='N')                                                   % rotate to V, following a back-rotation later
        imgbuf = zeros(size(img3d,2),size(img3d,1),size(img3d,3));
        parfor nn = 1:size(img3d,3)
            imgbuf(:,:,nn) = img3d(:,:,nn)';
        end
        img3d = imgbuf; clear imgbuf;
    end   
    