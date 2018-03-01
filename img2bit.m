% /// convert digital image data into binary data with threshold = f_thrd*maximum
function [img3d,img3dg] = img2bit(img3d, dxy, dz, f_thrd,img_index_bgn)
    % isotropize structure by setting dxy=dz
    [x,y,z] = meshgrid(1:size(img3d,2),1:size(img3d,1),1:size(img3d,3));
    [xo,yo,zo] = meshgrid(1:size(img3d,2),1:size(img3d,1),1:dxy/dz:size(img3d,3));
    img3d = interp3(x,y,z,img3d,xo,yo,zo); 
    index = find(isnan(img3d)==1); img3d(index) = 0; clear x y z xo yo zo;
    
    
    % pad volume with zero to avoid edge effects;
    img3d = padarray(img3d,[5 5 5]);
    dimx = size(img3d,1); dimy = size(img3d,2); dimz = size(img3d,3);
    
    
    % guassian fiter on structure
    img3dg = smooth3(img3d,'gaussian',[5 5 5]);
    
    
    % find the maximum intensity 
    max_layer_inty = zeros(dimz,1);
    parfor nn = 1 : dimz
        max_layer_inty(nn,1) = max(max(img3d(:,:,nn)));
    end
    
    % convert image
    parfor nn = 1 : dimz
        img3dbuf = img3d(:,:,nn);
        p = find(img3dbuf >= round(f_thrd*max_layer_inty(nn,1)) & img3dbuf>1 );
        img3dbuf = 0*img3dbuf; img3dbuf(p) = 1;
        img3d(:,:,nn) = img3dbuf;
    end
    img3d = uint8(img3d); clear img3dbuf;
