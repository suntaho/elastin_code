% /// read images and smooth intensities along sequential images
function [img3d,max_layer_inty] = readimages(fp_prefix, fp_dig, fp_ext, img_index_bgn, img_index_end)
    formatbuf=['%0' num2str(fp_dig) '.0f'];                                 % digital format in filename
    
    
    % define array for file names
    fp = [];
    for nn = img_index_bgn : img_index_end
        filename = [fp_prefix num2str(nn,formatbuf) '.' fp_ext];            % file name of first image
        fp = [fp ; filename];
    end
    

    % initialize array for 3d image
    imgbuf = imread(fp(1,:));
    dimx = size(imgbuf,1); dimy = size(imgbuf,2); dimz = img_index_end-img_index_bgn+1;
    img3d = zeros(dimx , dimy , dimz);
    clear imgbuf;
    
    
    % read image document
    parfor nn = 1 : dimz
        img3d(:,:,nn) = imread(fp(nn,:));
    end
    
    
    % find average values of pixel intensity > X% of maximum for each layer image
    avg_layer_inty = zeros(dimz,1); X=100;
    img3dbuf = zeros(dimx*dimy, dimz);
    parfor nn = 1 : dimz
        img3dbuf(:,nn) = sort( reshape(img3d(:,:,nn),[dimx*dimy 1]), 'descend');
        avg_layer_inty(nn) = 0;
        for mm = X:100
            avg_layer_inty(nn) = avg_layer_inty(nn) + prctile(img3dbuf(:,nn), mm)/(100-X+1.0);
        end
    end
    clear img3dbuf;
    
    
    % smooth avg_lay_inty function
    x = [1:dimz]'/dimz; fit_layer_inty = 0*avg_layer_inty;
    p = polyfit(x,avg_layer_inty,dimz);
    parfor nn = 1 : dimz
        fit_layer_inty(nn) = 0.0;
        for mm = 1 : dimz+1
            fit_layer_inty(nn) = fit_layer_inty(nn) + p(mm)*x(nn)^(dimz-mm+1);
        end
    end
    figure(1); plot(x*dimz+img_index_bgn-1, avg_layer_inty, 'r', x*dimz+img_index_bgn-1, fit_layer_inty, 'g'); 
    xlabel('layer index'); ylabel(['average of pixel intensity >=' num2str(X) '% of maximum']);
    legend('image value before untilt/unbend','fitting value before untilt/unbend');
    title('[readimages.m]: Intensities along sequential layer images');
    
    
    % re-scale img3d with factor fit_layer_inty/avg_layer_inty
    parfor nn = 1:dimz
        img3d(:,:,nn) = img3d(:,:,nn)*fit_layer_inty(nn,1)/avg_layer_inty(nn,1);
    end
    clear avg_layer_inty fit_layer_inty;

    
    
    