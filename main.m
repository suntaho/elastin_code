% /// For re-construction of 3-dim(x,y,z) fiber network, using 2-dim(x,y) confocal images of centerized vessels.
% /// Enable parallel computations.
% ///.
% /// For skeletonizing code, we link to Philip's works by
% /// reference (1): Journal of Bone and Mineral Research, 28(8):1837-1845, (2013);  http://de.mathworks.com/matlabcentral/fileexchange/43527-skel2graph-3d
% /// reference (2): Computer Vision, Graphics, and Image Processing,  56(6):462-78, (1994);  http://de.mathworks.com/matlabcentral/fileexchange/43400-skeleton3d
% ///
% /// author: I-Lin Ho,  email: sunta.ho@msa.hinet.net


clear; clc;

% !!!!!! First modify user variables in globalvars.m
% === call user variables
globalvars;                                                                 % input user-defined variables


% === initialize parallel computation
global ncpu;                                                                % user-defined variables in globalvars.m
if (isempty(gcp('nocreate'))==0) delete(gcp); end                           % disable existed session
parpool(ncpu); tic;                                       


% === read and smooth intensities along sequential images
global fp_prefix; global fp_ext; global fp_dig;                             % user-defined variable in globalvars.m
global img_index_bgn; global img_index_end;
[img3d] = readimages(fp_prefix, fp_dig, fp_ext, img_index_bgn, img_index_end);                                                           


% === unbend and untile image for bended vessel
global vessel_axis; global poly_cir_mix_fac                                 % user-defined variable in globalvars.m
global Inty_threshold_tilt; global Inty_threshold_bend;
[img3d] = antidistort(img3d,vessel_axis,Inty_threshold_tilt,Inty_threshold_bend,poly_cir_mix_fac);


% === choose scope of image for fiber analysis
[ibnd,jbnd] = pickrange( sum(img3d,3),'analysis scope by top view' );
[kbnd,vbnd] = pickrange( reshape( sum(img3d,2),[size(img3d,1) size(img3d,3)] ), 'analysis depth by side view' ); 
img3d = img3d( jbnd(1):jbnd(2), ibnd(1):ibnd(2), kbnd(1):kbnd(2) ); clear ibnd jbnd kbnd vbnd;


% === isotropize 3d strucutre (by setting dxy=dz).
% === obatin binary data for particle and gaussian-filter data for potential (MD)
global f_thrd; global dxy; global dz;
[img3d,img3dg] = img2bit(img3d, dxy, dz, f_thrd,img_index_bgn);


% === use Molecular Dynamics to refine 3-dim. structure
% === command: 'mes md_f.cpp' to compile MD code for fast calculation
% === note some anti-virus softwares could block .mexw64 or relevant documents
global eps; global sig; global temperature; global Kmd; global bol_md;                  
if (bol_md==1)
    % mex COMPFLAGS="$COMPFLAGS /openmp" md_f.cpp; 					                         % use openmp with VS C++ 2010
    mex   CXXFLAGS="$CXXFLAGS -fopenmp" LDFLAGS="$LDFLAGS -fopenmp" md_f.cpp; 	% use openmp with linux gcc, double fast
    % mex md_f.cpp									% general OS system without OPENMD
    display('Read log.txt for MD process......');
    [img3d, xavg, yavg, zavg] = md_f( double(img3d), img3dg, ncpu, eps, sig, temperature,Kmd);
    img3d = uint8(img3d); 
end


% === analyz fiber
if(bol_md==1)
    task = analyze_fiber(img3d, img3dg, dxy, bol_md,xavg,yavg,zavg);
else
    task = analyze_fiber(img3d, img3dg, dxy, bol_md,1,1,1);
end


toc; delete(gcp);
