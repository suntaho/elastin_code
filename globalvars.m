% /// user-defined variables
function globalvars
    global ncpu; ncpu = 24;                                                     % number of cpus for parallel computations
    
    % images format
    global img_index_bgn; img_index_bgn = 2;                                    % index of begin and end (confocal) images
    global img_index_end; img_index_end = 100;                                  % index of begin and end (confocal) images
    global dxy; dxy = 0.25;                                                     % um, dimension of each pixel in a image
    global dz; dz = 0.5;                                                        % um, distance between two sequential images
    global bol_md; bol_md=1;                                         % 1: enable MD,  0: disable MD process
    
    % file name of images
    global fp_prefix;                                                           % prefix of file name
           fp_prefix = '12-02-2014 SHR #1 young_1_MSA SHR #1young vessel 1_C1_Z';
    global fp_ext; fp_ext = 'tif';                                              % extension of file name
    global fp_dig; fp_dig = 3;                                                  % digital number of file name, e.g. dig=3 for name 'prefix001.ext' and dig=5 for name 'prefix00001.ext'
    
    % orientation of vessel in 2-dim images
    global vessel_axis; vessel_axis = 'V';                                      % 'V' for vertical vessel axis | in each 2D image; 'H' for horizontal one --   
    
    % parameter for correctly determining vessel orientation and curvature
    global Inty_threshold_tilt; Inty_threshold_tilt = 70.0;                     % select these pixels with intensity > (Inty_threshold_tilt)% of maximum to depict orientation/tilt of vessel
    global Inty_threshold_bend; Inty_threshold_bend = 90.0;                     % select these pixels with intensity > (Inty_threshold_bend)% of maximum to depict curvature/bend of vessel
    global poly_cir_mix_fac; poly_cir_mix_fac = 1.0;                            % linear mix of poly-fit and circle-fit. =1 for circle; =0 for poly.
    
    % conversion from digital to binary
    global f_thrd; f_thrd = 0.3;                                               % convert digital into binary data by filter threshold = f_thrd*maximum
    
    % L-J potential for Molecular Dynamics
    global eps; eps = 30.0;                                                     % potential strength of L-J potential
    global sig; sig = 0.45;                                                      % equilibrium distance of L-J potential
    global temperature; temperature = 1.0;					% temperature for MD
    global Kmd; Kmd = 30.0;                                                     % elastic constant
    
