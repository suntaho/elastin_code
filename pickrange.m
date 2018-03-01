% /// pick image range for correction of orientation and curvation
function [ibnd,jbnd] = pickrange(imgbuf,strbuf)
    figure(99); imagesc(imgbuf); box off; title(['Press and hold mouse on figure 99 to choose range for ' strbuf]);
    display(['Press and hold mouse on figure 99 to choose range for ' strbuf]);
    
    mm = waitforbuttonpress; 
    point1 = get(gca,'CurrentPoint');                                       % button down detected 
    finalRect = rbbox;                                                      % return figure units 
    point2 = get(gca,'CurrentPoint');                                       % button up detected 
    point1 = point1(1,1:2);                                                 % extract x and y 
    point2 = point2(1,1:2); 
    p1 = min(point1,point2);                                                % calculate locations 
    offset = abs(point1-point2);                                            % and dimensions 
    xrng = [p1(1) p1(1)+offset(1) p1(1)+offset(1) p1(1) p1(1)]; 
    yrng = [p1(2) p1(2) p1(2)+offset(2) p1(2)+offset(2) p1(2)];  
    % plot range
    if (p1(1)<1) p1(1) = 1; end
    if (p1(2)<1) p1(2) = 1; end
    if ( (p1(1)+offset(1))>size(imgbuf,2) ) offset(1) = size(imgbuf,2)-p1(1); end
    if ( (p1(2)+offset(2))>size(imgbuf,1) ) offset(2) = size(imgbuf,1)-p1(2); end
    hold on; axis manual; plot(xrng,yrng,'r'); 
    title(['processing...' strbuf]);
    display(['processing...' strbuf]); drawnow;
    % return values
    ibnd = round([ p1(1);p1(1)+offset(1) ]);
    jbnd = round([ p1(2);p1(2)+offset(2) ]);
    close(figure(99));