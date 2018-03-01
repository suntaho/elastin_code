function task = analyze_fiber(img3d, img3dg, dr, bol_md,xavg,yavg,zavg)
task=0;
linelng=10;             % to remove segment having length < linelng
img3d=uint8(ceil(img3d));
skel = Skeleton3D(img3d);                       % skeletonization; refer to http://de.mathworks.com/matlabcentral/fileexchange/43400-skeleton3d

figure(6);                                                          
% plot skeletonizing voxel
w=size(skel,1);
l=size(skel,2);
h=size(skel,3);
if(bol_md==1)
    index=find(skel(:));
    x=xavg(index); y=yavg(index); z=zavg(index);
else
    [x,y,z]=ind2sub([w,l,h],find(skel(:)));
end
r=2.0*ones(size(z,1),1);
c=(z-min(z))/(max(z)-min(z));
scatter(y,x,r,c,'o','filled','sizedata',12);       
set(gcf,'Color','white');
view([0 0 -1]);  axis equal;axis off;


%% define fiber segment; refer to http://de.mathworks.com/matlabcentral/fileexchange/43527-skel2graph-3d

% initial step: condense, convert to voxels and back, detect cells
[~,node,link] = Skel2Graph3D(skel,linelng);
% total length of network
wl = sum(cellfun('length',{node.links}));
skel2 = Graph2Skel3D(node,link,w,l,h);
[~,node2,link2] = Skel2Graph3D(skel2,linelng);
% calculate new total length of network
wl_new = sum(cellfun('length',{node2.links}));
% iterate the same steps until network length changed by less than 0.5%
while(wl_new~=wl)
    wl = wl_new;   
     skel2 = Graph2Skel3D(node2,link2,w,l,h);
     [A2,node2,link2] = Skel2Graph3D(skel2,linelng);
     wl_new = sum(cellfun('length',{node2.links}));
end;

% display result
figure(7); cmap = colormap('Jet');
imagesc( (sum(img3dg,3)/size(img3dg,3)).^0.5 ); box off; colormap gray;
hold on;
if(bol_md==1)
    for i=1:length(node2)
        x1 = node2(i).comx;
        y1 = node2(i).comy;
        z1 = node2(i).comz;
        for j=1:length(node2(i).links)    % draw all connections of each node 
            % draw edges as lines using voxel positions
            for k=1:length(link2(node2(i).links(j)).point)-1            
                [x3,y3,z3]=ind2sub([w,l,h],link2(node2(i).links(j)).point(k));
                [x2,y2,z2]=ind2sub([w,l,h],link2(node2(i).links(j)).point(k+1));
                x2avg=xavg(x2,y2,z2); y2avg=yavg(x2,y2,z2); z2avg=zavg(x2,y2,z2);
                x3avg=xavg(x3,y3,z3); y3avg=yavg(x3,y3,z3); z3avg=zavg(x3,y3,z3);
                zv=0.5*(z3avg+z2avg);
                zv=round(62*(zv-min(z))/(max(z)-min(z))+1);
                line([y3avg y2avg],[x3avg x2avg],'Color', cmap(zv,:),'LineWidth',1.5);
            end;
        end;
    end;
else
    for i=1:length(node2)
        x1 = node2(i).comx;
        y1 = node2(i).comy;
        z1 = node2(i).comz;
        for j=1:length(node2(i).links)    % draw all connections of each node 
            % draw edges as lines using voxel positions
            for k=1:length(link2(node2(i).links(j)).point)-1            
                [x3,y3,z3]=ind2sub([w,l,h],link2(node2(i).links(j)).point(k));
                [x2,y2,z2]=ind2sub([w,l,h],link2(node2(i).links(j)).point(k+1));
                zv=0.5*(z3+z2);
                zv=round(62*(zv-min(z))/(max(z)-min(z))+1);
                line([y3 y2],[x3 x2],'Color',cmap(zv,:),'LineWidth',1.5);
            end;
        end;
    end;
end
axis image;axis off; set(gcf,'Color','white');
axis equal; axis off;


%% save centerline into matrix

pntmax=-1;
segmax=0;
lngthrd=1;   % threshold of fiber length for filter, disable here due to the usage of linelng already

for i=1:length(node2)  
    segmax=segmax+length(node2(i).links);
    for j=1:length(node2(i).links)    % draw all connections of each node
        if( length(link2(node2(i).links(j)).point)>pntmax )
            pntmax=length(link2(node2(i).links(j)).point);
        end
    end;
end;

cnt_num_line_buf=zeros(segmax,1);       % record number of point for each segment/line
pnt_lines_buf=zeros(segmax,pntmax,3);   % record each point-position for each segment/line 

indx=0;
for i=1:length(node2)    
    for j=1:length(node2(i).links)    % record all connections of each node
        indx=indx+1;
        cnt_num_line_buf(indx,1)=length(link2(node2(i).links(j)).point);
        % record edges as lines using voxel positions
        for k=1:cnt_num_line_buf(indx,1)
            [x1,y1,z1]=ind2sub([w,l,h],link2(node2(i).links(j)).point(k));
            if(bol_md==1)
                xvv=xavg(x1,y1,z1); yvv=yavg(x1,y1,z1); zvv=zavg(x1,y1,z1);
                pnt_lines_buf(indx,k,1)=xvv; pnt_lines_buf(indx,k,2)=yvv; pnt_lines_buf(indx,k,3)=zvv;
            else
                pnt_lines_buf(indx,k,1)=x1; pnt_lines_buf(indx,k,2)=y1; pnt_lines_buf(indx,k,3)=z1;
            end
        end;
    end;
end;

ifrepeat=ones(segmax,1);
for i=1:segmax-1
    for j=i+1:segmax   
        if(cnt_num_line_buf(i,1)==cnt_num_line_buf(j,1))
            nn1=cnt_num_line_buf(i,1);
            if(ifrepeat(i)==1 && ifrepeat(j)==1 )
                sum1a=sum(  pnt_lines_buf(i,1:nn1,1)'-pnt_lines_buf(j,1:nn1,1)'  );
                if(sum1a==0)
                    sum2a=sum(  pnt_lines_buf(i,1:nn1,2)'-pnt_lines_buf(j,1:nn1,2)'  );
                    if(sum2a==0)
                        sum3a=sum(  pnt_lines_buf(i,1:nn1,3)'-pnt_lines_buf(j,1:nn1,3)'  );
                        if(sum3a==0)
                            ifrepeat(j)=0; break;
                        end
                    end
                end
                sum1b=sum(  pnt_lines_buf(i,1:nn1,1)'-flipud(pnt_lines_buf(j,1:nn1,1)')  );
                if(sum1b==0)
                    sum2b=sum(  pnt_lines_buf(i,1:nn1,2)'-flipud(pnt_lines_buf(j,1:nn1,2)')  );
                    if(sum2b==0)
                        sum3b=sum(  pnt_lines_buf(i,1:nn1,3)'-flipud(pnt_lines_buf(j,1:nn1,3)')  );
                        if(sum3b==0)
                            ifrepeat(j)=0; break;
                        end
                    end
                end
            end
        end
    end
end
segmax=sum(ifrepeat);               % update the number of segments
cnt_num_line_buf2=zeros(segmax,1);       % record number of point for each segment/line
pnt_lines_buf2=zeros(segmax,pntmax,3);   % record each point-position for each segment/line 

indx=0;
for i=1:size(ifrepeat,1)
    if(ifrepeat(i)==1)
        indx=indx+1;
        cnt_num_line_buf2(indx,1)=cnt_num_line_buf(i,1);
        pnt_lines_buf2(indx,:,2)=pnt_lines_buf(i,:,1)*dr;
        pnt_lines_buf2(indx,:,1)=pnt_lines_buf(i,:,2)*dr;
        pnt_lines_buf2(indx,:,3)=pnt_lines_buf(i,:,3)*dr;
    end
end

% filter out too-short segments
segmax=size(find(cnt_num_line_buf2>lngthrd));               % update the number of segments
segmax=segmax(1,1);
cnt_num_line=zeros(segmax,1);       % record number of point for each segment/line
pnt_lines=zeros(segmax,pntmax,3);   % record each point-position for each segment/line 
vbgn=zeros(segmax,1);
vend=zeros(segmax,1);

indx=0;
for i=1:size(cnt_num_line_buf2,1)
    if(cnt_num_line_buf2(i,1)>lngthrd)
        indx=indx+1;
        cnt_num_line(indx,1)=cnt_num_line_buf2(i,1);
        pnt_lines(indx,:,1)=pnt_lines_buf2(i,:,1);
        pnt_lines(indx,:,2)=pnt_lines_buf2(i,:,2);
        pnt_lines(indx,:,3)=pnt_lines_buf2(i,:,3);
    end
end

% build vbgn and vend
xsbuf=zeros(segmax,1); ysbuf=zeros(segmax,1); zsbuf=zeros(segmax,1);
xnbuf=zeros(segmax,1); ynbuf=zeros(segmax,1); znbuf=zeros(segmax,1);
for i=1:segmax
    xsbuf(i,1)=pnt_lines(i,1,1); ysbuf(i,1)=pnt_lines(i,1,2); zsbuf(i,1)=pnt_lines(i,1,3);
    nn=cnt_num_line(i,1);
    xnbuf(i,1)=pnt_lines(i,nn,1); ynbuf(i,1)=pnt_lines(i,nn,2); znbuf(i,1)=pnt_lines(i,nn,3);
end
for i=1:segmax
    ind1=find( xsbuf==xsbuf(i) & ysbuf==ysbuf(i) & zsbuf==zsbuf(i) );
    ind2=find( xnbuf==xsbuf(i) & ynbuf==ysbuf(i) & znbuf==zsbuf(i) );
    vbgn(i,1)=size(ind1,1)+size(ind2,1);
    ind1=find( xsbuf==xnbuf(i) & ysbuf==ynbuf(i) & zsbuf==znbuf(i) );
    ind2=find( xnbuf==xnbuf(i) & ynbuf==ynbuf(i) & znbuf==znbuf(i) );
    vend(i,1)=size(ind1,1)+size(ind2,1);
end
clear xsbuf ysbuf zsbuf xnbuf ynbuf znbuf ifrepeat;

maxX=-9999; maxY=-9999; maxZ=-9999;
minX=9999; minY=9999; minZ=9999;
for i=1:segmax
    if(maxX< max(max(pnt_lines(i,1:cnt_num_line(i,1),1))) )
        maxX=max(max(pnt_lines(i,1:cnt_num_line(i,1),1)));
    end
    if(minX> min(min(pnt_lines(i,1:cnt_num_line(i,1),1))) )
        minX=min(min(pnt_lines(i,1:cnt_num_line(i,1),1)));
    end
    if(maxY< max(max(pnt_lines(i,1:cnt_num_line(i,1),2))) )
        maxY=max(max(pnt_lines(i,1:cnt_num_line(i,1),2)));
    end
    if(minY> min(min(pnt_lines(i,1:cnt_num_line(i,1),2))) )
        minY=min(min(pnt_lines(i,1:cnt_num_line(i,1),2)));
    end
    if(maxZ< max(max(pnt_lines(i,1:cnt_num_line(i,1),3))) )
        maxZ=max(max(pnt_lines(i,1:cnt_num_line(i,1),3)));
    end
    if(minZ> min(min(pnt_lines(i,1:cnt_num_line(i,1),3))) )
        minZ=min(min(pnt_lines(i,1:cnt_num_line(i,1),3)));
    end
end


%% analyze length distribution

ni=size(pnt_lines,1);

ind=find(vbgn==1); vbgn(ind)=0;
ind=find(vend==1); vend(ind)=0;
% calculate striaght length for each segment
lines=pnt_lines;
cnt=cnt_num_line;
curveR=cnt;
lineR=cnt;
cntZ=cnt;
for i=1:size(cnt,1)
    dis=0.0;
    fi=cnt(i,1);
    for k = 1:fi-1
        x=lines(i,k:k+1,1);
        y=lines(i,k:k+1,2);
        z=lines(i,k:k+1,3);
        dis=dis+sqrt( (x(2)-x(1))^2+(y(2)-y(1))^2+(z(2)-z(1))^2 );
    end
    zavg=0.0;
    for k=1:fi
         zavg=zavg+lines(i,k,3)/fi;
    end
    cntZ(i,1)=zavg;
    curveR(i,1)=dis;
    lineR(i,1)=sqrt( (lines(i,fi,1)-lines(i,1,1))^2+(lines(i,fi,2)-lines(i,1,2))^2+(lines(i,fi,3)-lines(i,1,3))^2 );
end
% 
cx=zeros(ni,1); ctha=zeros(ni,1); cphi=zeros(ni,1); cntha=zeros(ni,1); cnphi=zeros(ni,1);
maxL=max(curveR); minL=min(curveR);
dL=(maxL-minL)/(ni-2); dZ=(maxZ-minZ)/(ni-2);
cx=linspace(minL,maxL,ni)'; cz=cntZ;
for i=1:size(cnt,1)
    [Err, N, P]=fit_3D_data(lines(i,1:cnt(i),1)',lines(i,1:cnt(i),2)',lines(i,1:cnt(i),3)','line','off','off');
    dx=N(1);
    dy=N(2);
    dz=N(3);
    tha=acos(abs(dz)/sqrt(dx^2+dy^2+dz^2))*180/pi;
    phi1=(atan2(dy,dx)*180/pi);
    phi2=(atan2(-dy,-dx)*180/pi);
    if(dx>0) 
        phi=phi2;
    else
        phi=phi1;
    end
    cntha(i)=cntha(i)+curveR(i);
    cnphi(i)=cnphi(i)+curveR(i);
    if(phi<0)
        phi=180+phi;
        tha=-tha;
    end
    ctha(i)=ctha(i)+(tha)*curveR(i);
    cphi(i)=cphi(i)+(phi)*curveR(i);
end
indtha=find(cntha~=0);
ctha(indtha)=ctha(indtha)./cntha(indtha);
indphi=find(cnphi~=0);
cphi(indphi)=cphi(indphi)./cnphi(indphi);


figure(8);
 plot(cz(indtha),ctha(indtha),'b.');   axis square; axis([0 35 -90 90]); box on;
 xlabel('fiber depth (um)'); ylabel('polar angle \theta (degree)');
figure(9);
 plot(cz(indphi),cphi(indphi),'b.');   axis square; axis([0 35 0 180]); box on;
  xlabel('fiber depth (um)'); ylabel('azimuthal angle \phi (degree)');
 
  task=1;

 