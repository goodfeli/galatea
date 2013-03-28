function [imdisp] = dispims(imstack,drows,dcols,flip,border,n2,fud)
% [imdisp] = dispims(imstack,drows,dcols,flip,border,frame_rows)
%
% display a stack of images
% Originally written by Sam Roweis 


[pp,N] = size(imstack);
if(nargin<7) fud=0; end
if(nargin<6) n2=ceil(sqrt(N)); end

if(nargin<3) dcols=drows; end
if(nargin<4) flip=0; end
if(nargin<5) border=2; end

drb=drows+border;
dcb=dcols+border;

imdisp=min(imstack(:))+zeros(n2*drb,ceil(N/n2)*dcb);

for nn=1:N

  ii=rem(nn,n2); if(ii==0) ii=n2; end
  jj=ceil(nn/n2);

  if(flip)
    daimg = reshape(imstack(:,nn),dcols,drows)';
  else
    daimg = reshape(imstack(:,nn),drows,dcols);
  end

  imdisp(((ii-1)*drb+1):(ii*drb-border),((jj-1)*dcb+1):(jj*dcb-border))=daimg';

end

if(fud)
imdisp=flipud(imdisp);
end

imagesc(imdisp); colormap gray; axis equal; axis off;
drawnow;

