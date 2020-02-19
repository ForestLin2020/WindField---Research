x=repmat(1:10,10,1);
y=x';
deg=120; 
% all points have same direction
% if you have a degree array the same size as x, 
% you can use cosd and sind on "deg" without
% using repmat after
u=repmat(cosd(deg),size(x)); 
v=repmat(sind(deg),size(x));
% you can multiply u and v by magnitude of required
quiver(x,y,u,v);