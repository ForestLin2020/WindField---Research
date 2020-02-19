% Parameters
cLx=4;
cLz=2;
a=2*pi/(cLx*pi);
b=pi/2;
g=2*pi/(cLz*pi);
N8=(2*sqrt(2)/sqrt((a*a+g*g)*(4*a*a+4*g*g+pi*pi)));



%Re309
av_val=[0.268359169835888,0.0415737669588199,0.0373076787266736,0.0207721407214892,0.0624519067613835,0.102761062088787,-0.257139476000576,0.0726058071975180,-0.0812934255737902];

% Domain of mesh
% z=0:cLz*pi/20:cLz*pi;
% x=0:cLx*pi/20:cLx*pi;
% y=-1:0.1:1;


[x,y,z]=meshgrid(0:cLx*pi/20:cLx*pi,-1:0.1:1,0:cLz*pi/20:cLz*pi );

% Velocity Equations
ux=av_val(1)*sqrt(2)*sin(pi*y/2) + av_val(2)*(4/sqrt(3))*cos(pi*y/2).*cos(pi*y/2).*cos(g*z) + av_val(6)*4*sqrt(2)/(sqrt(3*(a^2+g*g)))*(-g)*cos(a*x).*cos(pi*y/2).*cos(pi*y/2).*sin(g*z) + av_val(7)*(2*sqrt(2)/(sqrt(a*a+g*g)))*g*sin(a*x).*sin(pi*y/2).*sin(g*z) + av_val(8)*N8*pi*a*sin(a*x).*sin(pi*y/2).*sin(g*z)+ av_val(9)*sqrt(2)*sin(3*pi*y/2);
uy=av_val(3)*(2/(sqrt(4*g*g+pi*pi)))*2*g*cos(pi*y/2).*cos(g*z)+ av_val(8)*N8*2*(a*a+g*g)*cos(a*x).*cos(pi*y/2).*sin(g*z);

uz=av_val(3)*(2/(sqrt(4*g*g+pi*pi)))*pi*sin(pi*y/2).*sin(g*z) + av_val(4)*(4/sqrt(3))*cos(pi*y/2).*cos(pi*y/2).*cos(a*x) + av_val(5)*2*sin(a*x).*sin(pi*y/2) + av_val(6)*(4*sqrt(2)/(sqrt(3*(a*a+g*g))))*a*sin(a*x).*cos(pi*y/2).*cos(pi*y/2).*cos(g*z) + av_val(7)*(2*sqrt(2)/(sqrt(a*a+g*g)))*a*cos(a*x).*sin(pi*y/2).*cos(g*z)- av_val(8)*N8*pi*g*cos(a*x).*sin(pi*y/2).*cos(g*z);


quiver3(x,y,z,ux,uy,uz)