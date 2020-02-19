
% f = @(t,x) [rand(1)*2*pi-pi; rand(1)*2*pi-pi];
f = @(t,x) [rand(1)*2-1; rand(1)*2-1];

vectfield(f, -2:.1:2, -2:.1:2)
