using LinearAlgebra;
include("function.jl");

x = ones(4,1);
dx = 1e-05;
tol = 1e-05;
imax = 100;
k = [0.15, 0.25, 0.35, 0.45];

x = newton_raphson(x, dx, k, imax, tol);
display(x);