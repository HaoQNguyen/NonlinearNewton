using LinearAlgebra;
using Zygote;
include("function.jl");

x = ones(4,1);
dx = 1e-05;
tol = 1e-05;
imax = 100;

println(" ----- Implement finite difference -----\n");

k = [0.15, 0.25, 0.35, 0.45];
x = newton_raphson(x, dx, k, imax, tol);
display(x);

k = [0, 0, 0, 0];
x = newton_raphson(x, dx, k, imax, tol);
display(x);

println("\n ---------- Implement autodiff ----------\n");

x = ones(4,1);
ki = [1.5, 2.5, 3.5, 4.5];

k = [0.15, 0.25, 0.35, 0.45];
x = newton_raphson_autodiff(x, ki, imax, tol);
display(x);

k = [0, 0, 0, 0];
x = newton_raphson_autodiff(x, ki, imax, tol);
display(x);