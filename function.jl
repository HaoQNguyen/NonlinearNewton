function f(x, k)
    
    L = 1.2;
    W = 0.25;
    LT = 25;
    k1 = 1.5;
    k2 = 2.5;
    k3 = 3.5;
    k4 = 4.5;

    y = zeros(4,1);
    y[1] = x[1] + x[2] + x[3] + x[4] + 4 * L + 3 * W - LT;
    y[2] = k1 * x[1] + k[1] * x[1]^3 - (k2 * x[2] + k[2] * x[2]^3);
    y[3] = k2 * x[2] + k[2] * x[2]^3 - (k3 * x[3] + k[3] * x[3]^3);
    y[4] = k3 * x[3] + k[3] * x[3]^3 - (k4 * x[4] + k[4] * x[4]^3);

    return y;

end

function jacobian!(x, dx, k, J)

    delx = copy(x);

    for i = 1:length(x)

        delx[i] -= dx;
        y = f(x,k);
        dely = f(delx,k);
        J[:,i] = (y - dely) / dx;
        delx[i] = x[i];

    end
    
end

function newton_raphson(x, dx, k, imax, tol)

    res = zeros(imax);
    con = zeros(imax);
    i = 1;
    res[i] = norm(f(x,k), Inf);
    J = zeros(length(x), length(x));
    jacobian!(x, dx, k, J);

    i = 2;
    while i < imax

        xold = copy(x);
        jacobian!(x, dx, k, J);
        x = xold - inv(J) * f(xold, k);

        res[i] = abs(norm(f(x,k), Inf));
        con[i] = abs(norm(f(x,k), Inf) - norm(f(xold,k), Inf));

        if res[i] < tol && con[i] < tol
            break;
        end

        i += 1;

    end

    return x;

end

# --------------- Fucntions related to autodiff implementation ---------------

f1(x, L, W, LT) = x[1] + x[2] + x[3] + x[4] + 4 * L + 3 * W - LT;
f2(x, ki, k) = ki[1] * x[1] + k[1] * x[1]^3 - (ki[2] * x[2] + k[2] * x[2]^3);
f3(x, ki, k) = ki[2] * x[2] + k[2] * x[2]^3 - (ki[3] * x[3] + k[3] * x[3]^3);
f4(x, ki, k) = ki[3] * x[3] + k[3] * x[3]^3 - (ki[4] * x[4] + k[4] * x[4]^3);

function f_autodiff(x, ki, k, L, W, LT)
    return [f1(x, L, W, LT);
            f2(x, ki, k);
            f3(x, ki, k);
            f4(x, ki, k)];
end

function jacobian_autodiff!(x, ki, k, L, W, LT)

    # sparse vector from the output tuple from gradient()
    (j1,) = gradient(x -> f1(x, L, W, LT), x);
    (j2,) = gradient(x -> f2(x, ki, k), x);
    (j3,) = gradient(x -> f3(x, ki, k), x);
    (j4,) = gradient(x -> f4(x, ki, k), x);

    # transpose vecors into column matrix
    j1 = transpose(j1);
    j2 = transpose(j2);
    j3 = transpose(j3);
    j4 = transpose(j4);

    # concatenate arrays vertically
    return [j1; j2; j3; j4];

end

function newton_raphson_autodiff(x, ki, imax, tol)

    L = 1.2;
    W = 0.25;
    LT = 25;

    res = zeros(imax);
    con = zeros(imax);
    i = 1;
    res[i] = norm(f_autodiff(x, ki, k, L, W, LT), Inf);
    J = zeros(length(x), length(x));
    J = jacobian_autodiff!(x, ki, k, L, W, LT);

    i = 2;
    while i < imax

        xold = copy(x);
        J = jacobian_autodiff!(x, ki, k, L, W, LT);
        x = xold - inv(J) * f_autodiff(xold, ki, k, L, W, LT)

        res[i] = norm(f_autodiff(x, ki, k, L, W, LT), Inf);
        con[i] = abs(norm(f_autodiff(x, ki, k, L, W, LT), Inf) - norm(f_autodiff(xold, ki, k, L, W, LT), Inf));

        if res[i] < tol && con[i] < tol
            break;
        end

        i += 1;

    end

    return x;

end