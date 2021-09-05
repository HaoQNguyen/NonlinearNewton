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

function jacobian(x, dx, k)

    J = ones(length(x), length(x));
    delx = copy(x);

    for i = 1:length(x)

        delx[i] .-= dx;
        y = f(x,k);
        dely = f(delx,k);
        J[:,i] = (y - dely) / dx;
        delx[i] = x[i];

    end

    return J;
    
end

function newton_raphson(x, dx, k, imax, tol)

    res = zeros(imax);
    con = zeros(imax);
    i = 1;
    res[i] = norm(f(x,k), -Inf);
    J = jacobian(x, dx, k);

    i = 2;
    while i < imax

        xold = copy(x);
        J = jacobian(x, dx, k);
        x = xold - inv(J) * f(xold, k);

        res[i] = norm(f(x,k), -Inf);
        con[i] = norm(f(x,k), -Inf) - norm(f(xold,k), -Inf);

        if res[i] < tol && con[i] < tol
            break;
        end

        i += 1;

    end

    return x;

end