m = 100; n = 100;
A = randn(m, n);
A = A + A';
A = A + n * eye(n);
b = randn(m, 1);
x_star = A \ b;
sprintf('Error: %g\n', norm(A * x_star - b, 2))

x = randn(n, 1);
r = A * x - b;
p = -r;
for i=1:10    
    Ap = A * p;
    old_r_sum = dot(r, r);
    alpha = old_r_sum / dot(p, Ap);

    x = x + alpha * p;
    r = r + alpha * Ap;

    be = dot(r, r) / old_r_sum;
    p = -r + be * p;

    sprintf('Iteration %d: Error: %g\n', i, norm(A * x - b, 2))

end