function [x, p, r] = cg(A, b, x, restart, iters, p, r)
    
    if restart == 1
        r = A * x - b;
        p = -r;
    end
    
    for i=1:iters
        Ap = A * p;
        old_r_sum = dot(r, r);
        if old_r_sum <= 1e-12
            break;
        end
        alpha = old_r_sum / (p' * Ap);

        x = x + alpha * p;
        r = r + alpha * Ap;

        be = (r' * r) / old_r_sum;
        p = -r + be * p;

    end
end