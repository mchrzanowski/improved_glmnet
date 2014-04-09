function z = glm(X, y, lambda, eta, z, iters)
    
    assert(eta > 0 && eta <= 1);
    assert(lambda > 0);
    
    XX = X' * X;
    [m, n] = size(XX);

    XX_I = XX + eye(m) * lambda * (1 - eta);
    K = [XX_I -XX; -XX XX_I];

    Xy = X' * y;
    g_start = [-Xy; Xy] + ones(size(Xy, 1) * 2, 1) * lambda * eta;
    
    A_prev = 0;
    g_A = 0;
    p = 0; r = 0;
    delz_A = zeros(2 * n, 1);
    K_A = 0;
    
    for i=1:iters
        g = g_start + K * z;
        nonpos_g = find(g <= 0);
        pos_z = find(z > 0);
        A = union(nonpos_g, pos_z);
        A_size = size(A, 1);
        
        if A_size == 0
            break;
        end
        
        if A_size == size(A_prev, 1) && size(setdiff(A, A_prev), 1) == 0
            [delz_A, p, r] = cg(K_A, g_A, delz_A, 0, 3, p, r);
        else
            K_A = K(A, A);
            delz_A = zeros(A_size, 1);
            g_A = -g(A);
            [delz_A, p, r] = cg(K_A, g_A, delz_A, 1, 3, p, r);
        end
        A_prev = A;
        
        if norm(g_A) <= 1e-14
            break;
        end
        
        delz = zeros(2 * n, 1);
        %sprintf('size of A: %d\tsize of delz_A: %d\n', A_size, size(delz_A, 1))
        delz(A) = delz_A;
        
        neg_delz = find(delz < 0);
        D = intersect(neg_delz, pos_z);
        if size(D, 1) == 0
            break;
        end
        
        alphas = -z(D) ./ delz(D);
        assert(min(alphas) > 0);
        alpha = min(min(alphas), 1);
                
        z = z + alpha * delz;
        z = bsxfun(@max, z, zeros(2 * n, 1));
        
    end
    
    sprintf('Iterations required: %d\n', i)
    
end