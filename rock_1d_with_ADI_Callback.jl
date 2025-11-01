coordinates_min = (-1.0, -1.0, -1.0) # minimum coordinates (min(x), min(y), min(z))
coordinates_max = (1.0, 1.0, 1.0) # maximum coordinates (max(x), max(y), max(z))
dim = 2 # dim of ADI
gridx = range(coordinates_min[1], coordinates_max[1], length=5)
gridy = range(coordinates_min[2], coordinates_max[2], length=5)
gridz = range(coordinates_min[3], coordinates_max[3], length=5)


nx = length(gridx)
ny = length(gridy)
nz = length(gridz)
dt = 2

D = 5.0e-2


@kernel function diffusion!(dϕ, @Const(ϕ), gridx, gridy, gridz, Nx, Ny, Nz, D, direction::Val{xyz}) where xyz
    i, j, k = @index(Global, NTuple)

    # FIXME how does dt come into play??!?

    if direction == Val(:x)
        idx_plus = (i == Nx) ? (i, j, k) : (i + 1, j, k)
        idx_minus = (i == 1) ? (i, j, k) : (i - 1, j, k)

        Δ_plus = (i == Nx) ? (gridx[i] - gridx[i-1]) : (gridx[i+1] - gridx[i])
        Δ_minus = (i == 1) ? Δ_plus : (gridx[i] - gridx[i-1])

        D_center = D
        D_plus = D
        D_minus = D

    elseif direction == Val(:y)
        idx_plus = (j == Ny) ? (i, j, k) : (i, j + 1, k)
        idx_minus = (j == 1) ? (i, j, k) : (i, j - 1, k)

        Δ_plus = (j == Ny) ? (gridy[j] - gridy[j-1]) : (gridy[j+1] - gridy[j])
        Δ_minus = (j == 1) ? Δ_plus : (gridy[j] - gridy[j-1])

        D_center = D
        D_plus = D
        D_minus = D

    elseif direction == Val(:z)
        idx_plus = (k == Nz) ? (i, j, k) : (i, j, k + 1)
        idx_minus = (k == 1) ? (i, j, k) : (i, j, k - 1)

        Δ_plus = (k == Nz) ? (gridz[k] - gridz[k-1]) : (gridz[k+1] - gridz[k])
        Δ_minus = (k == 1) ? Δ_plus : (gridz[k] - gridz[k-1])

        D_center = D
        D_plus = D
        D_minus = D

    else
        error("Invalid direction chosen")
    end

    half = eltype(ϕ)(0.5)

    # Compute 1D diffusion
    @inbounds dϕ[i, j, k] = @fastmath (
        (D_plus + D_center) * half * (ϕ[idx_plus...] - ϕ[i, j, k]) / Δ_plus
        -
        (D_center + D_minus) * half * (ϕ[i, j, k] - ϕ[idx_minus...]) / Δ_minus
    ) / ((Δ_plus + Δ_minus) * half) #/ rho_c


end

U = [initial_condition_diffusion(x, y, z) for x in gridx, y in gridy, z in gridz]
A_1 = build_operator(gridx, 2, D)
A_2 = build_operator(gridy, 2, D)
A_3 = build_operator(gridz, 2, D)


backend = CPU()
backend = ROCBackend()

U = adapt(backend, U)
gridx = adapt(backend, gridx)
gridy = adapt(backend, gridy)
gridz = adapt(backend, gridz)
A_1 = adapt(backend, A_1)
A_2 = adapt(backend, A_2)
A_3 = adapt(backend, A_3)

RHS = zero(U)
RHS2 = zero(U)
using GPUArraysCore: @allowscalar
# X direction implicit, Y direction explicit for dt/2
@time for i in 1:nx, k in 1:nz
    @allowscalar RHS[i, :, k] = A_2 * U[i, :, k]
end

@time diffusion!(backend)(RHS2, U, gridx, gridy, gridz, nx, ny, nz, D, Val(:y), ndrange=(nx, ny, nz))

RHS .- RHS2

@time for j in 1:ny, k in 1:nz
    RHS[:, j, k] = (A_1) * U[:, j, k]
end

@time diffusion!(backend)(RHS2, U, gridx, gridy, gridz, nx, ny, nz, D, Val(:x), ndrange=(nx, ny, nz))

RHS .- RHS2

@time for i in 1:nx, j in 1:ny
    RHS[i, j, :] = A_3 * U[i, j, :]
end

@time diffusion!(CPU())(RHS2, U, gridx, gridy, gridz, nx, ny, nz, D, Val(:z), ndrange=(nx, ny, nz))

RHS .- RHS2



@kernel function thomas(U, RHS, gridx, gridy, gridz, ::Val{N}, dt, D, direction::Val{xy}) where {N,xy}
    ij, k = @index(Global, NTuple)

    Float_used = eltype(RHS)
    half = Float_used(0.5)

    # A_1 + I instead of A_1 only
    one = Float_used(1.0)


    # private memory
    # FIXME right memory
    b = @private Float_used (N,) # Ax = b <- this b
    lower = @private Float_used (N,) # subdiagonal of A / solution vector x later
    diagonal = @private Float_used (N,) # main diagonal of A
    upper = @private Float_used (N - 1,) # superdiagonal of A


    if direction == Val(:x)
        grid = gridx
        # load RHS into b
        for l in 1:N
            b[l] = RHS[l, ij, k]
        end

    elseif direction == Val(:y)
        grid = gridy
        for l in 1:N
            b[l] = RHS[ij, l, k]
        end
    else
        error("Invalid direction chosen")
    end



    # build A_1 FIXME A_1 + I
    # Left boundary (i=1)
    Δ_plus = grid[2] - grid[1]
    D_center = D
    D_plus = D
    factor_left = (dt / 2) * half * (D_center + D_plus) / (Δ_plus)^2

    diagonal[1] = -factor_left + one
    upper[1] = factor_left

    for l in 2:N-1
        Δ_minus = grid[l] - grid[l-1]
        Δ_plus = grid[l+1] - grid[l]

        factor = (dt / 2) * 2 / (Δ_minus + Δ_plus)

        D_minus = D
        D_center = D
        D_plus = D

        lower[l-1] = factor * half * (D_center + D_minus) / Δ_minus
        # FIXME + I
        diagonal[l] = -factor * (half * (D_center + D_plus) / Δ_plus + half * (D_center + D_minus) / Δ_minus) + one
        upper[l] = factor * half * (D_center + D_plus) / Δ_plus
    end

    # Right boundary (i=N)
    Δ_minus = grid[N] - grid[N-1]
    D_minus = D
    D_center = D
    factor_right = (dt / 2) * half * (D_center + D_minus) / (Δ_minus)^2

    lower[N-1] = factor_right
    diagonal[N] = -factor_right + one

    # not let it be undefined (TODO: NOT needed i think)
    # lower[N] = 0

    for l in 2:N
        w = lower[l-1] / diagonal[l-1]
        diagonal[l] -= w * upper[l-1]
        b[l] -= w * b[l-1]
    end

    # 'lower' is now the cache for the solution vector
    lower[N] = b[N] / diagonal[N]
    for l in (N-1):-1:1
        lower[l] = (b[l] - upper[l] * lower[l+1]) / diagonal[l]
    end


    # copy solution back to U
    if direction == Val(:x)
        for l in 1:N
            U[l, ij, k] = lower[l]
        end

    else # direction == Val(:y)
        for l in 1:N
            U[ij, l, k] = lower[l]
        end
    end
end

# U = [initial_condition_diffusion(x, y, z) for x in gridx, y in gridy, z in gridz]

u_new = zero(U)
u_new2 = zero(U)
u_new3 = zero(U)

for j in 1:ny, k in 1:nz
    u_new[:, j, k] = (I + A_1) \ U[:, j, k]
end



thomas(backend)(u_new, u_new2, gridx, gridy, gridz, Val(nx), 2, D, Val(:x), ndrange=(ny, nz))





u_new

sum(abs, u_new2 .- u_new) / length(u_new)









thomas_x(backend)(u_new3, U, gridx, gridy, gridz, nx, ny, nz, 2, D, ndrange=(ny, nz))




@kernel function thomas_x(U, @Const(RHS), gridx, gridy, gridz, Nx, Ny, Nz, dt, D)
    j, k = @index(Global, NTuple)


    grid = gridx
    N = Nx

    Float_used = eltype(RHS)
    half = Float_used(0.5)

    # A_1 + I instead of A_1 only
    one = Float_used(1.0)
    # otherwise set one = 0 (lol)

    # private memory
    lower = @private Float_used (21,)
    diagonal = @private Float_used (21,)
    upper = @private Float_used (21 - 1,)
    b = @private Float_used (21,)

    # load RHS into b
    for i in 1:N
        b[i] = RHS[i, j, k]
    end




    # build A_1 FIXME A_1 + I
    # Left boundary (i=1)
    Δ_plus = grid[2] - grid[1]
    D_center = D
    D_plus = D
    factor_left = (dt / 2) * half * (D_center + D_plus) / (Δ_plus)^2

    diagonal[1] = -factor_left + one
    upper[1] = factor_left

    for i in 2:N-1
        Δ_minus = grid[i] - grid[i-1]
        Δ_plus = grid[i+1] - grid[i]

        factor = (dt / 2) * 2 / (Δ_minus + Δ_plus)

        D_minus = D
        D_center = D
        D_plus = D

        lower[i-1] = factor * half * (D_center + D_minus) / Δ_minus
        # FIXME + I
        diagonal[i] = -factor * (half * (D_center + D_plus) / Δ_plus + half * (D_center + D_minus) / Δ_minus) + one
        upper[i] = factor * half * (D_center + D_plus) / Δ_plus
    end

    # Right boundary (i=N)
    Δ_minus = grid[N] - grid[N-1]
    D_minus = D
    D_center = D
    factor_right = (dt / 2) * half * (D_center + D_minus) / (Δ_minus)^2

    lower[N-1] = factor_right
    diagonal[N] = -factor_right + one

    # not let it be undefined (TODO: NOT needed i think)
    lower[N] = 0

    for i in 2:N
        w = lower[i-1] / diagonal[i-1]
        diagonal[i] -= w * upper[i-1]
        b[i] -= w * b[i-1]
    end

    # 'lower' is now the cache for the solution vector
    lower[N] = b[N] / diagonal[N]
    for i in (N-1):-1:1
        lower[i] = (b[i] - upper[i] * lower[i+1]) / diagonal[i]
    end

    # copy solution back to U
    for i in 1:N
        U[i, j, k] = lower[i]
    end

end












@time for i in 1:nx, k in 1:nz
    A_2 = build_operator(gridy, 2, D)
    u_new[i, :, k] = (I + A_2) \ U[i, :, k]
end

@time thomas(CPU())(u_new3, U, gridx, gridy, gridz, ny, 2, D, Val(:y), ndrange=(nx, nz))

sum(abs, u_new3 .- u_new) / length(u_new)



function ADI_xy()

    # X direction implicit, Y direction explicit for dt/2
    for i in 1:nx, k in 1:nz
        RHS[i, :, k] = (I + A_2) * U[i, :, k]
    end

    for j in 1:ny, k in 1:nz
        u_new[:, j, k] = (I - A_1) \ RHS[:, j, k]
    end

    # Y direction implicit, X direction explicit for dt/2
    for j in 1:ny, k in 1:nz
        RHS[:, j, k] = (I + A_1) * u_new[:, j, k]
    end

    for i in 1:nx, k in 1:nz
        U[i, :, k] = (I - A_2) \ RHS[i, :, k]
    end


end



function ADI_xy_pseudo_code()

    nx ≈ 50
    ny ≈ 50
    nz ≈ 2000
    size(U) = (nx, ny, nz) # should I reorder so biggest comes first?
    A_i = "Tridiagonal Matrix"

    # X direction implicit, Y direction explicit for dt/2
    for i in 1:nx, k in 1:nz
        A_2 = build_operator_y(gridx[i], gridz[k], t)
        RHS[i, :, k] = (I + A_2) * U[i, :, k]
    end

    for j in 1:ny, k in 1:nz
        A_1 = build_operator_x(gridy[j], gridz[k], t)
        U[:, j, k] = (I - A_1) \ RHS[:, j, k]
    end

    t = t + dt / 2

    # Y direction implicit, X direction explicit for dt/2
    for j in 1:ny, k in 1:nz
        A_1 = build_operator_x(gridy[j], gridz[k], t)
        RHS[:, j, k] = (I + A_1) * U[:, j, k]
    end

    for i in 1:nx, k in 1:nz
        A_2 = build_operator_y(gridx[i], gridz[k], t)
        U[i, :, k] = (I - A_2) \ RHS[i, :, k]
    end

end