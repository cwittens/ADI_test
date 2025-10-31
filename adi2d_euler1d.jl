using Pkg
Pkg.activate(@__DIR__)
# Pkg.instantiate()

# Pkg.add("LinearAlgebra")
# Pkg.add("KernelAbstractions")
# Pkg.add("Trixi")
# Pkg.add("OrdinaryDiffEq")
# Pkg.add("Plots")
# Pkg.add("LaTeXStrings")
# Pkg.add("SparseArrays")

using LinearAlgebra
using KernelAbstractions
using Trixi
using OrdinaryDiffEq
using Plots
using SparseArrays
using LaTeXStrings


coordinates_min = (-1.0, -1.0, -1.0) # minimum coordinates (min(x), min(y), min(z))
coordinates_max = (1.0, 1.0, 1.0) # maximum coordinates (max(x), max(y), max(z))
dim = 2 # dim of ADI
gridx = range(coordinates_min[1], coordinates_max[1], length=21)
gridy = range(coordinates_min[2], coordinates_max[2], length=31)
gridz = range(coordinates_min[3], coordinates_max[3], length=41)


nx = length(gridx)
ny = length(gridy)
nz = length(gridz)
dt = 0.05

D = diffusivity() # defined in elixir_advection_diffusion.jl

function build_operator(grid, dt, D)
    nx = length(grid)
    dl = zeros(nx - 1)
    d = zeros(nx)
    du = zeros(nx - 1)

    for i in 2:nx-1  # interior points
        Δx_minus = grid[i] - grid[i-1]
        Δx_plus = grid[i+1] - grid[i]

        factor = (dt / dim) * 2 / (Δx_minus + Δx_plus)
        # factor = 2 / (Δx_minus + Δx_plus)

        D_minus = D
        D_center = D
        D_plus = D

        dl[i-1] = factor * 0.5 * (D_center + D_minus) / Δx_minus
        d[i] = -factor * (0.5 * (D_center + D_plus) / Δx_plus + 0.5 * (D_center + D_minus) / Δx_minus)
        du[i] = factor * 0.5 * (D_center + D_plus) / Δx_plus
    end

    # FIXME ccheck if factor of two is missing
    # Left boundary (i=1)
    Δx_plus = grid[2] - grid[1]
    D_center = D
    D_plus = D
    factor_left = (dt / dim) * 0.5 * (D_center + D_plus) / (Δx_plus)^2
    # factor_left =  0.5 * (D_center + D_plus) / (Δx_plus)^2

    d[1] = -factor_left
    du[1] = factor_left

    # Right boundary (i=nx)
    Δx_minus = grid[nx] - grid[nx-1]
    D_minus = D
    D_center = D
    factor_right = (dt / dim) * 0.5 * (D_center + D_minus) / (Δx_minus)^2
    # factor_right = 0.5 * (D_center + D_minus) / (Δx_minus)^2

    dl[nx-1] = factor_right
    d[nx] = -factor_right


    return Tridiagonal(dl, d, du)
end

A_1 = build_operator(gridx, 2, D)
A_2 = build_operator(gridy, 2, D)
A_3 = build_operator(gridz, 2, D)

function initial_condition_diffusion(x, y, z)
    # Store translated coordinate for easy use of exact solution
    x_shift = [x, y, z] .- [0.2, 0.2, 0.0]
    nu = diffusivity()
    c = 1
    scalar = c + exp(-4 * sum(abs2, x_shift))
    return scalar
end

U = [initial_condition_diffusion(x, y, z) for x in gridx, y in gridy, z in gridz] 
z_slice = 0.0
idz = findall(z -> isapprox(z, z_slice; atol=1e-8), gridz)[1]
heatmap(gridx, gridy, U[:, :, idz]', title="initial_condition at z = $(gridz[idz])", xlabel="x", ylabel="y",  size =(600, 500))




RHS = zero(U)
u_new = zero(U)   
tmpx = zero(U[:, 1, 1])
tmpy = zero(U[1, :, 1])
# ADI in (x-y) plane
T = 3.0
n_steps = Int(T / dt)
@time for step in 1:n_steps
    @show step


    # explizit euler in z direction for dt/2
    for i in 1:nx, j in 1:ny
        U[i, j, :] += dt/2 * A_3 * U[i, j, :]  
    end


    # ADI in x-y plane for dt:

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


    # explizit euler in z direction for dt/2
    for i in 1:nx, j in 1:ny
        U[i, j, :] += dt/2 * A_3 * U[i, j, :]  
    end



end

z_slice = 0.0
idz = findall(z -> isapprox(z, z_slice; atol=1e-8), gridz)[1]
heatmap(gridx, gridy, U[:, :, idz]', title="solution at T = $T and at z = $(gridz[idz])", xlabel=L"x", ylabel=L"y",  clim=(1,1.15), size =(600, 500))

pd = PlotData2D(sol[end], semi, slice=:xy, point=(0.0, 0.0, z_slice))
plot(pd, clim=(1,1.15), title = "Solution Trixi slice xy at z=$z_slice", size =(600, 500))