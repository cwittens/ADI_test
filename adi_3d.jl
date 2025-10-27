# using Pkg
# Pkg.activate(@__DIR__)
# Pkg.instantiate()

# Pkg.add("LinearAlgebra")
# Pkg.add("KernelAbstractions")
# Pkg.add("Trixi")
# Pkg.add("OrdinaryDiffEqTsit5")
# Pkg.add("Plots")
# Pkg.add("LaTeXStrings")
# Pkg.add("SparseArrays")

using LinearAlgebra
using KernelAbstractions
using Trixi
using OrdinaryDiffEqTsit5
using Plots
using SparseArrays
using LaTeXStrings

function create_adaptive_grid_N(x_start, x_transition1, x_transition2, x_end,
    N1, N2, N3, backend, Float_used)
    # Three segments
    grid1 = range(x_start, x_transition1, length=N1)
    grid2 = range(x_transition1, x_transition2, length=N2)
    grid3 = range(x_transition2, x_end, length=N3)
    println("dx_1 = ", step(grid1))
    println("dx_2 = ", step(grid2))
    println("dx_3 = ", step(grid3))

    # Concatenate (watch out for duplicate points at boundaries!)
    grid = vcat(grid1[1:end-1], grid2[1:end], grid3[2:end])
    grid = Float_used.(grid)
    # grid = adapt(backend, grid)
    return grid
end


gridx = create_adaptive_grid_N(coordinates_min[1], -0.1, 0.7, coordinates_max[1], 25, 30, 20, CPU(), Float64);
gridy = create_adaptive_grid_N(coordinates_min[2], -0.1, 0.7, coordinates_max[2], 40, 60, 30, CPU(), Float64);
gridz = create_adaptive_grid_N(coordinates_min[3], -0.1, 0.7, coordinates_max[3], 40, 60, 30, CPU(), Float64);



coordinates_min = (-1.0, -1.0, -1.0) # minimum coordinates (min(x), min(y), min(z))
coordinates_max = (1.0, 1.0, 1.0) # maximum coordinates (max(x), max(y), max(z))
dim = 3
gridx = range(coordinates_min[1], coordinates_max[1], length=101)
gridy = range(coordinates_min[2], coordinates_max[2], length=91)
gridz = range(coordinates_min[3], coordinates_max[3], length=81)


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

        D_minus = D
        D_center = D
        D_plus = D

        dl[i-1] = factor * 0.5 * (D_center + D_minus) / Δx_minus
        d[i] = -factor * (0.5 * (D_center + D_plus) / Δx_plus + 0.5 * (D_center + D_minus) / Δx_minus)
        du[i] = factor * 0.5 * (D_center + D_plus) / Δx_plus
    end

    # Left boundary (i=1)
    Δx_plus = grid[2] - grid[1]
    D_center = D
    D_plus = D
    factor_left = (dt / dim) * 0.5 * (D_center + D_plus) / (Δx_plus)^2

    d[1] = -factor_left
    du[1] = factor_left

    # Right boundary (i=nx)
    Δx_minus = grid[nx] - grid[nx-1]
    D_minus = D
    D_center = D
    factor_right = (dt / dim) * 0.5 * (D_center + D_minus) / (Δx_minus)^2

    dl[nx-1] = factor_right
    d[nx] = -factor_right


    return Tridiagonal(dl, d, du)
end

A_1 = build_operator(gridx, dt, D)
A_2 = build_operator(gridy, dt, D)
A_3 = build_operator(gridz, dt, D)


U = [initial_condition_diffusion(x, y, z) for x in gridx, y in gridy, z in gridz] # defined in elixir_advection_diffusion.jl
z_slice = -0.4
idz = findall(z -> isapprox(z, z_slice; atol=1e-8), gridz)[1]
heatmap(gridx, gridy, U[:, :, idz]', title="Initial Condition at $(gridz[idz])", xlabel="x", ylabel="y",  clim=(1,1.3), size =(600, 500))

RHS = zero(U)
u_new_x = zero(U)
u_new_y = zero(U)
u_new_z = zero(U)
T = 1.0
n_steps = Int(T / dt)
for step in 1:n_steps
    @show step
    # X direction implicit, Y and Z direction explicit
    RHS .= U
    # Y explicit
    for i in 1:nx, k in 1:nz
        RHS[i, :, k] += A_2 * U[i, :, k]
    end
    # Z explicit
    for i in 1:nx, j in 1:ny
        RHS[i, j, :] += A_3 * U[i, j, :] 
    end
    # X implicit
    for j in 1:ny, k in 1:nz
        u_new_x[:, j, k] = (I - A_1) \ RHS[:, j, k]
    end


    # Y direction implicit, X and Z direction explicit
    RHS .= u_new_x
    # X explicit
    for j in 1:ny, k in 1:nz
        RHS[:, j, k] += A_1 * u_new_x[:, j, k]  
    end
    # Z explicit
    for i in 1:nx, j in 1:ny
        RHS[i, j, :] += A_3 * U[i, j, :]  
    end
    # Y implicit
    for i in 1:nx, k in 1:nz
        u_new_y[i, :, k] = (I - A_2) \ RHS[i, :, k]
    end

    # Z direction implicit, X and Y direction explicit
    RHS .= u_new_y
    # X explicit
    for j in 1:ny, k in 1:nz
        RHS[:, j, k] += A_1 * u_new_x[:, j, k]  
    end
    # Y explicit
    for i in 1:nx, k in 1:nz
        RHS[i, :, k] += A_2 * u_new_y[i, :, k]  
    end
    # Z implicit
    for i in 1:nx, j in 1:ny
        u_new_z[i, j, :] = (I - A_3) \ RHS[i, j, :]
    end
    U .= u_new_z
end



z_slice = -0.4
idz = findall(z -> isapprox(z, z_slice; atol=1e-8), gridz)[1]
heatmap(gridx, gridy, U[:, :, idz]', title="solution at T = $T and at z = $(gridz[idz])", xlabel="x", ylabel="y",  clim=(1,1.3), size =(600, 500))
