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


coordinates_min = (-1.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max = (1.0, 1.0) # maximum coordinates (max(x), max(y))
dim = 2
gridx = range(coordinates_min[1], coordinates_max[1], length=401)
gridy = range(coordinates_min[2], coordinates_max[2], length=401)

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

nx = length(gridx)
ny = length(gridy)

dt = 0.1


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
# Modify for von Neumann BC

A_1 = build_operator(gridx, dt, D)
A_2 = build_operator(gridy, dt, D)



U = [initial_condition_diffusion(x, y) for x in gridx, y in gridy] # defined in elixir_advection_diffusion.jl
heatmap(gridx, gridy, U', title="Initial Condition", xlabel="x", ylabel="y", size =(600, 500))

RHS = zero(U)
u_new = zero(U)
T = 4.0
n_steps = Int(T / dt)
for _ in 1:n_steps
    # X direction implicit, Y direction explicit
    # For each row i, apply (I + Δt A_2) to u_old[i,:]
    for i in 1:nx
        RHS[i, :] = (I + A_2) * U[i, :]  # B1 is ny×ny, u_old[i,:] is length ny
    end

    for j in 1:ny
        u_new[:, j] = (I - A_1) \ RHS[:, j]  # M1 is nx×nx, RHS[:,j] is length nx
    end


    # Y direction implicit, X direction explicit
    for j in 1:ny
        RHS[:, j] = (I + A_1) * u_new[:, j]  # B2 is nx×nx, u_new[:,j] is length nx
    end

    for i in 1:nx
        U[i, :] = (I - A_2) \ RHS[i, :]  # M2 is ny×ny, RHS[:,i] is length ny
    end

end

heatmap(gridx, gridy, U', title="Solution at final time", xlabel=L"x", ylabel=L"y", clim=(1, 1.3), size =(600, 500), left_margin = 0Plots.mm, right_margin = 0Plots.mm, top_margin = 0Plots.mm, bottom_margin = 0Plots.mm)