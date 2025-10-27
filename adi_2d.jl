# using Pkg
# Pkg.activate(@__DIR__)
# Pkg.instantiate()

# Pkg.add("LinearAlgebra")
# Pkg.add("KernelAbstractions")
# Pkg.add("Trixi")
# Pkg.add("OrdinaryDiffEqTsit5")
# Pkg.add("Plots")
Pkg.add("SparseArrays")

using LinearAlgebra
using KernelAbstractions
using Trixi
using OrdinaryDiffEqTsit5
using Plots
using SparseArrays


coordinates_min = (-1.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max = (1.0, 1.0) # maximum coordinates (max(x), max(y))

gridx = range(coordinates_min[1], coordinates_max[1], length=101)
gridy = range(coordinates_min[2], coordinates_max[2], length=101)

dx = step(gridx)
dy = step(gridy)

nx = length(gridx)
ny = length(gridy)

dt = 0.5
diffusivity() = 5.0e-2

αx = diffusivity() * dt / dx^2
αy = diffusivity() * dt / dy^2

# Interior stencil
dl_x = fill(αx, nx-1)
d_x  = fill(-2αx, nx)
du_x = fill(αx, nx-1)

# Modify for von Neumann BC
d_x[1] = -αx    # left boundary
d_x[end] = -αx  # right boundary
A_1 = Tridiagonal(dl_x, d_x, du_x)

dl_y = fill(αy, ny-1)
d_y  = fill(-2αy, ny)
du_y = fill(αy, ny-1)

# Modify for von Neumann BC
d_y[1] = -αy    # left boundary
d_y[end] =  -αy  # right boundary
A_2 = Tridiagonal(dl_y, d_y, du_y)




U = [initial_condition_diffusion(x, y) for x in gridx, y in gridy] # defined in elixir_advection_diffusion.jl
heatmap(gridx, gridy, U', title="Initial Condition", xlabel="x", ylabel="y", colorbar_title="u", aspect_ratio=1)

RHS = zero(U)
u_new = zero(U)
T = 2.0
n_steps = Int(T / 2dt)
for _ in 1:n_steps
# X direction implicit, Y direction explicit
# For each row i, apply (I + Δt A_2) to u_old[i,:]
for i in 1:nx
    RHS[i,:] = (I + A_2) * U[i,:]  # B1 is ny×ny, u_old[i,:] is length ny
end

for j in 1:ny
    u_new[:,j] = (I - A_1) \ RHS[:,j]  # M1 is nx×nx, RHS[:,j] is length nx
end


# Y direction implicit, X direction explicit
for j in 1:ny
    RHS[:,j] = (I + A_1) * u_new[:,j]  # B2 is nx×nx, u_new[:,j] is length nx
end

for i in 1:nx
    U[i, :] = (I - A_2) \ RHS[i, :]  # M2 is ny×ny, RHS[:,i] is length ny
end

end

heatmap(gridx, gridy, U', title="Solution at final time", xlabel="x", ylabel="y", colorbar_title="u", aspect_ratio=1, clim=(1,1.3))