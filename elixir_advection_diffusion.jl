using OrdinaryDiffEqTsit5
using Trixi
using Plots

###############################################################################
# semidiscretization of the linear advection-diffusion equation

advection_velocity = (0.0, 0.0, 0.0)
equations = LinearScalarAdvectionEquation3D(advection_velocity)
diffusivity() = 5.0e-2
equations_parabolic = LaplaceDiffusion3D(diffusivity(), equations)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = (-1.0, -1.0, -1.0) # minimum coordinates (min(x), min(y), min(z))
coordinates_max = (1.0, 1.0, 1.0) # maximum coordinates (max(x), max(y), max(z))

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                periodicity = false,
                n_cells_max = 30_000) # set maximum capacity of tree data structure

# Define initial condition
function initial_condition_diffusion(x, y, z)
    # Store translated coordinate for easy use of exact solution
    x_shift = [x, y, z] .- [0.2, 0.2, 0.2]
    nu = diffusivity()
    c = 1
    scalar = c + exp(-4 * sum(abs2, x_shift))
    return scalar
end

function initial_condition_diffusion(x, t,
                                                      equation::LinearScalarAdvectionEquation3D)
    scalar = initial_condition_diffusion(x...)
    return SVector(scalar)
end


initial_condition = initial_condition_diffusion

# define periodic boundary conditions everywhere
boundary_conditions = boundary_condition_periodic
boundary_conditions_parabolic = boundary_condition_periodic

boundary_conditions = BoundaryConditionDirichlet((x, t, equations_parabolic) -> SVector(1.0))
boundary_conditions_parabolic = BoundaryConditionNeumann((x, t, equations_parabolic) -> SVector(0.0))

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolicParabolic(mesh,
                                             (equations, equations_parabolic),
                                             initial_condition, solver;
                                            #  solver_parabolic = ViscousFormulationBassiRebay1(),
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_parabolic))

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.5
tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

# The AliveCallback prints short status information in regular intervals
alive_callback = AliveCallback(analysis_interval = analysis_interval)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback)

###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
alg = Tsit5()
time_int_tol = 1.0e-11
sol = solve(ode, alg; abstol = time_int_tol, reltol = time_int_tol,
            ode_default_options()..., callback = callbacks)
plot(sol, clim=(1,1.3), title = "Solution Trixi", size =(600, 500))


z_slice = -0.3
pd = PlotData2D(sol[end], semi, slice=:xy, point=(0.0, 0.0, z_slice))
plot(pd, clim=(1,1.3), title = "Solution Trixi slice xy at z=$z_slice", size =(600, 500))