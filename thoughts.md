For step "RHS[i, :, k] = (I + A_2) * U[i, :, k]"
either
- build (I+A_2) for every x and z and then do matmul
- write @kernel which does (I+A_2)*vector for all x,y,z
 (takes U[:,:,:] and 'returns'/inplace RHS[:,:,:])
 



For step "U[:, j, k] = (I - A_1) \ RHS[:, j, k]"
either:
- build (I+A_1) for every y and z and and use (most likley already highly optimized version) of Thomas algorithm / solver for Tridiagonal Matrix.
- write my own version of Thomas algorithm in KA.jl so i can solve all the ny*nz = 50 * 2000 = 100000 solves in parallel (with varying A_1 for every version)




Also can it make sense to combine the two steps, so have one kernel which just does both at the same time?

For step "U[:, j, k] = (I - A_1) \ RHS[:, j, k]"
- write my own version of Thomas algorithm in KA.jl so i can solve all the ny*nz = 50 * 2000 = 100000 solves in parallel (with varying A_1 for every version)

