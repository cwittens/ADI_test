using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

Pkg.add("LinearAlgebra")
Pkg.add("KernelAbstractions")