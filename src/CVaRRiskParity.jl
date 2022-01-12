"""
# CVaRRiskParity

CVaRRiskParity is a package designed to calculate long-only
CV@R Risk Parity and Risk Budgeting portfolios,
given arbitrary simulations of relative losses of each asset.

Currently, it implements a cutting plane algorithm
with dedicated initialization for numerical stability and performance,
allowing for several thousand simulations.
"""

module CVaRRiskParity

include("cvar_cp.jl") # Cutting Plane algorithm, and related utilities

"""
    cvar_rbp(B::Vector{Float64}, alpha::Float64, rel_losses::Array{Float64,2}; tol::Float64=1e-6, maxiters::Int64=1000)

Compute the investment weights on the assets in order to build a
CV@R-`alpha` risk budgeting portfolio given risk appetites in `B`
and a matrix of relative losses in `rel_losses`.

The algorithm stops after reaching a provable optimality gap within
(both absolute and relative) tolerance `tol`, or after `maxiters` iterations,
if it fails to converge.

Returns (failed, w) :: Bool, Vector{Float64}

For more details on the algorithm, see `cutting_planes`.
"""
function cvar_rbp(B::Vector{Float64}, alpha::Float64, rel_losses::Array{Float64,2}; tol::Float64=1e-6, maxiters::Int64=1000)
  f, w, t = cutting_planes(B, alpha, rel_losses; tol=tol, maxiters=maxiters, debug=0)
  return f, w
end

export cvar_rbp

end
