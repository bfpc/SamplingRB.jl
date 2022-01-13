# Copyright (C) 2021 - 2022 Bernardo Freitas Paulo da Costa
#
# This file is part of CVaRRiskParity.jl.
#
# CVaRRiskParity.jl is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# CVaRRiskParity.jl is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# CVaRRiskParity.jl. If not, see <https://www.gnu.org/licenses/>.

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
    cvar_rbp(B::Vector{Float64}, alpha::Float64, rel_losses::Array{Float64,2}; tol::Float64=1e-6, maxiters::Int=1000)

Compute the investment weights on the assets in order to build a
CV@R-`alpha` risk budgeting portfolio given risk appetites in `B`
and a matrix of relative losses in `rel_losses`.

The algorithm stops after reaching a provable optimality gap within
(both absolute and relative) tolerance `tol`, or after `maxiters` iterations,
if it fails to converge.

Returns (failed, w) :: Bool, Vector{Float64}

For more details on the algorithm, see `cutting_planes`.
"""
function cvar_rbp(B::Vector{Float64}, alpha::Float64, rel_losses::Array{Float64,2}; tol::Float64=1e-6, maxiters::Int=1000)
  f, w, t = cutting_planes(B, alpha, rel_losses; tol=tol, maxiters=maxiters, debug=0)
  return f == 1, w
end

export cvar_rbp

end
