# Copyright (C) 2021 - 2022 Bernardo Freitas Paulo da Costa
#
# This file is part of RiskBudgeting.jl.
#
# RiskBudgeting.jl is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# RiskBudgeting.jl is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# RiskBudgeting.jl. If not, see <https://www.gnu.org/licenses/>.

"""
# RiskBudgeting

RiskBudgeting is a package designed to calculate long-only weights for
coherent Risk Parity and Risk Budgeting portfolios,
given arbitrary simulations of relative losses of each asset.

It provides a general cutting plane algorithm for a coherent risk measures.
Special risk measures, such as CVaR and distortion risk measures are
already provided for ease of use.

In the special case of CVaR, it implements a cutting plane algorithm
with dedicated initialization for numerical stability and performance,
allowing for several thousand simulations.
It also implements two stochastic gradient algorithms,
taking samples from a user-defined function that allows for
arbitrary distributions.
One is based on the Lagrangian reformulation of the problem,
while the other is a projected version into the feasible domain.
"""

module RiskBudgeting

@enum OptResult Converged NotConverged LikelyUnbounded

# Conditional Value-at-Risk, aka Expected Shortfall and Average VaR
include("cvar_cp.jl") # Cutting Plane algorithm, and related utilities
include("cvar_sgd.jl") # Stochastic gradient algorithm

include("cp.jl") # Generic cutting plane algorithm
include("risk_measures.jl") # Risk Measures for the generic algorithm

"""
    cvar_rbp(B::Vector{Float64}, alpha::Float64, rel_losses::Array{Float64,2}; tol::Float64=1e-6, maxiters::Int=1000)

Compute the investment weights on the assets in order to build a
CV@R-`alpha` risk budgeting portfolio given risk appetites in `B`
and a matrix of relative losses in `rel_losses`.
This matrix has size  (|B|, nsamples), so that each column corresponds
to a sample of relative losses, and each row to a given asset.

The algorithm stops after reaching a provable optimality gap within
(both absolute and relative) tolerance `tol`, or after `maxiters` iterations,
if it fails to converge.

Returns (failed, w) :: Bool, Vector{Float64}

For more details on the algorithm, see `cutting_planes`.
"""
function cvar_rbp(B::Vector{Float64}, alpha::Float64, rel_losses::Array{Float64,2}; tol::Float64=1e-6, maxiters::Int=1000)
  f, w, t = cutting_planes(B, alpha, rel_losses; tol=tol, maxiters=maxiters, debug=0)
  return f == NotConverged, w
end


"""
    cvar_rbp(B::Vector{Float64}, α::Float64, loss_sampler::Function; maxiters::Int=10000)

Compute the investment weights on the assets in order to build a
CV@R-`alpha` risk budgeting portfolio given risk appetites in `B`
and a function `loss_sampler()` that returns a sample vector
of relative losses (from their joint distribution).

The algorithm stops after `maxiters` iterations, that is, after sampling
`maxiters` vectors of relative losses.

Returns w :: Vector{Float64}

For more details on the algorithm, see `sgd_Lagrangian`.
"""
function cvar_rbp(B::Vector{Float64}, α::Float64, loss_sampler::Function; maxiters::Int=10000)
  v, t = sgd_Lagrangian(B, α, loss_sampler; maxiters=maxiters, ϵ=1e-12, debug=0)
  # TODO: convergence test for tolerance

  return v./sum(v)
end

export cvar_rbp

export CVaR, Entropic, WorstCase, Distortion

end
