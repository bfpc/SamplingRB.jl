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

import JuMP, Ipopt, ForwardDiff
using JuMP: Model, @variable, @constraint, @objective, @NLconstraint
using LinearAlgebra: norm

function oracle(v::Vector{Float64}, t::Float64, ξ::Vector{Float64}, α::Float64)
    return t + max(ξ'v - t,0)/(1-α)
end

function subgrad_oracle(v::Vector{Float64}, t::Float64, ξ::Vector{Float64}, α::Float64)
    if ξ'v >= t
        return ξ/(1-α), -α/(1-α)
    else
        return zero(v), 1
    end
end

function projection_model(B)
    dim = length(B)
    m = Model()

    # Weights and RB constraint
    @variable(m, 0 <= v[1:dim])
    @NLconstraint(m, sum(B[i] * log(v[i]) for i=1:dim) >= 0.0)

    # Initial point
    @variable(m, v0[1:dim])
    @objective(m, Min, sum( (v[i] - v0[i])^2 for i=1:dim))

    return m
end

"""
    projected_sgd(B::Vector{Float64}, α::Float64, loss_sampler::Function;
                  maxiters::Int=100000, debug::Int=0)

Compute the investment exposure on the assets in order to build a
CV@R-α  risk budgeting portfolio using a projected stochastic gradient
descent method.

This is made possible by the introduction of an auxiliary variable  t
which replaces the CVaR by an expectation, for which stochastic gradients
are easy to calculate.

The algorithm stops after maxiter iterations.

debug >= 1  prints the iteration numbers, exposures and current estimate of VaR.

Returns a pair (v, V@R-α)
"""
function projected_sgd(B::Vector{Float64}, α::Float64, loss_sampler::Function;
                       maxiters::Int=10000, debug::Int=0)
    @assert 0 <= α <= 1
    dim = length(B)

    # Build model
    m = projection_model(B)
    JuMP.set_optimizer(m, Ipopt.Optimizer)
    JuMP.set_optimizer_attribute(m, "print_level", 0)

    # Starting point
    v = ones(dim);
    t = 0.;

    for niter = 1:maxiters
        dv, dt = subgrad_oracle(v, t, loss_sampler(), α)
        v .-= dv/niter
        t  -= dt/niter
        # Only project if out of domain
        if any(v .<= 0) || sum(B[i] * log(v[i]) for i=1:dim) < 0
            JuMP.fix.(m[:v0], v)
            JuMP.optimize!(m)
            v  .= JuMP.value.(m[:v])
        end
        debug > 0 && println("Iteration: ", niter, " v: ", v, "; t: ", t)
    end
    return v, t
end

