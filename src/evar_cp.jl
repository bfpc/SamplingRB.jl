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

import JuMP, Ipopt, ForwardDiff
using JuMP: Model, @variable, @constraint, @objective, @NLconstraint
using LinearAlgebra: norm

"""
Future cost function for Entropic Value-at-Risk representation
"""
function fcf_evar(x, losses; α=0.9)
    dim, nscen = size(losses)
    weights = x[1:dim]
    t = x[dim+1]
    # @assert t >= 0

    curmax = -Inf
    for i = 1:nscen
        li = @view losses[:,i]
        curmax = max(curmax, li'weights)
    end
    if isapprox(t,0) || t < 0
        return curmax
    end

    acc = 0.0
    for i = 1:nscen
        li = @view losses[:,i]
        acc += exp( (li'weights - curmax)/t )
    end
    return curmax + t*log(acc/nscen) - t*log(α)
end

"""
Build a cutting plane model for the Risk-Budgeting constraint,
given the list of budgets in B.

This model needs a compact feasible set;
since the weights are positive, we simply require an upper bound.
Most of the time, this (already large) default value should work.
"""
function cp_model_evar(B; ub_w=2000)
    dim = length(B)
    m = Model()

    # Weights and RB constraint
    @variable(m, 0 <= w[1:dim] <= ub_w)
    @NLconstraint(m, sum(bi * log(wi) for (bi,wi) in zip(B,w)) >= 0.0)

    # Entropic VaR auxiliary
    @variable(m, t >= 0)

    # Epigraph
    @variable(m, z)
    @objective(m, Min, z)

    m.ext[:cuts] = []
    return m
end

"""
    cutting_planes_evar(B::Vector{Float64}, α::Float64, losses::Array{Float64, 2};
                        tol::Float64=1e-6, maxiters::Int=1000, ub_w::Float64=2000., debug::Int=0, normalize::Bool=true)

Compute the investment weights on the assets in order to build an
Entropic V@R-alpha  risk budgeting portfolio using a cutting plane method.

The portfolio is such that each asset  j  contributs to the total risk with  B[j],
which is also called a  risk appetite.

The algorithm stops after reaching a provable optimality gap within
(either absolute or relative) tolerance  tol, or after  maxiters  iterations,
if it fails to converge.

In the beginning, the algorithm needs a bounding box for the weights.
Since the logarithm already implies w > 0, we use a sufficiently large bound
ub_w  for all weights.  If the algorithm converges to a solution that has any
weight that large, it should be checked for soundness.  The most common case
is when the problem is unbounded below, for example if  alpha  is not large enough.

debug >= 1  prints the iteration numbers, gap and change in the weights.
debug >= 2  also prints the current upper and lower bounds and trial points (weights, t)
debug >= 3  also prints the risk budgeting constraint and the first-stage status

Returns a triplet
(status, weights, t)
where the weights sum to one if normalize == true
"""
function cutting_planes_evar(B::Vector{Float64}, α::Float64, losses::Array{Float64, 2};
                             tol::Float64=1e-6, maxiters::Int=1000, ub_w::Float64=2000., debug::Int=0, normalize::Bool=true)
    dim = size(losses, 1)
    @assert dim == length(B)
    @assert 0 <= α <= 1

    # Second stage value function
    g = x -> fcf_evar(x, losses; α)

    # Build model
    m = cp_model_evar(B; ub_w)
    JuMP.set_optimizer(m, Ipopt.Optimizer)
    JuMP.set_optimizer_attribute(m, "print_level", 0)
    if tol < 1e-7
        if tol < 1e-14
            @warn "Using very small tolerances (desired = $(tol)) is extremely dangereous."
        end
        JuMP.set_optimizer_attribute(m, "tol", tol/10)
    end

    # Add a first cut (lower bound)
    w_start = ones(dim);
    t_start = 1;
    f, grad = add_cut!(m, w_start, t_start, g)

    w_prev = w_start
    wk     = zeros(dim)
    tk     = 0.0
    status = NotConverged
    for niter = 1:maxiters
        JuMP.optimize!(m)
        lb  = JuMP.objective_value(m)
        wk .= JuMP.value.(m[:w])
        tk  = JuMP.value(m[:t])
        st  = JuMP.primal_status(m)

        f, grad = add_cut!(m, wk, tk, g)
        gap = f - lb
        debug > 0 && println("Iteration: ", niter, " - gap: ", gap, " - moving: ", norm(wk .- w_prev))
        debug > 1 && println("       UB: ", f)
        debug > 1 && println("       LB: ", lb)
        debug > 1 && println("     Sols: ", wk, ", ", tk)
        debug > 2 && println("      RBC: ", sum(bi * log(wi) for (bi,wi) in zip(B,wk)))
        debug > 2 && println("   status: ", st)
        if niter > 10 && (gap < tol || gap < tol*abs(f))
            in_boundary = any(isapprox.(wk, ub_w, atol=tol, rtol=tol))
            status = in_boundary ? LikelyUnbounded : Converged
            break
        end
        w_prev .= wk
    end
    normalize && wk ./= sum(wk)
    return status, wk, tk
end

