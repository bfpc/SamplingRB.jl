# Copyright (C) 2021 - 2022 Bernardo Freitas Paulo da Costa
#
# This file is part of SamplingRB.jl.
#
# SamplingRB.jl is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# SamplingRB.jl is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# SamplingRB.jl. If not, see <https://www.gnu.org/licenses/>.

import JuMP, Ipopt, ForwardDiff
using JuMP: Model, @variable, @constraint, @objective, @NLconstraint
using LinearAlgebra: norm

function cp_model(B; ub_w=2000)
    dim = length(B)
    m = Model()

    # Weights and RB constraint
    @variable(m, 0 <= w[1:dim] <= ub_w)
    @NLconstraint(m, sum(bi * log(wi) for (bi,wi) in zip(B,w)) >= 0.0)

    # Epigraph
    @variable(m, z)
    @objective(m, Min, z)

    m.ext[:cuts] = []
    return m
end

function add_cut!(m, wk, fcf)
    n    = length(wk)
    f    = fcf(wk)
    grad = ForwardDiff.gradient(fcf, wk)
    c = @constraint(m, m[:z] >= f + sum(grad[i] * (m[:w][i] - wk[i]) for i = 1:n))
    push!(m.ext[:cuts], c)
    return f, grad
end


"""
    cutting_planes(B::Vector{Float64}, rel_losses::Array{Float64,2},
                   risk_measure::AbstractRiskMeasure;
                   tol::Float64=1e-6, maxiters::Int=1000, ub_w::Float64=2000., debug::Int=0)

Compute the investment weights on the assets in order to build a
risk budgeting portfolio using a cutting plane method.
The portfolio is such that each asset  j  contributs to the total risk with  B[j],
which is also called a  risk appetite.

If the evaluation of the Risk Measure is efficient,
this scales to several thousand realizations of the simulations of relative losses.

The algorithm stops after reaching a provable optimality gap within
(either absolute or relative) tolerance  tol, or after  maxiters  iterations,
if it fails to converge.

In the beginning, the algorithm needs a bounding box for the weights.
Since the logarithm already implies w > 0, we use a sufficiently large bound
ub_w  for all weights.  If the algorithm converges to a solution that has any
weight that large, it should be checked for soundness.  The most common case
is when the problem is unbounded below,
for example if the risk measure is too conservative.

debug >= 1  prints the iteration numbers, gap and change in the weights.
debug >= 2  also prints the current upper and lower bounds and trial weights.
debug >= 3  also prints the risk budgeting constraint and the first-stage status

Returns a tuple
(result, weights)
"""
function cutting_planes(B::Vector{Float64}, risk_measure::AbstractRiskMeasure, rel_losses::Array{Float64,2};
                        tol::Float64=1e-6, maxiters::Int=1000, ub_w::Float64=2000., debug::Int=0)
    dim = size(rel_losses,1)
    @assert dim == length(B)

    # Second stage value function
    g = w -> value_function(risk_measure, w, rel_losses)

    # Build model
    m = cp_model(B; ub_w)
    JuMP.set_optimizer(m, Ipopt.Optimizer)
    JuMP.set_optimizer_attribute(m, "print_level", 0)
    if tol < 1e-7
        if tol < 1e-14
            @warn "Using very small tolerances (desired = $(tol)) is extremely dangereous."
        end
        JuMP.set_optimizer_attribute(m, "tol", tol/10)
    end

    # Add a first cut (lower bound) to avoid degenerate solutions
    w_start = ones(dim);
    f, grad = add_cut!(m, w_start, g)

    w_prev = w_start
    wk     = zeros(dim)
    status = NotConverged
    for niter = 1:maxiters
        JuMP.optimize!(m)
        lb = JuMP.objective_value(m)
        wk .= JuMP.value.(m[:w])
        st  = JuMP.primal_status(m)

        f, grad = add_cut!(m, wk, g)
        gap = f - lb
        debug > 0 && println("Iteration: ", niter, " - gap: ", gap, " - moving: ", norm(wk .- w_prev))
        debug > 1 && println("       UB: ", f)
        debug > 1 && println("       LB: ", lb)
        debug > 1 && println("  Weights: ", wk)
        debug > 2 && println("      RBC: ", sum(bi * log(wi) for (bi,wi) in zip(B,wk)))
        debug > 2 && println("   status: ", st)
        if niter > 10 && (gap < tol || gap < tol*abs(f))
            in_boundary = any(isapprox.(wk, ub_w, atol=tol, rtol=tol))
            status = in_boundary ? LikelyUnbounded : Converged
            break
        end
        w_prev .= wk
    end
    wk ./= sum(wk)
    return status, wk
end
