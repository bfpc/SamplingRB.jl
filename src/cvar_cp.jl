import JuMP, Ipopt, ForwardDiff
using JuMP: Model, @variable, @constraint, @objective, @NLconstraint
using LinearAlgebra: norm

function cvar_ru(w,t,loss,alpha)
    return max(loss'w - t,0)/(1-alpha)
    # Not adding "t" for each scenario is numerically better for SAA: add only once, not averaged
end

function fcf(x, losses; alpha=0.85)
    w = x[1:end-1]
    t = x[end]
    acc = 0.0
    n = size(losses, 2)
    for i = 1:n
        li = @view losses[:,i]
        acc += cvar_ru(w, t, li, alpha)
    end
    return t + acc/n
end

function cp_model(B; ub_w=2000)
    dim = length(B)
    m = Model()

    # Weights and RB constraint
    @variable(m, 0 <= w[1:dim] <= ub_w)
    @NLconstraint(m, sum(bi * log(wi) for (bi,wi) in zip(B,w)) >= 0.0)

    # CV@R auxiliary
    @variable(m, t)

    # Epigraph
    @variable(m, z)
    @constraint(m, z >= t)
    @objective(m, Min, z)

    m.ext[:cuts] = []
    return m
end

function add_cut!(m, wk, tk, fcf)
    x    = [wk; tk]
    n    = length(wk)
    f    = fcf(x)
    grad = ForwardDiff.gradient(fcf, x)
    c = @constraint(m, m[:z] >= f + sum(grad[i] * (m[:w][i] - wk[i]) for i = 1:n) + grad[end]*(m[:t] - tk))
    push!(m.ext[:cuts], c)
    return f, grad
end

"""
    cutting_planes(B::Vector{Float64}, alpha::Float64, rel_losses::Array{Float64,2};
                   tol::Float64=1e-6, maxiters::Int=1000, ub_w::Float64=2000., debug::Int=0)

Compute the investment weights on the assets in order to build a
CV@R-alpha  risk budgeting portfolio using a cutting plane method.

This leads to a decomposition of the simulations of relative losses in  rel_losses
in smaller subproblems, and allow us to scale for several thousand realizations.
The portfolio is such that each asset  j  contributs to the total risk with  B[j],
which is also called a  risk appetite.

The algorithm stops after reaching a provable optimality gap within
(both absolute and relative) tolerance  tol, or after  maxiters  iterations,
if it fails to converge.

In the beginning, the algorithm needs a bounding box for the weights.
Since the logarithm already implies w > 0, we use a sufficiently large bound
ub_w  for all weights.  If the algorithm converges to a solution that has any
weight that large, it should be checked for soundness.  The most common case
is when the problem is unbounded below, for example if  alpha  is not large enough.

debug >= 1  prints the iteration numbers, gap and change in the weights.
debug >= 2  also prints the current upper bound and trial points (weights, V@R)

Returns a triplet
(failed, weights, V@R-alpha)
"""
function cutting_planes(B::Vector{Float64}, alpha::Float64, rel_losses::Array{Float64,2};
                        tol::Float64=1e-6, maxiters::Int=1000, ub_w::Float64=2000., debug::Int=0)
    dim = size(rel_losses,1)
    @assert dim == length(B)
    @assert 0 <= alpha <= 1

    # Second stage value function
    g = x -> fcf(x,rel_losses, alpha=alpha)

    # Build model
    m = cp_model(B, ub_w=ub_w)
    JuMP.set_optimizer(m, Ipopt.Optimizer)
    JuMP.set_optimizer_attribute(m, "print_level", 0)

    # Add a first cut (lower bound) and an upper bound on t, to avoid degenerate solutions
    w_start = ones(dim);
    t_start = 0;
    f, grad = add_cut!(m, w_start, t_start, g)
    @constraint(m, m[:t] <= f)

    w_prev = w_start
    wk     = zeros(dim)
    tk     = 0.0
    for niter = 1:maxiters
        JuMP.optimize!(m)
        lb = JuMP.objective_value(m)
        wk .= JuMP.value.(m[:w])
        tk  = JuMP.value(m[:t])
        f, grad = add_cut!(m, wk, tk, g)
        gap = f - lb
        debug > 0 && println("Iteration: ", niter, " - gap: ", gap, " - moving: ", norm(wk .- w_prev))
        debug > 1 && println("       UB: ", f)
        debug > 1 && println("     Sols: ", wk, " - ", tk)
        if niter > 10 && (gap < tol || gap < tol*abs(f))
            return 0, wk, tk
        end
        niter += 1
        w_prev .= wk
    end
    return 1, wk, tk
end

