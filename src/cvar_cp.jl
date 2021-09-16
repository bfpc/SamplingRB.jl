import JuMP, Ipopt, ForwardDiff
using JuMP: Model, @variable, @constraint, @objective, @NLconstraint

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

function cp_model(B)
    dim = length(B)
    m = Model()

    # Weights and RB constraint
    @variable(m, 0 <= w[1:dim] <= 2000)
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

function normdiff(u,v)
    nd = 0.0
    for (ui,vi) in zip(u,v)
      nd += (ui-vi)^2
    end
    return nd
end

function cutting_planes(B, alpha, losses; tol=1e-6, debug=0, maxiters=1000, m=nothing)
    dim = size(losses,1)
    @assert dim == length(B)

    # Second stage value function
    g = x -> fcf(x,losses, alpha=alpha)

    # Build model if needed
    if m == nothing
        m = cp_model(B)
        JuMP.set_optimizer(m, Ipopt.Optimizer)
        JuMP.set_optimizer_attribute(m, "print_level", 0)
    end

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
        wk .= JuMP.value.(m[:w])
        tk  = JuMP.value(m[:t])
        f, grad = add_cut!(m, wk, tk, g)
        gap = f - JuMP.objective_value(m)
        debug > 0 && println("Iteration: ", niter, " - gap: ", gap, " - moving: ", normdiff(wk, w_prev))
        debug > 1 && println("       UB: ", f)
        debug > 1 && println("     Sols: ", wk, " - ", tk)
        if niter > 10 && gap < tol
            return 0, wk, tk
        end
        niter += 1
        w_prev .= wk
    end
    return 1, wk, tk
end

