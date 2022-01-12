import Pkg
Pkg.activate(".")

import NPZ
using CVaRRiskParity


# Parameters
dir  = "data"
pricesf = joinpath(dir, "allPrices.npz")
simulsf = d -> joinpath(dir, "$(d)_prices_sim.npz")
outf = joinpath("results", "095", "weights.npz")

B = ones(11)
alpha = 0.95

days = [1, 5, 485]

# Script itself
true_prices = NPZ.npzread(pricesf)["allPrices"]

ws = Vector{Float64}[]
hards = Int64[]

for d in days
    print(d, ", ")
    prices_sim   = NPZ.npzread(simulsf(d))["prices_sim"]
    prices_today = true_prices[d,:]
    relative_losses = 1 .- prices_sim' ./ prices_today
    failed, w = cvar_rbp(B, alpha, relative_losses)
    if failed
        println(" didn't reach tolerance $tol in $maxiters iterations")
        push!(hards, d)
    end
    push!(ws, w)
end
println("done!")
println()

println("Scenarios with too large weights:")
println("  (Likely the corresponding risk parity problem is unbounded below)")
for i in 1:length(ws)
    if any(ws[i] .> 1000)
        println("day ", days[i], " ", ws[i])
    end
end

# Saving weights for later use
mkpath(dirname(outf))
NPZ.npzwrite(outf, Dict("ws" => hcat(ws...)))
