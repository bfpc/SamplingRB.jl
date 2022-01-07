import Pkg
Pkg.activate(".")

import RData
import NPZ
using Printf: @sprintf
using CvarRiskParity


# Parameters
dir  = joinpath("..", "data", "processed")
pricesf = joinpath(dir, "allPrices.RData")
simulsf = d -> joinpath(dir, "$(d)_prices_sim.RData")
outf = joinpath("results", "095", "weights.npz")

B = ones(11)
alpha = 0.95

days = [1, 5, 485]

# Script itself
true_prices = RData.load(pricesf)["allPrices"]

ws = Vector{Float64}[]
hards = Int64[]

for d in days
    print(d, ", ")
    prices_sim   = RData.load(simulsf(d))["prices_sim"][1,1:50,:]
    prices_today = true_prices[d,:]
    relative_losses = 1 .- prices_sim' ./ prices_today
    status, w = cvar_rbp(B, alpha, relative_losses)
    if status == 1
        println(" didn't reach tolerance $tol in $maxiters iterations")
        push!(hards, d)
    end
    push!(ws, w)
end

println("Scenarios with too large weights, maybe not optimal since other active constraints:")
for i in 1:length(ws)
    if any(ws[i] .> 1000)
        println("day ", days[i], " ", ws[i])
    end
end

mkpath(dirname(outf))
NPZ.npzwrite(outf, Dict("ws" => hcat(ws...)))
