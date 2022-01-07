import Pkg
Pkg.activate("../")

include("cvar_cp.jl")

import RData

# Parameters
dir = "/home/osboxes/Math/Risk_Parity/data/processed/gauss_dcc_gauss_garch/"
true_prices = RData.load(dir * "allPrices.RData")["allPrices"]

B = ones(11)
alpha = 0.90
outf = "ws-090.npz"
tol=1e-8
maxiters=1000

start_day = 1
end_day   = 12

ws = Vector{Float64}[]
hards = Int64[]

for d = start_day:end_day
    print(d, ", ")
    prices_sim   = RData.load(dir * "$(d)_prices_sim.RData")["prices_sim"][1,:,:]
    print(size(prices_sim))
    prices_today = true_prices[d,:]
    losses = prices_today .- prices_sim'
    status, w, t = cutting_planes(B, alpha, losses, tol=tol, maxiters=maxiters; debug=1)
    if status == 1
        print(" didn't reach tolerance $tol in $maxiters iterations")
        push!(hards, d)
    end
    push!(ws, w)
end

import NPZ
NPZ.npzwrite(outf, Dict("ws" => hcat(ws...)))

println("Scenarios with too large weights, maybe not optimal since other active constraints:")
for i in 1:length(ws)
    if any(ws[i] .> 1000)
        println(i, " ", ws[i])
    end
end
