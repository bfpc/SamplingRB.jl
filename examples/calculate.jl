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

import Pkg
Pkg.activate(".")

import NPZ
using SamplingRB


# Parameters
dir  = "data"
pricesf = joinpath(dir, "allPrices.npz")
simulsf = d -> joinpath(dir, "$(d)_prices_sim.npz")
outf = joinpath("results", "095", "weights.npz")

B = ones(11)
alpha = 0.95
ρ = CVaR(alpha)
# γ = 2.
# ρ = Entropic(γ)
# ρ = WorstCase()

days = [1, 5, 485]

# Script itself
true_prices = NPZ.npzread(pricesf)["allPrices"]

ws = Vector{Float64}[]
hards = Int64[]
unbounded = Int64[]

for d in days
    print(d, ", ")
    prices_sim   = NPZ.npzread(simulsf(d))["prices_sim"]
    prices_today = true_prices[d,:]
    relative_losses = 1 .- prices_sim' ./ prices_today
    status, w = SamplingRB.cutting_planes(B, relative_losses, ρ)
    if status == SamplingRB.NotConverged
        println(" didn't reach tolerance $tol in $maxiters iterations")
        push!(hards, d)
    end
    if status == SamplingRB.LikelyUnbounded
        push!(unbounded, d)
    end
    push!(ws, w)
end
println("done!")
println()

if unbounded != []
    println("Scenarios with too large weights:")
    println("  (Likely the corresponding risk parity problem is unbounded below)")
    for i in unbounded
        println("day ", days[i], " ", ws[i])
    end
end

if hards != []
    println("Scenarios for which cutting planes did not converge:")
    for i in hards
        println("day ", days[i], " ", ws[i])
    end
end

# Saving weights for later use
mkpath(dirname(outf))
NPZ.npzwrite(outf, Dict("ws" => hcat(ws...)))
