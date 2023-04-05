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

using Test
using Random: MersenneTwister
using SamplingRB
using SamplingRB: risk, value_function

function simpletest()
  rng = MersenneTwister(1)

  # Parameters
  d    = 3  # dimension
  nsim = 10 # Nb of simulations

  B = ones(d)
  alpha = 0.90
  relative_losses = randn(rng, d, nsim)

  status, w = cvar_rbp(B, alpha, relative_losses)
  @test status == false
  @test w ≈ [0.2280, 0.2706, 0.5014] atol = 1e-4
end

function sgd_small()
  rng = MersenneTwister(1)

  # Parameters
  d    = 3  # dimension
  nsim = 10 # Nb of simulations

  B = ones(d)
  alpha = 0.90

  samples = randn(rng, d, nsim)
  function sampler()
    idx = rand(rng, 1:nsim)
    return samples[:,idx]
  end

  w, t = SamplingRB.projected_sgd(B, alpha, sampler)
  @test w/sum(w) ≈ [0.23091991, 0.27860780, 0.49047228] atol = 1e-4
end

function cvar_tests()
  v = collect(1:10)
  @test risk(CVaR(0.9), v) == 10
  @test risk(CVaR(0.75), v) == 9.2
  @test risk(CVaR(0.), v) == sum(v)/length(v)

  w = [4, 7, 10, 9, 5, 3, 6, 2, 8, 1]
  @test risk(CVaR(0.9), w) == 10
  @test risk(CVaR(0.75), w) == 9.2
  @test risk(CVaR(0.), w) == sum(w)/length(w)
end

function distortion_tests()
  g10(x) = min(10x, 1)
  g6(x)  = min( 6x, 1)

  v = collect(1:10)
  @test risk(Distortion(g10), v) == 10
  @test risk(Distortion(g6), v) == 9.6

  w = [4, 7, 10, 9, 5, 3, 6, 2, 8, 1]
  @test risk(Distortion(g10), w) == 10
  @test risk(Distortion(g6), w) == 9.6
end

function entropic_tests()
  v = collect(1:10)
  @test risk(Entropic(1.), v) ≈ 8.156044651432666
  @test risk(Entropic(2.), v) ≈ 8.92141418140683

  w = [4, 7, 10, 9, 5, 3, 6, 2, 8, 1]
  @test risk(Entropic(1.), w) ≈ 8.156044651432666
  @test risk(Entropic(2.), w) ≈ 8.92141418140683
end

# Compatibility between `risk` and `value_function` when both are defined
function compatibility_risk_vf()
  rng = MersenneTwister(2)

  # Parameters
  d    = 3  # dimension
  nsim = 10 # Nb of simulations

  weights = rand(rng, d)
  losses  = rand(rng, d, nsim)
  for ρ in [Entropic(1.0), Entropic(2.3), WorstCase()]
    @test risk(ρ, losses' * weights) == value_function(ρ, weights, losses)
  end
end

@testset "SamplingRB.jl" begin
  @testset "Simple" begin
    simpletest()
  end
  @testset "SGD" begin
    sgd_small()
  end
  @testset "Risk Measures" begin
    @testset "CV@R" begin cvar_tests() end
    @testset "Distortion" begin distortion_tests() end
    @testset "Entropic" begin entropic_tests() end
    @testset "ValueFunctions" begin compatibility_risk_vf() end
  end
end

