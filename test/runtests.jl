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

using Test
using Random: MersenneTwister
using CVaRRiskParity

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
  @test w ≈ [0.7264, 0.8620, 1.5970] atol = 1e-4
end

@testset "CVaRRiskParity.jl" begin
  @testset "Simple" begin
    simpletest()
  end
end

