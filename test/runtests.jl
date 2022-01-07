using CvarRiskParity
using Test
import Random: seed!

function simpletest()
  seed!(1)

  # Parameters
  d    = 3  # dimension
  nsim = 10 # Nb of simulations

  B = ones(d)
  alpha = 0.90
  relative_losses = randn(d, nsim)

  status, w = cvar_rbp(B, alpha, relative_losses)
  @test status == 0
  @test w ≈ [0.7264, 0.8620, 1.5970] atol = 1e-4
end

@testset "CvarRiskParity.jl" begin
  @testset "Simple" begin
    simpletest()
  end
end

