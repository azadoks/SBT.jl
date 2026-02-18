using Test
using PseudoPotentialData
using PseudoPotentialIO
using Polynomials
using SpecialFunctions

using SBT

@testset "SBT.jl" begin
    @testset "β³/2 exp(-βr)" begin
        N = 256
        rmin = 1.0 / 1024 / 32
        rmax = 20.0
        r = collect(logrange(rmin, rmax, N))
        k = sbtfreq(r)
        β = 2.0
        f_true = @. β^3 / 2 * exp(-β * r)
        g_true = @. sqrt(2/π) * β^4 / (k^2 + β^2)^2
        g_sbt, _ = SBT.sbt(0, f_true, r; normalize=true)
        @test all(isapprox.(g_sbt, g_true, atol=1e-10))
        # f_sbt = SBT.sbt(0, g_sbt, r, direction=:inverse)
        # @test all(isapprox.(f_sbt, f_true, atol=1e-10))
    end
    # include("hgh.jl")
end
