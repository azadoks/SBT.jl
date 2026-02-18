using Test
using PseudoPotentialData
using PseudoPotentialIO
using Polynomials
using SpecialFunctions

using SBT

@testset "SBT.jl" begin
    @testset "sbtfreq" begin
        for n in [5, 7, 9, 10, 100, 101, 102, 1000, 1013, 78934]
            k = sbtfreq(logrange(1e-5, 20, n))
            plan = SBT.SBTPlan{Float64}(logrange(1e-5, 20, n), 0, 500.0)
            @test all(k .== plan.k)
        end
    end

    @testset "β³/2 exp(-βr)" begin
        N = 256
        rmin = 1.0 / 1024 / 32
        rmax = 20.0
        r = collect(logrange(rmin, rmax, N))
        k = sbtfreq(r)
        β = 2.0
        f_true = @. β^3 / 2 * exp(-β * r)
        g_true = @. sqrt(2/π) * β^4 / (k^2 + β^2)^2
        g_sbt, _ = SBT.sbt(0, f_true, r; normalize=true, direction=:forward)
        @test all(isapprox.(g_sbt, g_true, atol=1e-10))
        f_sbt, _ = SBT.sbt(0, g_sbt, r, normalize=true, direction=:inverse)
        @test all(isapprox.(f_sbt, f_true, atol=1e-1))  # TODO: this is pretty bad
    end
    include("hgh.jl")
end
