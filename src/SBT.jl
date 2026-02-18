module SBT

using Bessels: sphericalbesselj
using SpecialFunctions: gamma
using FFTW

export SBTPlan
export sbt
export sbtfreq

@inbounds function construct_Mℓ_t(ℓmax, nr, Δt, κmin, ρmin, Δρ)
    t = collect(0:(nr - 1)) * Δt
    Mℓ_t = zeros(Complex{Float64}, nr, ℓmax + 1)

    # ℓ=0
    # Eq.(19)
    Φ₁ = imag.(log.(gamma.(1 / 2 .- im * t)))
    Φ₂ = -atan.(tanh.(π * t / 2))
    Mℓ_t[:, 1] .= sqrt(π / 2) * exp.(im * (Φ₁ + Φ₂)) / nr
    Mℓ_t[1, 1] /= 2
    Mℓ_t[:, 1] .*= exp.(im * t * (κmin + ρmin))

    # ℓ=1
    # Eq.(16)
    if ℓmax >= 1
        ϕ = atan.(tanh.(π * t / 2)) - atan.(2t)
        Mℓ_t[:, 2] .= exp.(2im * ϕ) .* Mℓ_t[:, 1]
    end

    # ℓ=2...ℓmax
    # Eq.(24)
    for iℓ in 2:ℓmax
        ℓ = iℓ - 1
        ϕℓ = atan.(2t / (2ℓ + 1))
        Mℓ_t[:, iℓ + 1] .= exp.(-2im * ϕℓ) .* Mℓ_t[:, iℓ - 1]
    end

    # Eq.(35)
    ℓs = 0:ℓmax
    xs = exp.(ρmin .+ κmin .+ collect(0:(2nr - 1)) .* Δρ)
    jℓ_x = [sphericalbesselj(ℓ, x) for x in xs, ℓ in ℓs]
    M̄ℓ_t = conj.(ifft(jℓ_x, 1)[1:(nr + 1), :])
    return (; Mℓ_t, M̄ℓ_t)
end

"""
Return the `k` values corresponding to a logarithmic grid `r`.
"""
function sbtfreq(r, kmax = 500)
    all(r .≈ logrange(minimum(r), maximum(r), length(r))) || error("Input r must be logarithmically spaced")
    ρmin = log(minimum(r))
    ρmax = log(maximum(r))
    Δρ = (ρmax - ρmin) / (length(r) - 1)
    κmax = log(kmax)
    κmin = κmax - (ρmax - ρmin)
    return exp(κmin) * exp.(collect(0:(length(r) - 1)) .* Δρ)
end

struct SBTPlan{T}
    ℓmax::Int
    r::Vector{T}
    rmin::T
    nr::Int
    Δρ::T
    k::Vector{T}
    kmin::T
    rext::Vector{T}
    post_div_fac::Vector{T}
    Mℓ_t::Matrix{Complex{T}}
    M̄ℓ_t::Matrix{Complex{T}}
    r32::Vector{T} # r^(3/2)
    rmin32::T # rmin^(3/2)
    rext32::Vector{T} # rext^(3/2)
end

function SBTPlan{T}(r::AbstractVector{T}, ℓmax::Int = 10, kmax::T = 500) where {T}
    if !all(r .≈ logrange(minimum(r), maximum(r), length(r)))
        error("Input r must be logarithmically spaced")
    end
    nr = length(r)
    rmin = minimum(r)

    ρmin = log(minimum(r))
    ρmax = log(maximum(r))
    Δρ = (ρmax - ρmin) / (nr - 1)

    κmax = log(kmax)
    κmin = κmax - (ρmax - ρmin)
    k = exp(κmin) * exp.(collect(0:(nr - 1)) .* Δρ)
    kmin = minimum(k)

    rext = minimum(r) * exp.(collect(-nr:-1) .* Δρ)
    post_div_fac = exp.(-collect(0:(nr - 1)) * 3 / 2 * Δρ)

    Δt = 2π / (2nr * Δρ)
    Mℓ_t, M̄ℓ_t = construct_Mℓ_t(ℓmax, nr, Δt, κmin, ρmin, Δρ)

    # Profiling showed that these are very expensive to compute
    # in the sbt function, so we precompute them here.
    r32 = r .^ (3 / 2)
    rmin32 = rmin^(3 / 2)
    rext32 = rext .^ (3 / 2)

    return SBTPlan{T}(
        ℓmax,
        r, rmin, nr, Δρ,
        k, kmin,
        rext, post_div_fac,
        Mℓ_t, M̄ℓ_t,
        r32, rmin32, rext32,
    )
end

@inbounds function sbt_large_k(
        ℓ::Integer,
        f::AbstractVector{T},
        plan::SBTPlan{T},
        np_in::Integer,
        direction::Symbol,
        normalize::Bool
    )::Vector{T} where {T}
    # Follows the procedure outlined in Sec.(3) of the paper following Eq.(32)
    sqrt_2_over_π = normalize ? sqrt(2 / T(π)) : T(1)
    if direction == :forward
        rmin = plan.rmin
        kmin = plan.kmin
        C = f[1] / plan.rmin^(np_in + ℓ)
    elseif direction == :inverse
        rmin = plan.kmin
        kmin = plan.rmin
        C = f[1] / plan.kmin^(np_in + ℓ)
    else
        error("Invalid direction: $direction")
    end
    # Step 1
    frα::Vector{T} = zeros(T, 2plan.nr)
    for i in 1:plan.nr
        frα[i] = C * plan.rext[i]^(np_in + ℓ) * plan.rext32[i] / plan.rmin32
    end
    for i in 1:plan.nr
        frα[plan.nr + i] = f[i] * plan.r32[i] / plan.rmin32
    end

    # Step 2
    y = rfft(frα)

    # Step 3
    yeM::Vector{Complex{T}} = zeros(Complex{T}, 2plan.nr)
    for i in 1:plan.nr
        yeM[i] = conj(y[i]) * plan.Mℓ_t[i, ℓ + 1]
    end

    # Steps 4 and 5
    z::Vector{T} = real.(ifft(yeM))
    g::Vector{T} = zeros(T, plan.nr)
    c::T = (rmin / kmin)^(3 / 2) * 2plan.nr * sqrt_2_over_π
    for i in 1:plan.nr
        g[i] = c * real(z[plan.nr + i]) * plan.post_div_fac[i]
    end
    return g
end

@inbounds function sbt_small_k(
        ℓ::Integer,
        f::AbstractVector{T},
        plan::SBTPlan{T},
        direction::Symbol,
        normalize::Bool
    )::Vector{T} where {T}
    sqrt_2_over_π = normalize ? sqrt(2 / T(π)) : T(1)
    frα::Vector{T} = zeros(T, 2plan.nr)
    if direction == :forward
        frα[1:plan.nr] .= plan.r .^ 3 .* f
    elseif direction == :inverse
        frα[1:plan.nr] .= plan.k .^ 3 .* f
    else
        error("Invalid direction: $direction")
    end
    y::Vector{Complex{T}} = rfft(frα)
    yeM::Vector{Complex{T}} = zeros(Complex{T}, plan.nr + 1)
    for i in eachindex(yeM)
        yeM[i] = conj(y[i]) * plan.M̄ℓ_t[i, ℓ + 1] * sqrt_2_over_π
    end
    z::Vector{T} = real.(irfft(yeM, 2plan.nr)) * 2plan.nr
    z[1:plan.nr] .*= plan.Δρ
    return z[1:plan.nr]
end

function sbt(
        ℓ::Integer,
        f::AbstractVector{T},
        plan::SBTPlan{T},
        np_in::Integer = 1; normalize = false,
        direction = :forward
    )::Vector{T} where {T}
    g_large_k = sbt_large_k(ℓ, f, plan, np_in, direction, normalize)
    g_small_k = sbt_small_k(ℓ, f, plan, direction, normalize)
    minloc = argmin(1:plan.nr) do i
        gi::Float64 = g_large_k[i]
        zi::Float64 = g_small_k[i]
        abs(gi - zi)
    end
    g_large_k[1:minloc] .= g_small_k[1:minloc]
    return g_large_k
end

function sbt(
        ℓ::Integer,
        f::AbstractVector{T},
        r::AbstractVector{T}; kmax = 500,
        np_in = 1,
        kwargs...
    ) where {T}
    plan = SBTPlan{T}(r, ℓ, convert(T, kmax))
    return sbt(ℓ, f, plan, np_in; kwargs...), plan.k
end

end # module SBT
