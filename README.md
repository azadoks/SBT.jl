# SBT.jl
Implementation of the Spherical Bessel Transform (SBT) on logarithmic radial grids as described in [1]. Largely adapted from the python implementation [pySBT](https://github.com/QijingZheng/pySBT).

[1] J. D. Talman, NumSBT: A subroutine for calculating spherical Bessel transforms numerically, Computer Physics Communications 180, 332 (2009).

## Quick start
```
# install the package
pkg> add https://github.com/azadoks/SBT.jl.git

# load the module
julia> using SBT

# set up a _logarithmic_ grid
julia> r = collect(logrange(1e-5, 20, 1000));

# evaluate a function on the grid
julia> f = 0.5 * 2.0^3 * exp.(-2.0 * r);

# Bessel order / angular momentum
julia> l = 0;

# transform!
julia> g, k = sbt(l, f, r);
```

## Planning an SBT
```
# set up
julia> using SBT
julia> lmax=5; kmax=500.0;
julia> r = collect(logrange(1e-5, 20, 1000));
julia> f = [0.5 * beta^3 * exp(-beta * ri) for ri in r, beta in (2, 3, 4)];

# create the SBT plan
julia> plan = SBTPlan(r, lmax, kmax)

# perform many transforms
julia> g = [sbt(0, f[:,i], plan) for i in axes(f, 2)]

# access the domain of g
julia> k = plan.k
```
