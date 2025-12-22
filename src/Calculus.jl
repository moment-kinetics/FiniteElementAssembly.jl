# Convenience functions for definite integrals and first derivatives.
export integral, first_derivative!

"""
Compute the 1D integral `∫ dvx prefactor(vx)*integrand(vx)`
This function is provided for convenience, for calculating diagnostic
integrals quickly. We use the weights associated with the collocation
points in vx. As a result, this means that this integral
is only exact for polynomials with order up to 2N-3 where
N=`ngrid` is the number of collocation points per element.
"""
function integral(prefactor::Function,
                integrand::AbstractArray{Float64,1},
                vx::FiniteElementCoordinate)
    @boundscheck (vx.n,) == size(integrand) || throw(BoundsError(integrand))
    integral = 0.0
    for ivx ∈ eachindex(vx.grid)
        integral += prefactor(vx.grid[ivx])::Float64 *
                    integrand[ivx] * vx.wgts[ivx]
    end
    return integral
end

"""
Compute the 2D integral `∫ dvx.dvy prefactor(vx,vy)*integrand(vx,vy)`
This function is provided for convenience, for calculating diagnostic
integrals quickly. We use the weights associated with the collocation
points in vx and vy. As a result, this means that this integral
is only exact for polynomials with order up to 2N-3 where
N=`ngrid` is the number of collocation points per element.
"""
function integral(prefactor::Function,
                integrand::AbstractArray{Float64,2},
                vx::FiniteElementCoordinate,
                vy::FiniteElementCoordinate)
    @boundscheck (vx.n, vy.n) == size(integrand) || throw(BoundsError(integrand))
    integral = 0.0
    for ivy ∈ eachindex(vy.grid), ivx ∈ eachindex(vx.grid)
        integral += prefactor(vx.grid[ivx], vy.grid[ivy])::Float64 *
                    integrand[ivx, ivy] * vx.wgts[ivx] * vy.wgts[ivy]
    end
    return integral
end

"""
Convenience function for taking a first derivative in 1D.
"""
function first_derivative!(df::AbstractArray{Float64,1},
                        f::AbstractArray{Float64,1},
                        coord::FiniteElementCoordinate)
    derivative_data = coord.derivative_data
    dummy_rhs = derivative_data.dummy_rhs
    dummy_df = derivative_data.dummy_df
    PP1D = derivative_data.PP1D
    lu_MM1D = derivative_data.lu_MM1D
    mul!(dummy_rhs,PP1D,f)
    ldiv!(dummy_df,lu_MM1D,dummy_rhs)
    @. df = dummy_df
    return nothing
end
