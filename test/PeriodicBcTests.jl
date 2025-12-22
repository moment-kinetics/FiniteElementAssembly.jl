"""
Module to test differentiation in 1D with periodic boundary conditions.
"""
module PeriodicBcTests

using Test: @testset, @test
using FiniteElementAssembly

function periodic_differentiation_1D_test(; ngrid = 3, nelement = 1, atol = 1.0e-13)
    # define inputs needed for the test
    coord_min = -1.0
    coord_max = 1.0
    # create the coordinate struct 'x'
    x = FiniteElementCoordinate("coord", ScalarCoordinateInputs(ngrid,
                            nelement, coord_min, coord_max, include_boundary_points),
                            bc = PeriodicBC())

    # a periodic function
    f = zeros(x.n)
    df = zeros(x.n)
    df_num = zeros(x.n)
    df_err = zeros(x.n)
    fac1 = 0.1
    fac2 = 0.9
    for ix in 1:x.n
        xarg = 2.0*x.grid[ix]/x.L
        f[ix] = fac1*cospi(xarg) + fac2*sinpi(xarg)
        df[ix] = (2.0*pi/x.L)*(fac2*cospi(xarg) - fac1*sinpi(xarg)) 
    end
    # differentiate f
    first_derivative!(df_num, f, x)
    @. df_err = abs(df - df_num)
    max_df_err = maximum(df_err)
    @test max_df_err < atol
    @test abs(df[1]-df[end]) < 1.0e-14
    return nothing
end

function runtests()
    @testset "Periodic differentiation 1D" begin
        println("Periodic Differentiation 1D Tests")
        periodic_differentiation_1D_test(; ngrid = 25, nelement = 1)
        periodic_differentiation_1D_test(; ngrid = 25, nelement = 2)
        periodic_differentiation_1D_test(; ngrid = 25, nelement = 3, atol = 1.0e-12)
    end
    return nothing
end

end # PeriodicBcTests

using .PeriodicBcTests

PeriodicBcTests.runtests()