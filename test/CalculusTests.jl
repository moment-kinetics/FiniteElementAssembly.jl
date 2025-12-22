"""
Module to test first derivatives and integral convenience functions in one dimension.
"""
module CalculusTests

using Test: @testset, @test
using StableRNGs
using FiniteElementAssembly: FiniteElementCoordinate,
                            first_derivative!,
                            ScalarCoordinateInputs,
                            include_boundary_points, exclude_boundary_points,
                            exclude_lower_boundary_point, exclude_upper_boundary_point
using FiniteElementAssembly: integral

"""
Pass this function to the `norm` argument of `isapprox()` to test the maximum error
between two arrays.
"""
maxabs_norm(x) = maximum(abs.(x))

function runtests()
    @testset "calculus" begin
        println("Calculus Tests")
        rng = StableRNG(42)

        @testset "GaussLegendre derivatives and fundamental theorem of calculus, testing exact polynomials" begin
            @testset "$nelement $ngrid $boundary_points" for nelement ∈ (1:5), ngrid ∈ (3:17),
                    boundary_points in (include_boundary_points, exclude_boundary_points,
                            exclude_lower_boundary_point, exclude_upper_boundary_point)

                # define inputs needed for the test
                coord_min = -1.0
                coord_max = 1.0
                # create the coordinate struct 'x'
                x = FiniteElementCoordinate("coord", ScalarCoordinateInputs(ngrid,
                                      nelement, coord_min, coord_max, boundary_points))
                # test polynomials up to order ngrid-1
                for n ∈ 0:ngrid-1
                    # create array for the function f(x) to be differentiated/integrated
                    f = Array{Float64,1}(undef, x.n)
                    # create array for the derivative df/dx and the expected result
                    df = similar(f)
                    expected_df = similar(f)
                    # initialize f, expected df, and expected integral of df
                    # n.b., intdf /= f[end] - f[1] when endpoints are not included
                    f[:] .= randn(rng)
                    expected_df .= 0.0
                    expected_intdf = 0.0
                    for p ∈ 1:n
                        coefficient = randn(rng)
                        @. f += coefficient * x.grid ^ p
                        @. expected_df += coefficient * p * x.grid ^ (p - 1)
                        expected_intdf += coefficient * (x.element_boundaries[end] ^ p
                                                        - x.element_boundaries[1] ^ p)
                    end
                    # differentiate f
                    first_derivative!(df, f, x)

                    # Note the error we might expect for a p=32 polynomial is probably
                    # something like p*(round-off) for x^p (?) so error on expected_df would
                    # be p*p*(round-off), or plausibly 1024*(round-off), so tolerance of
                    # 2e-11 isn't unreasonable.
                    @test isapprox(df, expected_df, rtol=2.0e-11, atol=6.0e-12,
                                   norm=maxabs_norm)

                    # integrate df/dx
                    intdf = integral((x->1.0),df, x)
                    # test int d f / d x = f(x_max) - f(x_min)
                    @test abs(intdf - expected_intdf) < 7.0e-14
                end
            end
        end
    end
end

end # CalculusTests


using .CalculusTests

CalculusTests.runtests()
