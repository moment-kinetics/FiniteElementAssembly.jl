"""
Module to test interpolation convenience functions.
"""
module InterpolationTests

using Test: @testset, @test
using StableRNGs
using FiniteElementAssembly: FiniteElementCoordinate,
                            ScalarCoordinateInputs,
                            include_boundary_points, exclude_boundary_points,
                            exclude_lower_boundary_point, exclude_upper_boundary_point,
                            interpolate_1D, value_in_coordinate_domain


function runtests()
    @testset "Interpolation" begin
        println("Interpolation Tests")
        rng = StableRNG(43)        
        @testset "Interpolation, testing exact polynomials" begin
            @testset "$nelement $ngrid $boundary_points" for nelement in (1:5), ngrid in (3:11),
                    boundary_points in (include_boundary_points, exclude_boundary_points,
                            exclude_lower_boundary_point, exclude_upper_boundary_point)
                tolerance = 1.0e-13
                # define inputs needed for the test
                coord_min = -1.0
                coord_max = 1.0
                # create the coordinate struct 'x'
                xcoord = FiniteElementCoordinate("xcoord", ScalarCoordinateInputs(ngrid,
                                      nelement, coord_min, coord_max, boundary_points))
                xdata = zeros(Float64, xcoord.n)
                ninterp = 100
                nextrap = 10
                interpolation_points = LinRange(coord_min, coord_max, ninterp)
                extrapolation_points_upper = LinRange(coord_max*(1.0+1.0e-8), 2.0*coord_max, nextrap)
                extrapolation_points_lower = LinRange(2.0*coord_min, coord_min*(1.0+1.0e-8), nextrap)
                # test polynomials up to order ngrid-1
                coefficients = zeros(ngrid)
                for n in 1:ngrid
                    # calculate a new random coefficient
                    coefficients[n] = randn(rng)
                    # function returning a polynomial of order n-1
                    function test_polynomial(x)
                        poly = 0.0
                        for i in 1:n
                            poly += coefficients[i]*x^(i-1)
                        end
                        return poly
                    end
                    # fill in the function to interpolate
                    @. xdata = test_polynomial(xcoord.grid)
                    # test the interpolation for each point
                    # use success variable to group test results
                    # to avoid very large numbers of tests being reported
                    success = true
                    for j in 1:ninterp
                        xj = interpolation_points[j]
                        polyj = interpolate_1D(xj, xdata, xcoord)
                        success = success && abs(polyj - test_polynomial(xj)) < tolerance
                    end
                    @test success
                    success = true
                    # check the extrapolated points are outside the domain
                    for j in 1:nextrap
                        xj = extrapolation_points_upper[j]
                        success = success && !value_in_coordinate_domain(xj, xcoord)
                        xj = extrapolation_points_lower[j]
                        success = success && !value_in_coordinate_domain(xj, xcoord)
                    end
                    @test success
                end
            end
        end
        @testset "Interpolation, testing trigonometric function" begin
            @testset "$nelement $ngrid $boundary_points" for nelement in (1:5), ngrid in (20:25),
                    boundary_points in (include_boundary_points, exclude_boundary_points,
                            exclude_lower_boundary_point, exclude_upper_boundary_point)
                tolerance = 1.0e-12
                # define inputs needed for the test
                coord_min = -1.0
                coord_max = 1.0
                # create the coordinate struct 'x'
                xcoord = FiniteElementCoordinate("xcoord", ScalarCoordinateInputs(ngrid,
                                      nelement, coord_min, coord_max, boundary_points))
                xdata = zeros(Float64, xcoord.n)
                ninterp = 100
                nextrap = 10
                interpolation_points = LinRange(coord_min, coord_max, ninterp)
                extrapolation_points_upper = LinRange(coord_max*(1.0+1.0e-8), 2.0*coord_max, nextrap)
                extrapolation_points_lower = LinRange(2.0*coord_min, coord_min*(1.0+1.0e-8), nextrap)
                # calculate a new random coefficient
                phase = randn(rng)
                function test_function(x)
                    return sinpi(2*x/xcoord.L + phase)
                end
                # fill in the function to interpolate
                @. xdata = test_function(xcoord.grid)
                # test the interpolation for each point
                # use success variable to group test results
                # to avoid very large numbers of tests being reported
                success = true
                for j in 1:ninterp
                    xj = interpolation_points[j]
                    polyj = interpolate_1D(xj, xdata, xcoord)
                    success = success && abs(polyj - test_function(xj)) < tolerance
                end
                @test success
                success = true
                # check the extrapolated points are outside the domain
                for j in 1:nextrap
                    xj = extrapolation_points_upper[j]
                    success = success && !value_in_coordinate_domain(xj, xcoord)
                    xj = extrapolation_points_lower[j]
                    success = success && !value_in_coordinate_domain(xj, xcoord)
                end
                @test success
            end
        end
    end
    return nothing
end

end # module InterpolationTests

using .InterpolationTests

InterpolationTests.runtests()