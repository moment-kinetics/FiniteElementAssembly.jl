"""
Module to test the implementation of the FiniteElementCoordinate
interface allowing for scalar coordinate inputs
(number of grid points per element, number of elements, et cetera)
for uniformly spaced element boundaries, and a more general constructor
function where the element can location points and scale and shift
factors are specified directly.
"""
module InterfaceTests

using FiniteElementAssembly: FiniteElementCoordinate,
        ScalarCoordinateInputs, include_boundary_points, exclude_lower_boundary_point
using Test: @test, @testset

function test_coordinates(coord,coordnew)
    @test coord.name == coordnew.name
    @test coord.n == coordnew.n
    @test coord.ngrid == coordnew.ngrid
    @test coord.nelement == coordnew.nelement
    @test coord.bc == coordnew.bc
    @test isapprox(coord.L,coordnew.L,atol=2.0e-15)
    @test isapprox(coord.grid,coordnew.grid,atol=2.0e-15)
    @test isapprox(coord.wgts,coordnew.wgts,atol=2.0e-14) # relax tolerance here as wgts computed by different method
    @test isapprox(coord.element_scale,coordnew.element_scale,atol=2.0e-15)
    @test isapprox(coord.element_shift,coordnew.element_shift,atol=2.0e-15)
    @test isapprox(coord.element_boundaries,coordnew.element_boundaries,atol=2.0e-15)
    @test isapprox(coord.igrid,coordnew.igrid,atol=0)
    @test isapprox(coord.igrid_full,coordnew.igrid_full,atol=0)
    @test isapprox(coord.imin,coordnew.imin,atol=0)
    @test isapprox(coord.imax,coordnew.imax,atol=0)
    for j in 1:coord.nelement
        test_lpoly_data(coord.lpoly_data[j],coordnew.lpoly_data[j],coord.ngrid)
        test_lpoly_data(coord.element_data[j].lpoly_data,coordnew.element_data[j].lpoly_data,coord.ngrid)
        @test isapprox(coord.element_data[j].scale,coordnew.element_data[j].scale,atol=2.0e-15)
        @test isapprox(coord.element_data[j].shift,coordnew.element_data[j].shift,atol=2.0e-15)
    end
    return nothing
end

function test_lpoly_data(lpoly_data,lpoly_data_new,ngrid)
    @test isapprox(lpoly_data.x_nodes,
                   lpoly_data_new.x_nodes,
                    atol=2.0e-15)
    for i in 1:ngrid
        @test isapprox(lpoly_data.lpoly_data[i].other_nodes,
                lpoly_data_new.lpoly_data[i].other_nodes,
                atol=2.0e-15)
        @test isapprox(lpoly_data.lpoly_data[i].other_nodes_derivative,
                lpoly_data_new.lpoly_data[i].other_nodes_derivative,
                atol=2.0e-15)
        @test isapprox(lpoly_data.lpoly_data[i].one_over_denominator,
                lpoly_data_new.lpoly_data[i].one_over_denominator,
                atol=2.0e-15)
    end
    return nothing
end
function runtests()
    ngrid = 5
    nelement_global_vperp = 2
    Lvperp = 3.0

    nelement_global_vpa = 4
    Lvpa = 6.0
    # create the coordinate structs
    vperp = FiniteElementCoordinate("vperp", ScalarCoordinateInputs(ngrid,
                                nelement_global_vperp,
                                0.0,Lvperp,exclude_lower_boundary_point),
                                weight_function=((vperp)-> 2.0*pi*vperp))
    vpa = FiniteElementCoordinate("vpa", ScalarCoordinateInputs(ngrid,
                                nelement_global_vpa,
                                -0.5*Lvpa,0.5*Lvpa,include_boundary_points))


    vperpnew = FiniteElementCoordinate("vperp",vperp.element_data,weight_function=((vperp)-> 2.0*pi*vperp))
    vpanew = FiniteElementCoordinate("vpa",vpa.element_data)

    vpa_grid_expected = [-3.0, -2.740990253030983, -2.25, -1.7590097469690171, -1.5, -1.2409902530309829, -0.75, -0.25900974696901713, 0.0, 0.25900974696901713, 0.75, 1.2409902530309829, 1.5, 1.7590097469690171, 2.25, 2.740990253030983, 3.0]
    vpa_wgts_expected = [0.07500000000000004, 0.40833333333333344, 0.5333333333333337, 0.4083333333333334, 0.15000000000000008, 0.40833333333333344, 0.5333333333333337, 0.4083333333333334, 0.15000000000000008, 0.40833333333333344, 0.5333333333333337, 0.4083333333333334, 0.15000000000000008, 0.40833333333333344, 0.5333333333333337, 0.4083333333333334, 0.07500000000000002]
    vperp_grid_expected = [0.08565629417177645, 0.41526452045718576, 0.8753856485533752, 1.2903602034843291, 1.5, 1.7590097469690171, 2.25, 2.740990253030983, 3.0]
    vperp_wgts_expected = [0.11601874375179486, 1.1011643914948166, 2.572667154550636, 2.713246503133625, 1.272345024703867, 4.51297521391441, 7.539822368615508, 7.032377788028081, 1.4137166941154073]
    @testset "Coordinate Creation Interface Tests" begin
        println("Interface Tests: coordinate definition tests")
        @testset "vpa coordinate creation" begin
            println("    - test vpa coordinate creation")
            test_coordinates(vpa,vpanew)
            @test isapprox(vpa.grid, vpa_grid_expected, atol=2.0e-15)
            @test isapprox(vpa.wgts, vpa_wgts_expected, atol=2.0e-15)
        end
        @testset "vperp coordinate creation" begin
            println("    - test vperp coordinate creation")
            test_coordinates(vperp,vperpnew)
            @test isapprox(vperp.grid, vperp_grid_expected, atol=2.0e-15)
            @test isapprox(vperp.wgts, vperp_wgts_expected, atol=4.0e-15)
        end
    end
    return nothing
end
end # InterfaceTest


using .InterfaceTests

InterfaceTests.runtests()