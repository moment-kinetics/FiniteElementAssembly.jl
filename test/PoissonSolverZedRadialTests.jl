"""
Module to test the solution of Poisson's equation
in cylindrical (zed,radial) coordinates.
"""
module PoissonSolverZedRadialTests

using FiniteElementAssembly: FiniteElementCoordinate,
        ScalarCoordinateInputs, include_boundary_points, exclude_lower_boundary_point,
        assemble_operator, DirichletBC, NaturalBC
using FiniteElementMatrices
using Test: @test, @testset
using LinearAlgebra: lu, mul!, ldiv!

struct PoissonZedRadialMatrices
    # mass matrix in zed
    M_zed::Array{Float64,3}
    # stiffness matrix in zed
    K_zed::Array{Float64,3}
    # mass matrix in radial
    M_radial::Array{Float64,3}
    # stiffness matrix in radial
    K_radial::Array{Float64,3}
    function PoissonZedRadialMatrices(zed::FiniteElementCoordinate,radial::FiniteElementCoordinate)
        M_zed = Array{Float64,3}(undef,zed.ngrid,zed.ngrid,zed.nelement)
        K_zed = Array{Float64,3}(undef,zed.ngrid,zed.ngrid,zed.nelement)
        for ielement_zed in 1:zed.nelement
            zdata = zed.element_data[ielement_zed]
            @views M_zed[:,:,ielement_zed] = finite_element_matrix(lagrange_x,lagrange_x,0,zdata)
            @views K_zed[:,:,ielement_zed] = -finite_element_matrix(d_lagrange_dx,d_lagrange_dx,0,zdata)
        end

        M_radial = Array{Float64,3}(undef,radial.ngrid,radial.ngrid,radial.nelement)
        K_radial = Array{Float64,3}(undef,radial.ngrid,radial.ngrid,radial.nelement)
        for ielement_radial in 1:radial.nelement
            rdata = radial.element_data[ielement_radial]
            @views M_radial[:,:,ielement_radial] = finite_element_matrix(lagrange_x,lagrange_x,1,rdata)
            @views K_radial[:,:,ielement_radial] = -finite_element_matrix(d_lagrange_dx,d_lagrange_dx,1,rdata)
        end
        return new(M_zed, K_zed, M_radial, K_radial)
    end
end
function runtests(; ngrid_zed=5, nelement_global_zed = 4, Lzed = 1.0,
    ngrid_radial=5, nelement_global_radial = 4, Lradial = 1.0, print_to_screen=false)
    # create the coordinates
    z_min = -0.5*Lzed
    z_max = 0.5*Lzed
    zed = FiniteElementCoordinate("zed", ScalarCoordinateInputs(ngrid_zed,
                                nelement_global_zed,
                                z_min,z_max,include_boundary_points))
    r_max = Lradial
    radial = FiniteElementCoordinate("radial", ScalarCoordinateInputs(ngrid_radial,
                                nelement_global_radial,
                                0.0,r_max,exclude_lower_boundary_point),
                                weight_function=((r)-> 2.0*pi*r))
    # allocate fields, noting order (zed,radial) must be the same as in operator definition
    phi = zeros(zed.n,radial.n)
    exact_phi = zeros(zed.n,radial.n)
    err_phi = zeros(zed.n,radial.n)
    exact_rho = zeros(zed.n,radial.n)
    err_rho = zeros(zed.n,radial.n)
    rho = zeros(zed.n,radial.n)
    rhszr = zeros(zed.n,radial.n)
    # calculate unassembled FE matrices needed for the Laplacian
    zrm = PoissonZedRadialMatrices(zed, radial)
    # the weak form for the Laplacian operator, on a single element
    # Laplacian[F] = \nabla^2 F = rho
    function Laplacian_weak_form(jzed, ized, ielement_zed,
                        jradial, iradial, ielement_radial)
        return (zrm.M_zed[jzed, ized, ielement_zed]*
                zrm.K_radial[jradial, iradial, ielement_radial] +
                zrm.K_zed[jzed, ized, ielement_zed]*
                zrm.M_radial[jradial, iradial, ielement_radial])
    end
    function mass_matrix(jzed, ized, ielement_zed,
                        jradial, iradial, ielement_radial)
        return (zrm.M_zed[jzed, ized, ielement_zed]*
                zrm.M_radial[jradial, iradial, ielement_radial])
    end
    zr2D_Laplacian_stiffness_matrix = assemble_operator(Laplacian_weak_form, zed, radial,
                                        NaturalBC(), NaturalBC())
    lu_Laplacian_Neumann_bc = lu(zr2D_Laplacian_stiffness_matrix)
    zr2D_Laplacian_stiffness_matrix_Dirichlet_bc = assemble_operator(Laplacian_weak_form,
                                                    zed, radial, DirichletBC(), DirichletBC())
    lu_Laplacian_Dirichlet_bc = lu(zr2D_Laplacian_stiffness_matrix_Dirichlet_bc)
    zr2D_mass_matrix = assemble_operator(mass_matrix, zed, radial, NaturalBC(), NaturalBC())
    #println(zr2D_mass_matrix)
    lu_mass_matrix = lu(zr2D_mass_matrix)
    # test with Neumann boundary conditions
    z0 = z_min
    z1 = z_max
    r1 = r_max
    for ir in 1:radial.n
        r = radial.grid[ir]
        for iz in 1:zed.n
            z = zed.grid[iz]
            exact_rho[iz,ir] = (2*r^2*(r - r1)^2*(6*z^2 - 6*z*z0 - 6*z*z1 + z0^2 + 4*z0*z1 + z1^2) +
                         2*(z - z0)^2*(z - z1)^2*(8*r^2 - 9*r*r1 + 2*r1^2))
            exact_phi[iz,ir] = r^2*(r - r1)^2*(z - z0)^2*(z - z1)^2
        end
    end
    # normalise so maximum phi is 1
    normalisation = maximum(exact_phi)
    @. exact_phi /= normalisation
    @. exact_rho /= normalisation

    @testset "Poisson Solver (Zed, Radial) Tests" begin
        println("Poisson Solver (Zed, Radial) Tests")
        # test Laplacian inversion, find phi from rho
        # get view in 1D vector
        exact_rhoc = vec(exact_rho)
        phic = vec(phi)
        rhsc = vec(rhszr)
        mul!(rhsc,zr2D_mass_matrix,exact_rhoc)
        # solve the linear system
        ldiv!(phic, lu_Laplacian_Neumann_bc, rhsc)
        # subtract z=z_min, r=r_max component
        # phi solution only defined up to constant due to
        # Neumann boundary conditions
        @. phi -= phi[1,end]
        @. err_phi = abs(phi - exact_phi)
        max_err_phi = maximum(err_phi)
        if print_to_screen
            println("max_err_phi = $max_err_phi")
        end
        @test max_err_phi < 1.0e-11

        # test Laplacian inversion, find phi from rho
        # get view in 1D vector
        exact_rhoc = vec(exact_rho)
        phic = vec(phi)
        rhsc = vec(rhszr)
        mul!(rhsc,zr2D_mass_matrix,exact_rhoc)
        # impose zero boundary condition
        @. rhszr[1,:] = 0.0
        @. rhszr[end,:] = 0.0
        @. rhszr[:,end] = 0.0
        # solve the linear system
        ldiv!(phic, lu_Laplacian_Dirichlet_bc, rhsc)
        # subtract z=z_min, r=r_max component
        # phi solution only defined up to constant due to
        # Neumann boundary conditions
        @. err_phi = abs(phi - exact_phi)
        max_err_phi = maximum(err_phi)
        if print_to_screen
            println("max_err_phi = $max_err_phi")
        end
        @test max_err_phi < 1.0e-11

        # test differentiation, find rho from phi
        # get view in 1D vector
        rhoc = vec(rho)
        exact_phic = vec(exact_phi)
        rhsc = vec(rhszr)
        mul!(rhsc,zr2D_Laplacian_stiffness_matrix,exact_phic)
        # solve the linear system
        ldiv!(rhoc, lu_mass_matrix, rhsc)

        # test the results
        @. err_rho = abs(rho - exact_rho)
        max_err_rho = maximum(err_rho)
        if print_to_screen
            println("max_err_rho = $max_err_rho")
        end
        @test max_err_rho < 1.0e-11
    end
    return nothing
end

end # PoissonSolverZedRadialTests

using .PoissonSolverZedRadialTests

PoissonSolverZedRadialTests.runtests()