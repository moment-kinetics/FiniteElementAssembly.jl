"""
Module to test the solution of Poisson's equation
in (theta,zed) coordinates with
periodic boundary conditions in theta.
"""
module PoissonSolverThetaZedTests

using FiniteElementAssembly: FiniteElementCoordinate,
        ScalarCoordinateInputs, include_boundary_points,
        assemble_operator, DirichletBC, PeriodicBC, NaturalBC
using FiniteElementMatrices
using Test: @test, @testset
using LinearAlgebra: lu, mul!, ldiv!

struct PoissonThetaZedMatrices
    # mass matrix in theta
    M_theta::Array{Float64,3}
    # stiffness matrix in theta
    K_theta::Array{Float64,3}
    # mass matrix in zed
    M_zed::Array{Float64,3}
    # stiffness matrix in zed
    K_zed::Array{Float64,3}
    function PoissonThetaZedMatrices(theta::FiniteElementCoordinate,zed::FiniteElementCoordinate)
        M_theta = Array{Float64,3}(undef,theta.ngrid,theta.ngrid,theta.nelement)
        K_theta = Array{Float64,3}(undef,theta.ngrid,theta.ngrid,theta.nelement)
        for ielement_theta in 1:theta.nelement
            theta_data = theta.element_data[ielement_theta]
            @views M_theta[:,:,ielement_theta] = finite_element_matrix(lagrange_x,lagrange_x,0,theta_data)
            @views K_theta[:,:,ielement_theta] = -finite_element_matrix(d_lagrange_dx,d_lagrange_dx,0,theta_data)
        end
        M_zed = Array{Float64,3}(undef,zed.ngrid,zed.ngrid,zed.nelement)
        K_zed = Array{Float64,3}(undef,zed.ngrid,zed.ngrid,zed.nelement)
        for ielement_zed in 1:zed.nelement
            zed_data = zed.element_data[ielement_zed]
            @views M_zed[:,:,ielement_zed] = finite_element_matrix(lagrange_x,lagrange_x,0,zed_data)
            @views K_zed[:,:,ielement_zed] = -finite_element_matrix(d_lagrange_dx,d_lagrange_dx,0,zed_data)
        end
        return new(M_theta, K_theta, M_zed, K_zed)
    end
end

function runtests(; ngrid_theta=15, nelement_theta = 4, Ltheta = 1.0,
    ngrid_zed=5, nelement_zed = 8, Lzed = 1.0, print_to_screen=false)
    # create the coordinates
    theta = FiniteElementCoordinate("theta", ScalarCoordinateInputs(ngrid_theta,
                                nelement_theta,
                                -0.5*Ltheta,0.5*Ltheta,include_boundary_points))
    zed = FiniteElementCoordinate("zed", ScalarCoordinateInputs(ngrid_zed,
                                nelement_zed,
                                0.0,Lzed,include_boundary_points))
    # allocate fields, noting order (theta,zed) must be the same as in operator definition
    phi = zeros(theta.n,zed.n)
    exact_phi = zeros(theta.n,zed.n)
    err_phi = zeros(theta.n,zed.n)
    rho = zeros(theta.n,zed.n)
    rhs_thetaz = zeros(theta.n,zed.n)
    # calculate unassembled FE matrices needed for the Laplacian
    tzm = PoissonThetaZedMatrices(theta, zed)
    # the weak form for the Laplacian operator, on a single element
    # Laplacian[F] = d^2 F / d z^2 + d^2 F / d theta^2
    # Laplacian[F] = rho(theta,z)
    function Laplacian_weak_form(jtheta, itheta, ielement_theta,
                        jzed, ized, ielement_zed)
        return (tzm.M_theta[jtheta, itheta, ielement_theta]*
                tzm.K_zed[jzed, ized, ielement_zed] +
                tzm.K_theta[jtheta, itheta, ielement_theta]*
                tzm.M_zed[jzed, ized, ielement_zed])
    end
    function mass_matrix(jtheta, itheta, ielement_theta,
                        jzed, ized, ielement_zed)
        return (tzm.M_theta[jtheta, itheta, ielement_theta]*
                tzm.M_zed[jzed, ized, ielement_zed])
    end
    tz2D_Laplacian_stiffness_matrix_Dirichlet_bc = assemble_operator(Laplacian_weak_form,
                                                    theta, zed, PeriodicBC(), DirichletBC())
    lu_Laplacian_Dirichlet_bc = lu(tz2D_Laplacian_stiffness_matrix_Dirichlet_bc)
    tz2D_mass_matrix = assemble_operator(mass_matrix, theta, zed, PeriodicBC(), NaturalBC())

    function test_Poisson(;
                # parameters controlling the test functions
                kk=1::Int64, phase = 0.0,
                # tolerance
                atol = 1.0e-11)
        # initialise phi and source rho
        for ized in 1:zed.n
            for itheta in 1:theta.n
                exact_phi[itheta,ized] = ((zed.grid[end] - zed.grid[ized])*(zed.grid[ized] - zed.grid[1]) *
                                            cos(2.0*pi*kk*theta.grid[itheta]/theta.L + phase))
                rho[itheta,ized] = (( -2.0 - ((2.0*pi*kk/theta.L)^2)*(zed.grid[end] - zed.grid[ized])*(zed.grid[ized] - zed.grid[1]) ) *
                                    cos(2.0*kk*pi*theta.grid[itheta]/theta.L  + phase))
            end
        end
        # test Laplacian inversion, find phi from rho
        # get view in 1D vector
        rhoc = vec(rho)
        phic = vec(phi)
        rhsc = vec(rhs_thetaz)
        mul!(rhsc,tz2D_mass_matrix,rhoc)
        # impose zero boundary condition
        @. rhs_thetaz[:,1] = 0.0
        @. rhs_thetaz[:,end] = 0.0
        # impose periodic boundary condition
        @. rhs_thetaz[1,:] = 0.0
        # solve the linear system
        ldiv!(phic, lu_Laplacian_Dirichlet_bc, rhsc)
        # test the solution
        @. err_phi = abs(phi - exact_phi)
        max_err_phi = maximum(err_phi)
        if print_to_screen
            println("max_err_phi = $max_err_phi")
        end
        @test max_err_phi < 1.0e-11
        return nothing
    end
    @testset "Poisson Solver (Theta, Zed) Tests" begin
        println("Poisson Solver (Theta, Zed) Tests")
        test_Poisson(kk=1, phase = 0.0, atol = 1.0e-11)
        test_Poisson(kk=1, phase = pi/4.0, atol = 1.0e-11)
        test_Poisson(kk=1, phase = pi/3.0, atol = 1.0e-11)
        test_Poisson(kk=1, phase = pi/2.0, atol = 1.0e-11)
        test_Poisson(kk=2, phase = 0.0, atol = 1.0e-11)
        test_Poisson(kk=2, phase = pi/4.0, atol = 1.0e-11)
        test_Poisson(kk=2, phase = pi/3.0, atol = 1.0e-11)
        test_Poisson(kk=2, phase = pi/2.0, atol = 1.0e-11)
        test_Poisson(kk=3, phase = 0.0, atol = 1.0e-11)
        test_Poisson(kk=3, phase = pi/4.0, atol = 1.0e-11)
        test_Poisson(kk=3, phase = pi/3.0, atol = 1.0e-11)
        test_Poisson(kk=3, phase = pi/2.0, atol = 1.0e-11)
    end
    return nothing
end

end # PoissonSolverThetaZedTests

using .PoissonSolverThetaZedTests

PoissonSolverThetaZedTests.runtests()