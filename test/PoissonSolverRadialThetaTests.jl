"""
Module to test the solution of Poisson's equation
in cylindrical (radial,theta) coordinates with
periodic boundary conditions in theta.
"""
module PoissonSolverRadialThetaTests

using FiniteElementAssembly: FiniteElementCoordinate,
        ScalarCoordinateInputs, include_boundary_points, exclude_lower_boundary_point,
        assemble_operator, DirichletBC, PeriodicBC, NaturalBC
using FiniteElementMatrices
using Test: @test, @testset
using LinearAlgebra: lu, mul!, ldiv!

struct PoissonThetaRadialMatrices
    # mass matrix in theta
    M_theta::Array{Float64,3}
    # stiffness matrix in theta
    K_theta::Array{Float64,3}
    # mass matrix in radial
    M_radial::Array{Float64,3}
    # mass matrix in radial for theta stiffness term
    N_radial::Array{Float64,3}
    # stiffness matrix in radial
    K_radial::Array{Float64,3}
    function PoissonThetaRadialMatrices(radial::FiniteElementCoordinate,theta::FiniteElementCoordinate)
        M_theta = Array{Float64,3}(undef,theta.ngrid,theta.ngrid,theta.nelement)
        K_theta = Array{Float64,3}(undef,theta.ngrid,theta.ngrid,theta.nelement)
        for ielement_theta in 1:theta.nelement
            theta_data = theta.element_data[ielement_theta]
            @views M_theta[:,:,ielement_theta] = finite_element_matrix(lagrange_x,lagrange_x,0,theta_data)
            @views K_theta[:,:,ielement_theta] = -finite_element_matrix(d_lagrange_dx,d_lagrange_dx,0,theta_data)
        end
        M_radial = Array{Float64,3}(undef,radial.ngrid,radial.ngrid,radial.nelement)
        N_radial = Array{Float64,3}(undef,radial.ngrid,radial.ngrid,radial.nelement)
        K_radial = Array{Float64,3}(undef,radial.ngrid,radial.ngrid,radial.nelement)
        for ielement_radial in 1:radial.nelement
            rdata = radial.element_data[ielement_radial]
            @views M_radial[:,:,ielement_radial] = finite_element_matrix(lagrange_x,lagrange_x,2,rdata)
            @views N_radial[:,:,ielement_radial] = finite_element_matrix(lagrange_x,lagrange_x,0,rdata)
            @views K_radial[:,:,ielement_radial] = -(finite_element_matrix(d_lagrange_dx,d_lagrange_dx,2,rdata)
                                                    + finite_element_matrix(d_lagrange_dx,lagrange_x,1,rdata))
        end
        return new(M_theta, K_theta, M_radial, N_radial, K_radial)
    end
end

function runtests(; ngrid_theta=15, nelement_theta = 4, Ltheta = 1.0,
    ngrid_radial=7, nelement_radial = 4, Lradial = 1.0, print_to_screen=false)
    # create the coordinates
    theta = FiniteElementCoordinate("theta", ScalarCoordinateInputs(ngrid_theta,
                                nelement_theta,
                                -0.5*Ltheta,0.5*Ltheta,include_boundary_points))
    radial = FiniteElementCoordinate("radial", ScalarCoordinateInputs(ngrid_radial,
                                nelement_radial,
                                0.0,Lradial,exclude_lower_boundary_point),
                                weight_function=((r)-> 2.0*pi*r))
    # allocate fields, noting order (theta,radial) must be the same as in operator definition
    phi = zeros(radial.n,theta.n)
    exact_phi = zeros(radial.n,theta.n)
    err_phi = zeros(radial.n,theta.n)
    rho = zeros(radial.n,theta.n)
    rhs_rtheta = zeros(radial.n,theta.n)
    # calculate unassembled FE matrices needed for the Laplacian
    trm = PoissonThetaRadialMatrices(radial, theta)
    # the weak form for the Laplacian operator, on a single element
    # r^2 Laplacian[F] = r^2 rho
    # r^2 Laplacian[F] = r d( r d F / d r ) + d^2 F / d theta^2 = r^2 rho
    function Laplacian_weak_form(jradial, iradial, ielement_radial,
                jtheta, itheta, ielement_theta)
        return (trm.M_theta[jtheta, itheta, ielement_theta]*
                trm.K_radial[jradial, iradial, ielement_radial] +
                trm.K_theta[jtheta, itheta, ielement_theta]*
                trm.N_radial[jradial, iradial, ielement_radial])
    end
    function mass_matrix(jradial, iradial, ielement_radial,
                jtheta, itheta, ielement_theta,)
        return (trm.M_theta[jtheta, itheta, ielement_theta]*
                trm.M_radial[jradial, iradial, ielement_radial])
    end
    rt2D_Laplacian_stiffness_matrix_Dirichlet_bc = assemble_operator(Laplacian_weak_form,
                                                    radial, theta, DirichletBC(), PeriodicBC())
    lu_Laplacian_Dirichlet_bc = lu(rt2D_Laplacian_stiffness_matrix_Dirichlet_bc)
    rt2D_mass_matrix = assemble_operator(mass_matrix, radial, theta, NaturalBC(), PeriodicBC())
    function test_Poisson(;
                # parameters controlling the test functions
                kk=2::Int64, phase = 0.0,
                # tolerance
                atol = 1.0e-11)
        if kk < 2
            error("kk >=2 required for test")
        end
        for itheta in 1:theta.n
            for iradial in 1:radial.n
                exact_phi[iradial,itheta] = (1.0 - radial.grid[iradial])*(radial.grid[iradial]^kk)*cos(2.0*pi*kk*theta.grid[itheta]/theta.L + phase)
                rho[iradial,itheta] = (((2.0*pi*kk/theta.L)^2 - (kk+1)^2)*(radial.grid[iradial]^(kk-1)) -
                                    ((2.0*pi*kk/theta.L)^2 - (kk)^2)*(radial.grid[iradial]^(kk-2))) *
                                    cos(2.0*kk*pi*theta.grid[itheta]/theta.L  + phase)
            end
        end
        # test Laplacian inversion, find phi from rho
        # get view in 1D vector
        rhoc = vec(rho)
        phic = vec(phi)
        rhsc = vec(rhs_rtheta)
        mul!(rhsc,rt2D_mass_matrix,rhoc)
        # impose zero boundary condition
        @. rhs_rtheta[end,:] = 0.0
        # impose periodic boundary condition
        @. rhs_rtheta[:,1] = 0.0
        # solve the linear system
        ldiv!(phic, lu_Laplacian_Dirichlet_bc, rhsc)
        # test the solution
        @. err_phi = abs(phi - exact_phi)
        max_err_phi = maximum(err_phi)
        if print_to_screen
            println("max_err_phi = $max_err_phi")
        end
        @test max_err_phi < atol
        return nothing
    end

    @testset "Poisson Solver (Radial, Theta) Tests" begin
        println("Poisson Solver (Radial, Theta) Tests")
        test_Poisson(kk=2, phase = 0.0)
        test_Poisson(kk=2, phase = pi/4.0)
        test_Poisson(kk=2, phase = pi/3.0)
        test_Poisson(kk=2, phase = pi/2.0)
        test_Poisson(kk=3, phase = 0.0)
        test_Poisson(kk=3, phase = pi/4.0)
        test_Poisson(kk=3, phase = pi/3.0)
        test_Poisson(kk=3, phase = pi/2.0)
        test_Poisson(kk=4, phase = 0.0, atol=1.0e-10)
        test_Poisson(kk=4, phase = pi/4.0, atol=1.0e-10)
        test_Poisson(kk=4, phase = pi/3.0, atol=1.0e-10)
        test_Poisson(kk=4, phase = pi/2.0, atol=1.0e-10)
    end
    return nothing
end

end # PoissonSolverRadialThetaTests

using .PoissonSolverRadialThetaTests

PoissonSolverRadialThetaTests.runtests()