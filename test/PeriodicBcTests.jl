"""
Module to test differentiation in 1D with periodic boundary conditions.
"""
module PeriodicBcTests

using Test: @testset, @test
using LinearAlgebra
using FiniteElementAssembly
using FiniteElementMatrices

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

function periodic_ODE_solve_1D_test(; ngrid = 20, nelement = 1, atol = 1.0e-13)
    # define inputs needed for the test
    coord_min = -1.0
    coord_max = 1.0
    # create the coordinate struct 'x'
    x = FiniteElementCoordinate("coord", ScalarCoordinateInputs(ngrid,
                            nelement, coord_min, coord_max, include_boundary_points),
                            bc = PeriodicBC())

    # a test of periodic functions
    # solve d^2 phi / d x^2 = f = sin(2pi x/L + phase)
    # exact solution phi(x) = A + B z - (L/2pi)^2 sin(2pi x/L + phase )
    f = zeros(x.n)
    phi = zeros(x.n)
    exact_phi = zeros(x.n)
    phase = pi/4.0
    for ix in 1:x.n
        xarg = 2.0*x.grid[ix]/x.L + phase
        f[ix] = sinpi(xarg)
        exact_phi[ix] = - (x.L/(2.0*pi))^2*sinpi(xarg)
    end
    # dummy array
    weak_rhs = zeros(x.n)
    
    # required weak matrices
    M_x = Array{Float64,3}(undef,x.ngrid,x.ngrid,x.nelement)
    K_x = Array{Float64,3}(undef,x.ngrid,x.ngrid,x.nelement)
    for ielement_x in 1:x.nelement
        x_data = x.element_data[ielement_x]
        @views M_x[:,:,ielement_x] = finite_element_matrix(lagrange_x,lagrange_x,0,x_data)
        @views K_x[:,:,ielement_x] = -finite_element_matrix(d_lagrange_dx,d_lagrange_dx,0,x_data)
    end
    # assembled sparse matrices
    mass_matrix = assemble_operator(M_x, x, PeriodicBC())
    stiffness_matrix = assemble_operator(K_x, x, PeriodicBC())
    lu_mass_matrix = lu(mass_matrix)
    # Dirichlet BC.
    stiffness_matrix[1,:] .= 0.0
    stiffness_matrix[1,1] = 1.0
    stiffness_matrix[end,:] .= 0.0
    stiffness_matrix[end,end] = 1.0
    lu_Laplacian = lu(stiffness_matrix)
    
    # form the weak RHS
    mul!(weak_rhs,mass_matrix,f)
    # impose the Dirichlet BC to fix the constants A and B
    weak_rhs[1] = exact_phi[1]
    weak_rhs[end] = exact_phi[end]
    # solve the system
    ldiv!(phi, lu_Laplacian, weak_rhs)
    #println(phi)
    #println(exact_phi)
    max_err_phi = maximum(abs.(phi .- exact_phi))
    # println(max_err_phi)

    # check forward differentiation
    stiffness_matrix_differentiation = assemble_operator(K_x, x, PeriodicBC())
    
    numerical_f = zeros(x.n)
    mul!(weak_rhs,stiffness_matrix_differentiation,exact_phi)
    ldiv!(numerical_f,lu_mass_matrix,weak_rhs)
    #println(f)
    #println(numerical_f)
    max_err_f = maximum(abs.(f .- numerical_f))
    # println(max_err_f)
    @test max_err_f < 100*atol
    @test max_err_phi < atol
    return nothing
end

function runtests()
    @testset "Periodic differentiation 1D" begin
        println("Periodic Differentiation 1D Tests")
        periodic_differentiation_1D_test(; ngrid = 25, nelement = 1)
        periodic_differentiation_1D_test(; ngrid = 25, nelement = 2)
        periodic_differentiation_1D_test(; ngrid = 25, nelement = 3, atol = 1.0e-12)
        periodic_ODE_solve_1D_test(; ngrid = 20, nelement = 1)
        periodic_ODE_solve_1D_test(; ngrid = 20, nelement = 2)
        periodic_ODE_solve_1D_test(; ngrid = 20, nelement = 3)
    end
    return nothing
end

end # PeriodicBcTests

using .PeriodicBcTests

PeriodicBcTests.runtests()