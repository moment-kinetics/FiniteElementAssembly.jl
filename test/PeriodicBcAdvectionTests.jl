"""
Module to test advection in 1D with periodic boundary conditions.
"""
module PeriodicBcAdvectionTests

using Test: @testset, @test
using FiniteElementAssembly
using FiniteElementMatrices
using LinearAlgebra

struct AdvectionData
    Mx::Array{Float64,3}
    Sx::Array{Float64,3}
end
function AdvectionData(x::FiniteElementCoordinate)
    Mx = Array{Float64,3}(undef,x.ngrid,x.ngrid,x.nelement)
    Sx = Array{Float64,3}(undef,x.ngrid,x.ngrid,x.nelement)
    for ielement in 1:x.nelement
        x_data = x.element_data[ielement]
        Mx[:,:,ielement] = finite_element_matrix(lagrange_x,lagrange_x,0,x_data)
        # integrate advection term v d F / dx by parts in weak form
        Sx[:,:,ielement] = -finite_element_matrix(lagrange_x,d_lagrange_dx,0,x_data)
        # uncomment to not integrate v d F / dx by parts
        #Sx[:,:,ielement] = finite_element_matrix(d_lagrange_dx,lagrange_x,0,x_data)
    end
    return AdvectionData(Mx,Sx)
end

function advection_1D_test(; ngrid = 20, nelement = 1, atol = 1.0e-13,
             delta_t::Float64=1.0, ntime::Int64=1, vx::Float64=1.0, first_order::Bool=false)
    # define inputs needed for the test
    coord_min = -1.0
    coord_max = 1.0
    # create the coordinate struct 'x'
    x = FiniteElementCoordinate("coord", ScalarCoordinateInputs(ngrid,
                            nelement, coord_min, coord_max, include_boundary_points),
                            bc = PeriodicBC())
    ad = AdvectionData(x)
    # mass matrix
    mass_matrix_1D = assemble_operator(ad.Mx, x, PeriodicBC())
    # 1st-order backward Euler time advance
    time_advance_1st_order_weak_form = Array{Float64,3}(undef,x.ngrid,x.ngrid,x.nelement)
    @. time_advance_1st_order_weak_form = ad.Mx + delta_t*vx*ad.Sx
    # 2nd-order backward Euler time advance
    time_advance_2nd_order_weak_form = Array{Float64,3}(undef,x.ngrid,x.ngrid,x.nelement)
    @. time_advance_2nd_order_weak_form = ad.Mx + (2.0/3.0)*delta_t*vx*ad.Sx
    #@. time_advance_2nd_order_weak_form = 1.5*ad.Mx + delta_t*vx*ad.Sx
    # assembled time advance matrix
    time_advance_1st_order_1D = assemble_operator(time_advance_1st_order_weak_form, x, PeriodicBC())
    lu_time_advance_1st_order = lu(time_advance_1st_order_1D)
    time_advance_2nd_order_1D = assemble_operator(time_advance_2nd_order_weak_form, x, PeriodicBC())
    lu_time_advance_2nd_order = lu(time_advance_2nd_order_1D)
    
    # a periodic function
    f0 = zeros(x.n) # f at time level n
    f1 = zeros(x.n) # f at time level n-1
    f2 = zeros(x.n) # f at time level n-2
    # the exact solution
    fexact = zeros(x.n)
    # dummy array for RHS
    rhs = zeros(x.n)
    # dummy array for error
    ferr = zeros(x.n)

    # set the initial condition
    # and the expected solution
    phase = 0.25
    for ix in 1:x.n
        xarg = 2.0*x.grid[ix]/x.L
        f0[ix] = cospi(xarg + phase)
        xparg = 2.0*(x.grid[ix] - vx*delta_t*ntime)/x.L
        fexact[ix] = cospi(xparg + phase)
    end

    function backward_Euler_1(f::Vector{Float64})
        mul!(rhs, mass_matrix_1D, f)
        ldiv!(f, lu_time_advance_1st_order, rhs)
        return nothing
    end
    function backward_Euler_2(f0::Vector{Float64},f1::Vector{Float64},f2::Vector{Float64})
        # rhs = Mx*( (4/3) f1 - (1/3) f2)
        # use f2 as dummy as we do not need it again
        @. f2 = (4.0/3.0)*f1 - (1.0/3.0)*f2
        mul!(rhs, mass_matrix_1D, f2)
        # find new f0
        ldiv!(f0, lu_time_advance_2nd_order, rhs)
        # update history
        copyto!(f2,f1)
        copyto!(f1,f0)
        return nothing
    end

    if first_order
        ntime_1st_order = ntime
        ntime_2nd_order = 0
    else
        ntime_1st_order = Int64(ntime > 0)
        ntime_2nd_order = ntime
    end
    # println(ntime_1st_order)
    # println(ntime_2nd_order)
    time = 0.0
    # backward Euler for step 1
    # store initial condition in history
    copyto!(f1,f0)
    for it in 1:ntime_1st_order
        time += delta_t
        backward_Euler_1(f0) 
    end
    # update history to prepare for second order advance
    copyto!(f2,f1)
    copyto!(f1,f0)
    # second order backward Euler for remaining steps
    for it in 2:ntime_2nd_order
        time += delta_t
        backward_Euler_2(f0,f1,f2) 
    end

    # test f
    # println(fexact)
    # println(f0)
    @. ferr = abs(f0 - fexact)
    max_ferr = maximum(ferr)
    # check total integral vanishes
    @test abs(integral((x)->1.0,f0,x)) < 1.0e-10
    # test max(abs.()) norm
    @test max_ferr < atol
    # test boundary condition
    @test abs(f0[1]-f0[end]) < 1.0e-14
    return nothing
end

function runtests()
    @testset "Periodic Advection 1D" begin
        println("Periodic Advection 1D Tests")
        advection_1D_test(; ngrid=15, nelement=1, ntime=10000, vx=1.0, delta_t=0.0001, first_order=false, atol=5.0e-7)
        advection_1D_test(; ngrid=15, nelement=1, ntime=10000, vx=1.0, delta_t=0.0001, first_order=true, atol=8e-3)
    end
    return nothing
end

end # PeriodicBcAdvectionTests

using .PeriodicBcAdvectionTests

PeriodicBcAdvectionTests.runtests()