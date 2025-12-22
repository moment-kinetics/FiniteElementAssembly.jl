"""
Module to test the implementation of the slowing down operator for a
trace distribution of alpha particles colliding with a background
of ions and electrons. The collision operator implemented here is taken
from chapter 3.4 in the following reference [1].

[1] Helander, P., & Sigmar D. J., (2002). Collisional Transport in Magnetized Plasmas, Cambridge University Press,  Chapter 3.4, pages 40-42

Note that we supply an extended source when solving

d F_{alpha}/ d t = C_{alpha}[F_{alpha}] + S_{alpha}

and we test against the corresponding analytical solution. The solution
for an extended source is constructed from the familiar delta-function
source solution by the method of Green's functions.
"""
module SlowingDownTest

using FiniteElementAssembly: FiniteElementCoordinate,
        ScalarCoordinateInputs, exclude_lower_boundary_point,
        assemble_operator, NaturalBC, integral
using FiniteElementMatrices
using LagrangePolynomials
using Test: @test, @testset
using LinearAlgebra: lu, mul!, ldiv!
using SpecialFunctions: erf

function nu_alphai_func(n_i::Float64,m_i::Float64,
            Z_i::Float64,m_alpha::Float64,Z_alpha::Float64)
    nu_alphai = 2.0*n_i*(Z_i*Z_alpha)^2/(m_i*m_alpha)
    return nu_alphai
end
function nu_alphae_func(T_e::Float64,n_e::Float64,
            m_e::Float64,m_alpha::Float64,Z_alpha::Float64)
    nu_alphae = (4.0/3.0)*(1.0/sqrt(2*pi))*(n_e/T_e^1.5)*(Z_alpha)^2*(m_e^0.5/m_alpha)
    return nu_alphae
end
function source_shape(v::Float64,v_b::Float64,v_s::Float64)
    return exp(-(v^2 - v_b^2)^2/(4.0*(v_s*v_b)^2))
end
"""
Solution shape function due to a source of the form `exp( - (v^2 - v_b^2))^2/(4*v_th^2 v_b^2 ) )`.
"""
function shape_function_from_source(v,v0,vth)
    y = (v - v0)/vth
    a = vth/v0
    # shape = \int^\infty_y (1 + a x)^2 exp(-x^2) exp(-(a x^3 + 0.25 a^2 x^4))
    # we expand exp(-(a x^3 + 0.25 a^2 x^4)) to O(a^6) and compute the integral analytically
    # shape including O(a^8) terms
    shape = (+a^0*(-sqrt(pi)*erf(y)/2 + sqrt(pi)/2)
        +a^1*(-y^2*exp(-y^2)/2 + exp(-y^2)/2)
        +a^2*(y^5*exp(-y^2)/4 - y^3*exp(-y^2)/2 - y*exp(-y^2)/4 + sqrt(pi)*erf(y)/8 - sqrt(pi)/8)
        +a^3*(-y^8*exp(-y^2)/12 + 7*y^6*exp(-y^2)/24 + y^4*exp(-y^2)/8 + y^2*exp(-y^2)/4 + exp(-y^2)/4)
        +a^4*(y^11*exp(-y^2)/48 - 11*y^9*exp(-y^2)/96 - y^5*exp(-y^2)/8 - 5*y^3*exp(-y^2)/16 - 15*y*exp(-y^2)/32 + 15*sqrt(pi)*erf(y)/64 - 15*sqrt(pi)/64)
        +a^5*(-y^14*exp(-y^2)/240 + y^12*exp(-y^2)/30 - 23*y^10*exp(-y^2)/960 + 7*y^8*exp(-y^2)/192 + 7*y^6*exp(-y^2)/48 + 7*y^4*exp(-y^2)/16 + 7*y^2*exp(-y^2)/8 + 7*exp(-y^2)/8)
        +a^6*(y^17*exp(-y^2)/1440 - 11*y^15*exp(-y^2)/1440 + 5*y^13*exp(-y^2)/384 - y^11*exp(-y^2)/96 - y^9*exp(-y^2)/24 - 3*y^7*exp(-y^2)/16 - 21*y^5*exp(-y^2)/32 - 105*y^3*exp(-y^2)/64 - 315*y*exp(-y^2)/128 + 315*sqrt(pi)*erf(y)/256 - 315*sqrt(pi)/256)
        +a^7*(y^18*exp(-y^2)/720 - y^16*exp(-y^2)/480 + 19*y^14*exp(-y^2)/960 + 77*y^12*exp(-y^2)/640 + 231*y^10*exp(-y^2)/320 + 231*y^8*exp(-y^2)/64 + 231*y^6*exp(-y^2)/16 + 693*y^4*exp(-y^2)/16 + 693*y^2*exp(-y^2)/8 + 693*exp(-y^2)/8)
        +a^8*(y^19*exp(-y^2)/1440 + y^17*exp(-y^2)/720 + 113*y^15*exp(-y^2)/5760 + 7*y^13*exp(-y^2)/48 + 91*y^11*exp(-y^2)/96 + 1001*y^9*exp(-y^2)/192 + 3003*y^7*exp(-y^2)/128 + 21021*y^5*exp(-y^2)/256 + 105105*y^3*exp(-y^2)/512 + 315315*y*exp(-y^2)/1024 - 315315*sqrt(pi)*erf(y)/2048 + 315315*sqrt(pi)/2048)
        )
    # multiply Jacobian factor to convert dimensionless result to a velocity integral
    shape *= 4*pi*(v0^2*vth)
    return shape
end
"""
Analytical solution of slowing down problem for a source that is
isotropic in pitch angle and extended in v.
See [1,2] for solutions with isotropic sources that are delta functions in v.
The solution below due to an extended source can be constructed with the
Green's function approach for linear ODEs.

[1] Helander, P., & Sigmar D. J., (2002). Collisional Transport in Magnetized Plasmas, Cambridge University Press,  Chapter 3.4, pages 40-42
[2] Moseev, D., & Salewski, M. (2019). Bi-Maxwellian, slowing-down, and ring velocity distributions of fast ions in
magnetized plasmas. Physics of Plasmas, 26(2). https://doi.org/10.1063/1.5085429
"""
function slowing_down_pdf(vv, source_rate, source_v0, source_vth,
            T_e, n_e, m_e, m_f, Z_f, n_i, m_i, Z_i, nu_ref)
    sd_pdf = Array{Float64}(undef,vv.n)
    tau_s = (3.0/4.0)*sqrt(2.0*pi)*(m_f/sqrt(m_e))*(T_e^(1.5))*(Z_f^(-2))*(1.0/n_e)
    Z1 = (n_i/n_e)*(m_f/m_i)*(Z_i^2)
    vc3 = 3.0*sqrt(pi/2.0)*Z1*(T_e^(1.5))/(m_f*sqrt(m_e))
    amplitude = source_rate*tau_s/(4.0*pi*nu_ref)
    v0 = source_v0
    vth = source_vth
    norm = shape_function_from_source(0.0,v0,vth)
    for iv in 1:vv.n
        shape = shape_function_from_source(vv.grid[iv],v0,vth)/norm
        v3 = vv.grid[iv]^3
        sd_pdf[iv] = amplitude*shape/(vc3 + v3)
    end
    return sd_pdf
end

struct CalphaMatrices
    # Mass matrix in speed v
    M_vv::Array{Float64,3}
    # Derivative matrix with nu_{alpha,ion} kernel
    Pi_vv::Array{Float64,3}
    # Derivative matrix with nu_{alpha,electron} kernel
    Pe_vv::Array{Float64,3}
    function CalphaMatrices(vv::FiniteElementCoordinate,
                T_e::Float64, n_e::Float64, m_e::Float64,
                n_i::Float64, m_i::Float64, Z_i::Float64,
                m_alpha::Float64, Z_alpha::Float64)
        # collision frequencies
        nu_alphae = nu_alphae_func(T_e, n_e, m_e, m_alpha, Z_alpha)
        nu_alphai = nu_alphai_func(n_i, m_i, Z_i, m_alpha, Z_alpha)
        # matrices
        M_vv = Array{Float64,3}(undef,vv.ngrid,vv.ngrid,vv.nelement)
        Pi_vv = Array{Float64,3}(undef,vv.ngrid,vv.ngrid,vv.nelement)
        Pe_vv = Array{Float64,3}(undef,vv.ngrid,vv.ngrid,vv.nelement)
        for ielement_vv in 1:vv.nelement
            vv_data = vv.element_data[ielement_vv]
            @views M_vv[:,:,ielement_vv] = finite_element_matrix(lagrange_x,lagrange_x,2,vv_data)
            @views Pi_vv[:,:,ielement_vv] = -nu_alphai*finite_element_matrix(lagrange_x,d_lagrange_dx,0,vv_data)
            @views Pe_vv[:,:,ielement_vv] = -nu_alphae*finite_element_matrix(lagrange_x,d_lagrange_dx,3,vv_data)
        end
        # lower limit contribution from integration by parts
        # this is the sink of alphas at v=0
        ielement_v = 1
        lpoly_data = vv.lpoly_data[ielement_v]
        for iv in 1:vv.ngrid
            for ivp in 1:vv.ngrid
                Pi_vv[ivp,iv,ielement_v] -= (nu_alphai*
                                            lagrange_poly(lpoly_data.lpoly_data[ivp],0.0)*
                                            lagrange_poly(lpoly_data.lpoly_data[iv],0.0))
            end
        end
        return new(M_vv,Pi_vv,Pe_vv)
    end
end

struct SlowingDownTestDiagnostics
    max_err::Float64
    L2_err::Float64
    pdf_numerical::Array{Float64,1}
    pdf_analytical::Array{Float64,1}
    vgrid::Array{Float64,1}
end

function runtests(; ngrid_v=9::Int64, nelement_v = 32::Int64, Lv = 1.5,
    delta_t=1.0, ntime=1000::Int64, ci_test=false::Bool,
    # physics parameters
    m_e = 1.0/1836.0, m_i=2.0, m_alpha=4.0,
    T_e=1.0e-2, n_i=1.0, nu_ref=1.0,
    Z_i=1.0, Z_alpha=2.0,
    # source parameters
    v_b = 1.0, # birth speed
    v_s = 0.05, # source width
    s_rate = 1.0 # source rate
    # initial condition parameters
    # tolerance parameters
    )
    # trace limit for alphas
    n_e = Z_i*n_i
    # speed coordinate vv
    vv = FiniteElementCoordinate("vv", ScalarCoordinateInputs(ngrid_v,nelement_v,
                                0.0,Lv,exclude_lower_boundary_point),
                                weight_function=((v)-> 4.0*pi*v^2))

    cam = CalphaMatrices(vv, T_e, n_e, m_e,
                n_i, m_i, Z_i, m_alpha, Z_alpha)
    # assembled mass matrix
    mass_matrix_1D = assemble_operator(cam.M_vv, vv, NaturalBC())
    time_advance_1st_order_weak_form = Array{Float64,3}(undef,vv.ngrid,vv.ngrid,vv.nelement)
    @. time_advance_1st_order_weak_form = cam.M_vv - delta_t*nu_ref*(cam.Pe_vv + cam.Pi_vv)
    # assembled time advance matrix
    time_advance_1st_order_1D = assemble_operator(time_advance_1st_order_weak_form, vv, NaturalBC())
    lu_time_advance_1st_order = lu(time_advance_1st_order_1D)
    # initialise the source function
    source = zeros(Float64, vv.n)
    for iv in 1:vv.n
        source[iv] = source_shape(vv.grid[iv],v_b,v_s)
    end
    norm = integral((v) -> 1.0, source, vv)
    @. source *= s_rate/norm
    # obtain the numerical the solution of
    # d F / dt = C_alpha + S_alpha
    # F^n+1 - dt*C_alpha[F^n+1] = F^n + dt*S_alpha
    # initial pdf = 0
    pdf = zeros(Float64, vv.n)
    rhsv = zeros(Float64, vv.n)
    for it in 1:ntime
        @. pdf += delta_t*source
        mul!(rhsv,mass_matrix_1D,pdf)
        ldiv!(pdf, lu_time_advance_1st_order,rhsv)
    end
    # test the solution against the analytical solution
    sd_pdf = slowing_down_pdf(vv, s_rate, v_b, v_s,
                T_e, n_e, m_e, m_alpha, Z_alpha,
                n_i, m_i, Z_i, nu_ref)
    pdf_err = zeros(Float64, vv.n)
    @. pdf_err  = sd_pdf .- pdf
    max_err = maximum(abs.(pdf_err))
    L2_err = sqrt(integral(v -> 1.0,pdf_err.^2,vv)/integral(v -> 1.0,ones(Float64,vv.n),vv))
    if ci_test
        @testset "Calpha(speed) Tests: Time Evolution" begin
            println("Calpha(speed) Tests: Time Evolution")
            # test absolute errors
            @test max_err < 5.0e-10
            @test L2_err < 2.0e-10
        end
    else
        println("v        pdf                 sd                  err")
        for iv in 1:vv.n
            println("$(round(vv.grid[iv],sigdigits=2)) $(pdf[iv]) $(sd_pdf[iv]) $(pdf_err[iv])")
        end
        println("max_err $(max_err)")
        println("L2_err $(L2_err)")
    end
    return SlowingDownTestDiagnostics(max_err,L2_err,pdf,sd_pdf,vv.grid)
end


end # SlowingDownTest

using .SlowingDownTest
diagnostics = SlowingDownTest.runtests(ci_test=true)