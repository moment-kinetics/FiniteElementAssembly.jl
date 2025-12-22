"""
Module to test the implementation of electron-ion collisions
in a speed-pitch angle coordinate system. The ions are assumed
to have a Maxwellian distribution function characterised by density
and temperature. The analytical results used in this module can be obtained
from equations 3.40, 3.45-3.48 in the following reference [1]

[1] Helander, P., & Sigmar D. J., (2002). Collisional Transport in Magnetized Plasmas, Cambridge University Press,  Chapter 3.4, pages 40-42

Note that the equations implemented are

d F_e / dt = C_{ei}[F_e,F_{i,Maxwellian}]

with pitch angle scattering, energy drag and diffusion terms.
Appropriate modification of the ion and electron input parameters
renders the system the general cross-species collision operator between
a species A (e) and the species B (i) of Maxwellian distribution function with no flow.
"""
module ElectronIonCollisionsTest

using FiniteElementAssembly: FiniteElementCoordinate,
        ScalarCoordinateInputs,
        exclude_boundary_points, exclude_lower_boundary_point, include_boundary_points,
        assemble_operator, NaturalBC,
        integral
using FiniteElementMatrices
using Test: @test, @testset
using LinearAlgebra: lu, mul!, ldiv!
using SpecialFunctions: erf

function Chandrasekhar_G(x)
    return (erf(x) - (2.0/sqrt(pi))*x*exp(-x^2))/(2.0*x^2)
end
function nu_D_v2(v,c_i,c_e,nu_ei)
    vi = v/c_i
    phi = erf(vi)
    G = Chandrasekhar_G(vi)
    if vi > 1.0e-5
        result = (phi - G)*c_e^3/v
    else
       result = (4.0/(3.0*sqrt(pi)))*c_e^3/c_i
    end
    return result*nu_ei
end

function nu_s_v3(v, m_e, m_i, T_e, T_i, nu_ei)
    c_e = sqrt(2.0*T_e/m_e)
    c_i = sqrt(2.0*T_i/m_i)
    vi = v/c_i
    if vi > 1.0e-5
        result = Chandrasekhar_G(vi)*v^2*c_e
    else
       result = (2.0/sqrt(pi))*(vi/3.0)*v^2*c_e
    end
    result *= 2.0*nu_ei*(T_e/T_i)*(1.0 + m_i/m_e)
    return result
end

function nu_par_v4(v,c_i,c_e,nu_ei)
    vi = v/c_i
    if vi > 1.0e-5
        result = Chandrasekhar_G(vi)*v*c_e^3
    else
       result = (2.0/sqrt(pi))*(vi/3.0)*v*c_e^3
    end
    result *= 2.0*nu_ei
    return result
end

function calculate_collision_frequencies(vv_grid,m_e,m_i,T_e,T_i,nu_ei)
    ce = sqrt(2.0*T_e/m_e)
    ci = sqrt(2.0*T_i/m_i)
    nu_D_v2_data = deepcopy(vv_grid)
    nu_s_v3_data = deepcopy(vv_grid)
    nu_par_v4_data = deepcopy(vv_grid)
    for iv in 1:size(vv_grid,1)
        nu_D_v2_data[iv] = nu_D_v2(vv_grid[iv],ci,ce,nu_ei)
        nu_s_v3_data[iv] = nu_s_v3(vv_grid[iv],m_e,m_i,T_e,T_i,nu_ei)
        nu_par_v4_data[iv] = nu_par_v4(vv_grid[iv],ci,ce,nu_ei)
    end
    return nu_D_v2_data, nu_s_v3_data, nu_par_v4_data
end

struct CeiMatrices
    # Mass matrix in speed v
    M_vv::Array{Float64,3}
    # Mass matrix with nu_D*v^2 kernel
    N_vv::Array{Float64,3}
    # Stiffness matrix in v
    K_vv::Array{Float64,3}
    # Mass matrix in pitch angle xi
    M_xi::Array{Float64,3}
    # Stiffness matrix in xi
    K_xi::Array{Float64,3}
    function CeiMatrices(vv::FiniteElementCoordinate,xi::FiniteElementCoordinate,
                m_e, m_i, T_e, T_i, nu_ei, v_diffusion_and_drag::Bool,
                xi_scattering::Bool)
        c_e = sqrt(2*T_e/m_e)
        c_i = sqrt(2*T_i/m_i)
        xifactor = Float64(xi_scattering)
        vfactor = Float64(v_diffusion_and_drag)
        M_vv = Array{Float64,3}(undef,vv.ngrid,vv.ngrid,vv.nelement)
        N_vv = Array{Float64,3}(undef,vv.ngrid,vv.ngrid,vv.nelement)
        K_vv = Array{Float64,3}(undef,vv.ngrid,vv.ngrid,vv.nelement)
        for ielement_vv in 1:vv.nelement
            vv_data = vv.element_data[ielement_vv]
            @views M_vv[:,:,ielement_vv] = finite_element_matrix(lagrange_x,lagrange_x,2,vv_data)
            @views N_vv[:,:,ielement_vv] = finite_element_matrix(lagrange_x,lagrange_x,vv_data,
                                                kernel_function=(v -> nu_D_v2(v,c_i,c_e,nu_ei)),
                                                additional_quadrature_points=10+Int64(floor(1/ielement_vv)*30))
            @views K_vv[:,:,ielement_vv] = -vfactor*(# energy diffusion
                                            finite_element_matrix(d_lagrange_dx,d_lagrange_dx,vv_data,
                                                kernel_function=(v -> 0.5*nu_par_v4(v,c_i,c_e,nu_ei)),
                                                additional_quadrature_points=10+Int64(floor(1/ielement_vv)*30)) +
                                            # slowing down
                                            finite_element_matrix(lagrange_x,d_lagrange_dx,vv_data,
                                                kernel_function=(v -> (m_e/(m_i+m_e))*nu_s_v3(v, m_e, m_i, T_e, T_i, nu_ei)),
                                                additional_quadrature_points=10+Int64(floor(1/ielement_vv)*30)))
        end

        M_xi = Array{Float64,3}(undef,xi.ngrid,xi.ngrid,xi.nelement)
        K_xi = Array{Float64,3}(undef,xi.ngrid,xi.ngrid,xi.nelement)
        for ielement_xi in 1:xi.nelement
            xi_data = xi.element_data[ielement_xi]
            @views M_xi[:,:,ielement_xi] = finite_element_matrix(lagrange_x,lagrange_x,0,xi_data)
            @views K_xi[:,:,ielement_xi] = 0.5*xifactor*(finite_element_matrix(d_lagrange_dx,d_lagrange_dx,2,xi_data) -
                                            finite_element_matrix(d_lagrange_dx,d_lagrange_dx,0,xi_data))
        end
        return new(M_vv,N_vv,K_vv,M_xi,K_xi)
    end
end

struct Cei_diagnostics
    # grids
    xi_grid::Vector{Float64}
    vv_grid::Vector{Float64}
    # evaluation tests
    Cei_result::Array{Float64,2}
    Cei_exact::Array{Float64,2}
    Cei_err::Array{Float64,2}
    max_err_Cei::Float64
    delta_density::Float64
    delta_energy::Float64
    # time advance tests
    Finitial::Array{Float64,2}
    Fnew::Array{Float64,2}
    Fexact::Array{Float64,2}
    Fisotropic::Array{Float64,1}
    max_err_Fisotropic::Float64
    Ferr::Array{Float64,2}
    max_err_Fexact::Float64
    density_initial::Float64
    density_final::Float64
    density_expected::Float64
    density_t::Vector{Float64}
    energy_initial::Float64
    energy_final::Float64
    energy_expected::Float64
    energy_t::Vector{Float64}
    time::Vector{Float64}
    # diagnostics of collision frequencies
    nu_D_v2_data::Array{Float64,1}
    nu_s_v3_data::Array{Float64,1}
    nu_par_v4_data::Array{Float64,1}
end

function moment_diagnostics(Fnew,vv,xi,it,m_e,nwrite,ci_test)
    density = integral(((v,xi) -> 1.0),Fnew,vv,xi)
    energy = integral(((v,xi) -> 0.5*m_e*v^2),Fnew,vv,xi)
    if !ci_test && (it == 1 || mod(it,nwrite) == 0)
        println("it = $it")
        println("n = $(density) $(density - 1.0)")
        println("E = $(energy)")
    end
    return density, energy
end

function fix_density_moment(Fnew,vv,xi,density_initial)
    density = integral(((v,xi) -> 1.0),Fnew,vv,xi)
    @. Fnew *= (density_initial/density)
    return nothing
end

function runtests(; ngrid_xi=5::Int64, nelement_global_xi = 4::Int64,
    ngrid_vv=5::Int64, nelement_global_vv = 4::Int64, Lvv = 3.0,
    # cref_e=1.0, cref_i=1.0/60.0,
    delta_t=1.0, ntime=1000::Int64, ci_test=true::Bool,
    second_order_dt=true::Bool, nwrite=500::Int64, fix_density=false::Bool,
    # physics parameters
    m_e = 1.0, m_i=1836.0, T_e=1.0, T_i=1.0,
    nu_ref=1.0, Z_e=-1.0, Z_i=1.0, n_i=1.0,
    xi_scattering=true::Bool,
    v_diffusion_and_drag=true::Bool,
    # initial condition parameters
    vth=1.0,
    # tolerance parameters
    atol_moments = 1.0e-2, rtol_exact = 1.0e-5, rtol_isotropic = 1.0e-12,
    atol_delta_energy = 1.0e-8, atol_delta_density = 1.0e-12)
    # physical parameters
    c_e = sqrt(2*T_e/m_e)
    c_s = sqrt(2*T_i/m_e)
    nu_ei = nu_ref*(2.0*n_i*Z_i^2*Z_e^2/(m_e^2*c_e^3))
    # create the coordinates
    xi_min = -1.0
    xi_max = 1.0
    xi = FiniteElementCoordinate("xi", ScalarCoordinateInputs(ngrid_xi,
                                nelement_global_xi,
                                xi_min,xi_max,include_boundary_points))
    vv = FiniteElementCoordinate("vv", ScalarCoordinateInputs(ngrid_vv,
                                nelement_global_vv,
                                0.0,Lvv,exclude_lower_boundary_point),
                                weight_function=((v)-> 2.0*pi*v^2))
    cei_matrices = CeiMatrices(vv, xi, m_e, m_i, T_e, T_i, nu_ei,
                    v_diffusion_and_drag, xi_scattering)
    pdf = zeros(vv.n,xi.n)
    Cei_exact = zeros(vv.n,xi.n)
    Cei_result = zeros(vv.n,xi.n)
    Cei_err = zeros(vv.n,xi.n)
    rhsvxi = zeros(vv.n,xi.n)
    # the weak form for the Cei operator, on a single element
    function Cei_weak_form(jv, iv, ielement_v,
                        jxi, ixi, ielement_xi)
        return (cei_matrices.N_vv[jv, iv, ielement_v]*
                cei_matrices.K_xi[jxi, ixi, ielement_xi] +
                cei_matrices.K_vv[jv, iv, ielement_v]*
                cei_matrices.M_xi[jxi, ixi, ielement_xi])
    end
    function mass_matrix(jv, iv, ielement_v,
                        jxi, ixi, ielement_xi)
        return (cei_matrices.M_vv[jv, iv, ielement_v]*
                cei_matrices.M_xi[jxi, ixi, ielement_xi])
    end
    function time_advance_matrix_1st_order(jv, iv, ielement_v,
                        jxi, ixi, ielement_xi)
        return (mass_matrix(jv, iv, ielement_v,
                        jxi, ixi, ielement_xi) -
                delta_t*Cei_weak_form(jv, iv, ielement_v,
                                    jxi, ixi, ielement_xi))
    end
    function time_advance_matrix_2nd_order(jv, iv, ielement_v,
                        jxi, ixi, ielement_xi)
        return (1.5*mass_matrix(jv, iv, ielement_v,
                        jxi, ixi, ielement_xi) -
                delta_t*Cei_weak_form(jv, iv, ielement_v,
                                    jxi, ixi, ielement_xi))
    end
    Cei_stiffness_matrix_2D = assemble_operator(Cei_weak_form, vv, xi, NaturalBC(), NaturalBC())
    mass_matrix_2D = assemble_operator(mass_matrix, vv, xi, NaturalBC(), NaturalBC())
    time_advance_matrix_1st_order_2D = assemble_operator(time_advance_matrix_1st_order, vv, xi, NaturalBC(), NaturalBC())
    time_advance_matrix_2nd_order_2D = assemble_operator(time_advance_matrix_2nd_order, vv, xi, NaturalBC(), NaturalBC())

    luM = lu(mass_matrix_2D)
    lu_time_advance_1st_order = lu(time_advance_matrix_1st_order_2D)
    lu_time_advance_2nd_order = lu(time_advance_matrix_2nd_order_2D)

    # test the operator
    for ixi in 1:xi.n
        for iv in 1:vv.n
            speed = vv.grid[iv]
            pdf[iv,ixi] = exp(-(speed/c_s)^2)/(pi^1.5*c_s^3)
            Cei_exact[iv,ixi] = 0.0
        end
    end
    pdfc = vec(pdf)
    rhsc = vec(rhsvxi)
    Ceic = vec(Cei_result)
    # form the rhs of the weak form
    mul!(rhsc,Cei_stiffness_matrix_2D,pdfc)
    # solve the linear system
    ldiv!(Ceic, luM, rhsc)
    # test the results
    @. Cei_err = abs(Cei_result - Cei_exact)
    max_err_Cei = maximum(Cei_err)
    delta_density = integral(((v,xi) -> 1.0),Cei_result,vv,xi)
    delta_energy = integral(((v,xi) -> 0.5*m_e*v^2),Cei_result,vv,xi)
    if !ci_test
        println("Test evaluation")
        println("Evaluation error: $max_err_Cei")
        println("delta_density: $delta_density")
        println("delta_energy: $delta_energy")
    end
    Finitial = zeros(vv.n,xi.n)
    Fnew = zeros(vv.n,xi.n)
    Fnm1 = zeros(vv.n,xi.n)
    Fnm2 = zeros(vv.n,xi.n)
    Fexact = zeros(vv.n,xi.n)
    Ferr = zeros(vv.n,xi.n)
    Fisotropic = zeros(vv.n)
    xifactor = 0.3*Float64(xi_scattering)
    for ixi in 1:xi.n
        for iv in 1:vv.n
            speed = vv.grid[iv]
            Finitial[iv,ixi] = (1.0 + xi.grid[ixi]*xifactor)*exp(-(speed/vth)^2)/(pi^1.5*vth^3)
            Fexact[iv,ixi] = exp(-(speed/c_s)^2)/(pi^1.5*c_s^3)
        end
    end
    density_initial = integral(((v,xi) -> 1.0),Finitial,vv,xi)
    # normalise Finitial to 1
    @. Finitial /= density_initial
    density_initial = integral(((v,xi) -> 1.0),Finitial,vv,xi)
    # make sure that Fexact has the exact same density as Finitial
    normfac = integral(((v,xi) -> 1.0),Fexact,vv,xi)
    @. Fexact *= density_initial/normfac

    function time_advance_1st_order(Fnew)
        Fc = vec(Fnew)
        rhsc = vec(rhsvxi)
        # form the RHS of the weak form
        mul!(rhsc,mass_matrix_2D,Fc)
        # solve the linear system
        ldiv!(Fc,lu_time_advance_1st_order,rhsc)
        return nothing
    end
    function time_advance_2nd_order(Fnew,Fnm1,Fnm2)
        Fc = vec(Fnew)
        Fcm1 = vec(Fnm1)
        Fcm2 = vec(Fnm2)
        # get F for the RHS
        # we can overwrite Fcm2 as we do not need it again
        @. Fcm2 = 2.0*Fcm1 - 0.5*Fcm2
        rhsc = vec(rhsvxi)
        # form the RHS of the weak form
        mul!(rhsc,mass_matrix_2D,Fcm2)
        # solve the linear system for the new Fc
        ldiv!(Fc,lu_time_advance_2nd_order,rhsc)
        # update the past-time variables
        @. Fcm2 = Fcm1
        @. Fcm1 = Fc
        return nothing
    end
    density_t = zeros(ntime+1)
    energy_t = zeros(ntime+1)
    time = zeros(ntime+1)
    # set the initial condition
    @. Fnew = Finitial
    ntime_first_order = (ntime > 0 ? (second_order_dt ? 1 : ntime) : 0)
    density_t[1], energy_t[1] = moment_diagnostics(Fnew,vv,xi,0,m_e,nwrite,ci_test)
    for it in 1:ntime_first_order
        time_advance_1st_order(Fnew)
        if fix_density
            fix_density_moment(Fnew,vv,xi,density_initial)
        end
        density_t[it+1], energy_t[it+1] = moment_diagnostics(Fnew,vv,xi,it,m_e,nwrite,ci_test)
    end
    # set the initial condition
    @. Fnm2 = Finitial
    @. Fnm1 = Fnew
    for it in 1+ntime_first_order:ntime
        time_advance_2nd_order(Fnew,Fnm1,Fnm2)
        if fix_density
            fix_density_moment(Fnew,vv,xi,density_initial)
        end
        density_t[it+1], energy_t[it+1] = moment_diagnostics(Fnew,vv,xi,it,m_e,nwrite,ci_test)
    end
    # get a time variable
    for it in 0:ntime
        time[it+1] = delta_t*it
    end
    # test the result
    @. Ferr = abs(Fnew - Fexact)
    max_err_Fexact = maximum(Ferr)
    for iv in 1:vv.n
        # integrate over xi, divide by length of xi domain (Lxi = 2)
        @views Fisotropic[iv] = 0.5*integral((xi->1.0), Fnew[iv,:], xi)
        # use the isotropic F(v) to determine the deviation from isotropy
        @. Ferr[iv,:] = abs(Fnew[iv,:] - Fisotropic[iv])
    end
    max_err_Fisotropic = maximum(Ferr)
    # test moments
    density_initial = integral(((v,xi) -> 1.0),Finitial,vv,xi)
    density_final = integral(((v,xi) -> 1.0),Fnew,vv,xi)
    energy_initial = integral(((v,xi) -> 0.5*m_e*v^2),Finitial,vv,xi)
    energy_final = integral(((v,xi) -> 0.5*m_e*v^2),Fnew,vv,xi)
    energy_expected = 0.75*m_e*c_s^2
    density_expected = density_initial
    if !ci_test
        println("Test time advance, delta t = $delta_t ntime = $ntime")
        println("deviation from exact result: $max_err_Fexact")
        println("deviation from isotropy: $max_err_Fisotropic")
        println("density(t=0): $density_initial (t=end): $density_final delta: $(density_final - density_initial)")
        println("energy(t=0): $energy_initial (t=end): $energy_final delta: $(energy_final - energy_initial)")
        println("deviation from expected density: $(density_final - density_expected)")
        println("deviation from expected energy: $(energy_final - energy_expected)")
    end
    if ci_test
        @testset "Cei(speed,pitch) Tests: Time Evolution m_e=$m_e m_i=$m_i" begin
            println("Cei(speed,pitch) Tests: Time Evolution m_e=$m_e m_i=$m_i")
            @test max_err_Fexact < rtol_exact*maximum(Fexact)
            @test max_err_Fisotropic < rtol_isotropic*maximum(Fexact)
            @test abs(delta_density) < atol_delta_density
            @test abs(delta_energy) < atol_delta_energy
            @test abs(density_initial - 1.0) < atol_moments
            @test abs(energy_initial - 0.75*m_e*vth^2) < atol_moments
            @test abs(density_final - density_expected) < atol_moments
            @test abs(energy_final - energy_expected) < atol_moments
        end
    end
    nu_D_v2_data, nu_s_v3_data, nu_par_v4_data = calculate_collision_frequencies(vv.grid,m_e,m_i,T_e,T_i,nu_ei)
    return Cei_diagnostics(xi.grid, vv.grid,
                # evaluation test results
                Cei_result, Cei_exact, Cei_err, max_err_Cei, delta_density, delta_energy,
                # time advance test results
                Finitial, Fnew, Fexact, Fisotropic, max_err_Fisotropic, Ferr, max_err_Fexact,
                density_initial, density_final, density_expected, density_t,
                energy_initial, energy_final, energy_expected, energy_t, time,
                # collision frequencies
                nu_D_v2_data, nu_s_v3_data, nu_par_v4_data)
end

end # ElectronIonCollisionsTest

using .ElectronIonCollisionsTest

# Electron reference units
diagnostics = ElectronIonCollisionsTest.runtests(ngrid_vv=5, nelement_global_vv=16, Lvv=8.0,
                            ngrid_xi=5, nelement_global_xi=16, delta_t=1.0*60, ntime=10000,
                            m_e=1.0, m_i=1.0*60*60, T_e=1.0, T_i=1.1, ci_test=true, vth=sqrt(2))
# Ion reference units
diagnostics = ElectronIonCollisionsTest.runtests(ngrid_vv=5, nelement_global_vv=16, Lvv=8.0*60,
                            ngrid_xi=5, nelement_global_xi=16, delta_t=1.0, ntime=10000,
                            m_e=1.0/60/60, m_i=1.0, T_e=1.0, T_i=1.1, ci_test=true, vth=sqrt(2)*60,
                            atol_delta_density=1.0e-11, atol_delta_energy=1.0e-7)
