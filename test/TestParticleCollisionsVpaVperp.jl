"""
Module to test the implementation of electron-ion collisions
in a vpa-vperp angle coordinate system. The ions are assumed
to have a Maxwellian distribution function characterised by density, mean flow,
and temperature. The analytical results for the Rosenbluth potentials used in this module can be obtained
from the following references [1,2,3]

[1] Helander, P., & Sigmar D. J., (2002). Collisional Transport in Magnetized Plasmas, Cambridge University Press,  Chapter 3.4, pages 37-38
[2] Hazeltine, R.D., Meiss, J.D. (2003) Plasma Confinement, Dover, New York, Chpt. 5, Sec. 5.2, eqn. (5.49)
[3] Hardman, M.R., Abazorius, M., Omotani, J., Barnes, M., Newton, S.L., Cook, J.W.S., Farrell, P.E., Parra, F.I., A higher-order finite-element implementation of the nonlinear Fokker--Planck collision operator for charged particle collisions in a low density plasma, Computer Physics Communications, Volume 314, 2025, 109675, https://doi.org/10.1016/j.cpc.2025.109675., Appendix B

Note that the equations implemented are

d F_e / dt = C_{ei}[F_e,F_{i,Maxwellian}]

Appropriate modification of the ion and electron input parameters
renders the system the general cross-species collision operator between
a species A (e) and the species B (i) of Maxwellian distribution function.
"""
module TestParticleCollisionsVpaVperp

using FiniteElementAssembly: FiniteElementCoordinate,
        ScalarCoordinateInputs, exclude_lower_boundary_point, include_boundary_points,
        assemble_operator, NaturalBC,
        integral
using FiniteElementMatrices
using Test: @test, @testset
using LinearAlgebra: lu, mul!, ldiv!
using SpecialFunctions: erf


function eta_func(upar::Float64,vth::Float64,
             vpa::Float64,vperp::Float64)
    speed = sqrt( (vpa - upar)^2 + vperp^2)/vth
    return speed
end

# 1D derivative functions

function dGdeta(eta::Float64)
    # d \tilde{G} / d eta
    dGdeta_fac = (1.0/sqrt(pi))*exp(-eta^2)/eta + (1.0 - 0.5/(eta^2))*erf(eta)
    return dGdeta_fac
end

function d2Gdeta2(eta::Float64)
    # d \tilde{G} / d eta
    d2Gdeta2_fac = erf(eta)/(eta^3) - (2.0/sqrt(pi))*exp(-eta^2)/(eta^2)
    return d2Gdeta2_fac
end

function ddGddeta(eta::Float64)
    # d / d eta ( (1/ eta) d \tilde{G} d eta
    ddGddeta_fac = (1.5/(eta^2) - 1.0)*erf(eta)/(eta^2) - (3.0/sqrt(pi))*exp(-eta^2)/(eta^3)
    return ddGddeta_fac
end

function dHdeta(eta::Float64)
    dHdeta_fac = (2.0/sqrt(pi))*(exp(-eta^2))/eta - erf(eta)/(eta^2)
    return dHdeta_fac
end

function d2Gdvpa2_Maxwellian(dens::Float64,upar::Float64,vth::Float64,
                            vpa::Float64,vperp::Float64)
    eta = eta_func(upar,vth,vpa,vperp)
    fac = dGdeta(eta) + ddGddeta(eta)*((vpa - upar)^2)/(vth^2)
    d2Gdvpa2_fac = fac*dens/(eta*vth)
    return d2Gdvpa2_fac
end

function d2Gdvperpdvpa_Maxwellian(dens::Float64,upar::Float64,vth::Float64,
                            vpa::Float64,vperp::Float64)
    eta = eta_func(upar,vth,vpa,vperp)
    fac = ddGddeta(eta)*vperp*(vpa - upar)/(vth^2)
    d2Gdvperpdvpa_fac = fac*dens/(eta*vth)
    return d2Gdvperpdvpa_fac
end

function d2Gdvperp2_Maxwellian(dens::Float64,upar::Float64,vth::Float64,
                            vpa::Float64,vperp::Float64)
    eta = eta_func(upar,vth,vpa,vperp)
    fac = dGdeta(eta) + ddGddeta(eta)*(vperp^2)/(vth^2)
    d2Gdvperp2_fac = fac*dens/(eta*vth)
    return d2Gdvperp2_fac
end

function dGdvperp_Maxwellian(dens::Float64,upar::Float64,vth::Float64,
                            vpa::Float64,vperp::Float64)
    eta = eta_func(upar,vth,vpa,vperp)
    fac = dGdeta(eta)*vperp*dens/(vth*eta)
    return fac
end

function dHdvperp_Maxwellian(dens::Float64,upar::Float64,vth::Float64,
                            vpa::Float64,vperp::Float64)
    eta = eta_func(upar,vth,vpa,vperp)
    fac = dHdeta(eta)*vperp*dens/(eta*vth^3)
    return fac
end

function dHdvpa_Maxwellian(dens::Float64,upar::Float64,vth::Float64,
                            vpa::Float64,vperp::Float64)
    eta = eta_func(upar,vth,vpa,vperp)
    fac = dHdeta(eta)*(vpa-upar)*dens/(eta*vth^3)
    return fac
end

function F_Maxwellian(dens::Float64,upar::Float64,vth::Float64,
                        vpa::Float64,vperp::Float64)
    eta = eta_func(upar,vth,vpa,vperp)
    fac = (dens/(vth^3)/π^1.5)*exp(-eta^2)
    return fac
end

struct CeiMatrices
    # Mass matrix in vpa
    Mpar::Array{Float64,3}
    # Mass matrix in vperp
    Mperp::Array{Float64,3}
    # Stiffness matrix for vperp flux, d2Gdvperp2 term
    K_vperp_vperp::Array{Float64,6}
    # Stiffness matrix for vperp flux, d2Gdvperpdvpa term
    K_vperp_vpa::Array{Float64,6}
    # Stiffness matrix for vperp flux, dHdvperp term
    P_vperp::Array{Float64,6}
    # Stiffness matrix for vpa flux, d2Gdvpa2 term
    K_vpa_vpa::Array{Float64,6}
    # Stiffness matrix for vpa flux, d2Gdvperpdvpa term
    K_vpa_vperp::Array{Float64,6}
    # Stiffness matrix for vpa flux, dHdvpa term
    P_vpa::Array{Float64,6}
    function CeiMatrices(vpa::FiniteElementCoordinate,vperp::FiniteElementCoordinate,
                m_e, m_i, Z_e, Z_i, n_i, T_e, T_i, u_i, atol, rtol)
        c_e = sqrt(2*T_e/m_e)
        c_i = sqrt(2*T_i/m_i)
        Gfac = (Z_e*Z_i/m_e)^2
        Hfac = 2.0*((Z_e*Z_i)^2/(m_e*m_i))
        Mpar = Array{Float64,3}(undef,vpa.ngrid,vpa.ngrid,vpa.nelement)
        for ielement_vpa in 1:vpa.nelement
            vpa_data = vpa.element_data[ielement_vpa]
            @views Mpar[:,:,ielement_vpa] = finite_element_matrix(lagrange_x,lagrange_x,0,vpa_data)
        end
        Mperp = Array{Float64,3}(undef,vperp.ngrid,vperp.ngrid,vperp.nelement)
        for ielement_vperp in 1:vperp.nelement
            vperp_data = vperp.element_data[ielement_vperp]
            @views Mperp[:,:,ielement_vperp] = finite_element_matrix(lagrange_x,lagrange_x,1,vperp_data)
        end
        K_vperp_vperp = Array{Float64,6}(undef,vpa.ngrid,vperp.ngrid,
                            vpa.ngrid,vperp.ngrid,vpa.nelement,vperp.nelement)
        K_vperp_vpa = Array{Float64,6}(undef,vpa.ngrid,vperp.ngrid,
                            vpa.ngrid,vperp.ngrid,vpa.nelement,vperp.nelement)
        P_vperp = Array{Float64,6}(undef,vpa.ngrid,vperp.ngrid,
                            vpa.ngrid,vperp.ngrid,vpa.nelement,vperp.nelement)
        K_vpa_vpa = Array{Float64,6}(undef,vpa.ngrid,vperp.ngrid,
                            vpa.ngrid,vperp.ngrid,vpa.nelement,vperp.nelement)
        K_vpa_vperp = Array{Float64,6}(undef,vpa.ngrid,vperp.ngrid,
                            vpa.ngrid,vperp.ngrid,vpa.nelement,vperp.nelement)
        P_vpa = Array{Float64,6}(undef,vpa.ngrid,vperp.ngrid,
                            vpa.ngrid,vperp.ngrid,vpa.nelement,vperp.nelement)
        # ensure max_iterations > 0 to make sure atol, rtol respected
        max_iterations = 10
        # give a default set of additional quadrature points to make
        # integrals accurate at the 0th iteration
        additional_quadrature_points_vpa = 5
        additional_quadrature_points_vperp = 5
        for ielement_vperp in 1:vperp.nelement
            vperp_data = vperp.element_data[ielement_vperp]
            # only expect to need p-adaptivity near origin in vperp
            # so set a smaller quadrature increment for larger ielement_vperp
            quadrature_increment = 1 + 5*Int64(floor(2/ielement_vperp))
            #quadrature_increment = 10
            for ielement_vpa in 1:vpa.nelement
                vpa_data = vpa.element_data[ielement_vpa]
                # vperp flux, d f / d vperp contribution
                @views K_vperp_vperp[:,:,:,:,ielement_vpa,ielement_vperp] = -Gfac*finite_element_matrix(lagrange_x,lagrange_x,vpa_data,
                                                                                d_lagrange_dx,d_lagrange_dx,vperp_data;
                                                                                kernel_function=((vpa,vperp) -> vperp*d2Gdvperp2_Maxwellian(n_i,u_i,c_i,vpa,vperp)),
                                                                                max_iterations=max_iterations, quadrature_increment=quadrature_increment,
                                                                                additional_quadrature_points_x1 = additional_quadrature_points_vpa,
                                                                                additional_quadrature_points_x2 = additional_quadrature_points_vperp,
                                                                                atol=atol, rtol=rtol, verbose=false)
                # vperp flux, d f / d vpa contribution
                @views K_vperp_vpa[:,:,:,:,ielement_vpa,ielement_vperp] = -Gfac*finite_element_matrix(d_lagrange_dx,lagrange_x,vpa_data,
                                                                                lagrange_x,d_lagrange_dx,vperp_data;
                                                                                kernel_function=((vpa,vperp) -> vperp*d2Gdvperpdvpa_Maxwellian(n_i,u_i,c_i,vpa,vperp)),
                                                                                max_iterations=max_iterations, quadrature_increment=1quadrature_increment,
                                                                                additional_quadrature_points_x1 = additional_quadrature_points_vpa,
                                                                                additional_quadrature_points_x2 = additional_quadrature_points_vperp,
                                                                                atol=atol, rtol=rtol, verbose=false)
                # vperp flux, f contribution
                @views P_vperp[:,:,:,:,ielement_vpa,ielement_vperp] = Hfac*finite_element_matrix(lagrange_x,lagrange_x,vpa_data,
                                                                                lagrange_x,d_lagrange_dx,vperp_data;
                                                                                kernel_function=((vpa,vperp) -> vperp*dHdvperp_Maxwellian(n_i,u_i,c_i,vpa,vperp)),
                                                                                max_iterations=max_iterations, quadrature_increment=quadrature_increment,
                                                                                additional_quadrature_points_x1 = additional_quadrature_points_vpa,
                                                                                additional_quadrature_points_x2 = additional_quadrature_points_vperp,
                                                                                atol=atol, rtol=rtol, verbose=false)
                # vpa flux, d f / d vpa contribution
                @views K_vpa_vpa[:,:,:,:,ielement_vpa,ielement_vperp] = -Gfac*finite_element_matrix(d_lagrange_dx,d_lagrange_dx,vpa_data,
                                                                                lagrange_x,lagrange_x,vperp_data;
                                                                                kernel_function=((vpa,vperp) -> vperp*d2Gdvpa2_Maxwellian(n_i,u_i,c_i,vpa,vperp)),
                                                                                max_iterations=max_iterations, quadrature_increment=quadrature_increment,
                                                                                additional_quadrature_points_x1 = additional_quadrature_points_vpa,
                                                                                additional_quadrature_points_x2 = additional_quadrature_points_vperp,
                                                                                atol=atol, rtol=rtol, verbose=false)
                # vpa flux, d f / d vperp contribution
                @views K_vpa_vperp[:,:,:,:,ielement_vpa,ielement_vperp] = -Gfac*finite_element_matrix(lagrange_x,d_lagrange_dx,vpa_data,
                                                                                d_lagrange_dx,lagrange_x,vperp_data;
                                                                                kernel_function=((vpa,vperp) -> vperp*d2Gdvperpdvpa_Maxwellian(n_i,u_i,c_i,vpa,vperp)),
                                                                                max_iterations=max_iterations, quadrature_increment=quadrature_increment,
                                                                                additional_quadrature_points_x1 = additional_quadrature_points_vpa,
                                                                                additional_quadrature_points_x2 = additional_quadrature_points_vperp,
                                                                                atol=atol, rtol=rtol, verbose=false)
                # vpa flux, f contribution
                @views P_vpa[:,:,:,:,ielement_vpa,ielement_vperp] = Hfac*finite_element_matrix(lagrange_x,d_lagrange_dx,vpa_data,
                                                                                lagrange_x,lagrange_x,vperp_data;
                                                                                kernel_function=((vpa,vperp) -> vperp*dHdvpa_Maxwellian(n_i,u_i,c_i,vpa,vperp)),
                                                                                max_iterations=max_iterations, quadrature_increment=quadrature_increment,
                                                                                additional_quadrature_points_x1 = additional_quadrature_points_vpa,
                                                                                additional_quadrature_points_x2 = additional_quadrature_points_vperp,
                                                                                atol=atol, rtol=rtol, verbose=false)
            end
        end
        return new(Mpar, Mperp,
            K_vperp_vperp, K_vperp_vpa, P_vperp,
            K_vpa_vpa, K_vpa_vperp, P_vpa)
    end
end

function moment_diagnostics(Fnew,vpa,vperp,it,m_e,nwrite,ci_test)
    density = integral(((vpa,vperp) -> 1.0),Fnew,vpa,vperp)
    nupar = integral(((vpa,vperp) -> vpa),Fnew,vpa,vperp)
    energy = integral(((vpa,vperp) -> 0.5*m_e*(vpa^2+vperp^2)),Fnew,vpa,vperp)
    if !ci_test && (it == 1 || mod(it,nwrite) == 0)
        println("it = $it")
        println("n = $(density)")
        println("nu = $(nupar)")
        println("E = $(energy)")
    end
    return density, nupar, energy
end


struct Cei_vpa_vperp_diagnostics
    # grids
    vpa_grid::Vector{Float64}
    vperp_grid::Vector{Float64}
    # evaluation tests
    Cei_result::Array{Float64,2}
    Cei_exact::Array{Float64,2}
    Cei_err::Array{Float64,2}
    max_err_Cei::Float64
    delta_density::Float64
    delta_nupar::Float64
    delta_energy::Float64
    # time advance tests
    Finitial::Array{Float64,2}
    Fnew::Array{Float64,2}
    Fexact::Array{Float64,2}
    Ferr::Array{Float64,2}
    max_err_Fexact::Float64
    density_initial::Float64
    density_final::Float64
    density_expected::Float64
    density_t::Vector{Float64}
    nupar_initial::Float64
    nupar_final::Float64
    nupar_expected::Float64
    nupar_t::Vector{Float64}
    energy_initial::Float64
    energy_final::Float64
    energy_expected::Float64
    energy_t::Vector{Float64}
    time::Vector{Float64}
end



function runtests(;
    # grid info
    ngrid_vpa=5::Int64, nelement_vpa=16::Int64,
    ngrid_vperp=5::Int64, nelement_vperp=8::Int64,
    Lvpa=10.0::Float64, Lvperp=5.0::Float64,
    # physics parameters
    m_e = 1.0, m_i=1836.0, T_e=1.0, n_e=1.0, u_e=0.0,
    nu_ref=1.0, Z_e=-1.0, Z_i=1.0, n_i=1.0, u_i=0.0, T_i=1.0,
    # timestepping
    delta_t=1.0, ntime=1000::Int64, ci_test=false::Bool, nwrite=1::Int64,
    # test tolerances
    atol_delta_density=1.0e-13, atol_delta_nupar=1.0e-13, atol_delta_energy=1.0e-13,
    atol_density=1.0e-13, atol_nupar=1.0e-13, atol_energy=1.0e-13, rtol_exact=1.0e-13,
    # matrix tolerances
    matrix_atol=1.0e-13, matrix_rtol=1.0e-13,
    )
    vpa_min = -0.5*Lvpa
    vpa_max = 0.5*Lvpa
    # create the coordinates
    if !ci_test println("create the coordinates") end
    vpa = FiniteElementCoordinate("vpa", ScalarCoordinateInputs(ngrid_vpa,
                                nelement_vpa,vpa_min,vpa_max,include_boundary_points))
    vperp = FiniteElementCoordinate("vperp", ScalarCoordinateInputs(ngrid_vperp,
                                nelement_vperp,0.0,Lvperp,exclude_lower_boundary_point),
                                weight_function=((vperp)-> 2.0*pi*vperp))
    c_e = sqrt(2*T_e/m_e)
    c_i = sqrt(2*T_i/m_i)
    c_s = sqrt(2*T_i/m_e)
    nu_ei = nu_ref*(2.0*n_i*Z_i^2*Z_e^2/(m_e^2*c_e^3))
    if !ci_test println("create the local matrices") end
    cei_matrices = CeiMatrices(vpa, vperp, m_e, m_i, Z_e, Z_i, n_i, T_e, T_i, u_i, matrix_atol, matrix_rtol)
    function Cei_weak_form(jvpa, ivpa, ielement_vpa,
                        jvperp, ivperp, ielement_vperp)
        return (cei_matrices.K_vperp_vperp[jvpa, jvperp, ivpa, ivperp, ielement_vpa, ielement_vperp] +
                cei_matrices.K_vperp_vpa[jvpa, jvperp, ivpa, ivperp, ielement_vpa, ielement_vperp] +
                cei_matrices.P_vperp[jvpa, jvperp, ivpa, ivperp, ielement_vpa, ielement_vperp] +
                cei_matrices.K_vpa_vpa[jvpa, jvperp, ivpa, ivperp, ielement_vpa, ielement_vperp] +
                cei_matrices.K_vpa_vperp[jvpa, jvperp, ivpa, ivperp, ielement_vpa, ielement_vperp] +
                cei_matrices.P_vpa[jvpa, jvperp, ivpa, ivperp, ielement_vpa, ielement_vperp])
    end
    function mass_matrix(jvpa, ivpa, ielement_vpa,
                        jvperp, ivperp, ielement_vperp)
        return (cei_matrices.Mpar[jvpa, ivpa, ielement_vpa]*
                cei_matrices.Mperp[jvperp, ivperp, ielement_vperp])
    end
    function time_advance_matrix_1st_order(jvpa, ivpa, ielement_vpa,
                        jvperp, ivperp, ielement_vperp)
        return (mass_matrix(jvpa, ivpa, ielement_vpa,
                        jvperp, ivperp, ielement_vperp) -
                delta_t*Cei_weak_form(jvpa, ivpa, ielement_vpa,
                        jvperp, ivperp, ielement_vperp))
    end
    # assemble the sparse matrices
    if !ci_test println("assemble the sparse global matrices") end
    Cei_stiffness_matrix_2D = assemble_operator(Cei_weak_form, vpa, vperp, NaturalBC(), NaturalBC())
    mass_matrix_2D = assemble_operator(mass_matrix, vpa, vperp, NaturalBC(), NaturalBC())
    time_advance_matrix_1st_order_2D = assemble_operator(time_advance_matrix_1st_order,
                                        vpa, vperp, NaturalBC(), NaturalBC())
    # LU decomposition
    if !ci_test println("calculate the LU decomposition") end
    luM = lu(mass_matrix_2D)
    lu_time_advance_1st_order = lu(time_advance_matrix_1st_order_2D)
    # test the forward calculation of Cei[Fe]
    if !ci_test println("test the forward calculation of Cei[Fe]") end
    pdf = zeros(vpa.n,vperp.n)
    Cei_exact = zeros(vpa.n,vperp.n)
    Cei_result = zeros(vpa.n,vperp.n)
    Cei_err = zeros(vpa.n,vperp.n)
    rhsvpavperp = zeros(vpa.n,vperp.n)
    for ivperp in 1:vperp.n
        for ivpa in 1:vpa.n
            # N.B. the pdf=Fe is the final equilibrium Fe with T_e = T_i, u_e = u_i
            pdf[ivpa,ivperp] = F_Maxwellian(n_e,u_i,c_s,vpa.grid[ivpa],vperp.grid[ivperp])
            Cei_exact[ivpa,ivperp] = 0.0
        end
    end
    pdfc = vec(pdf)
    rhsc = vec(rhsvpavperp)
    Ceic = vec(Cei_result)
    # form the rhs of the weak form
    mul!(rhsc,Cei_stiffness_matrix_2D,pdfc)
    # solve the linear system
    ldiv!(Ceic, luM, rhsc)
    # test the results
    @. Cei_err = abs(Cei_result - Cei_exact)
    max_err_Cei = maximum(Cei_err)
    delta_density = integral(((vpa,vperp) -> 1.0),Cei_result,vpa,vperp)
    delta_energy = integral(((vpa,vperp) -> 0.5*m_e*(vpa^2 + vperp^2)),Cei_result,vpa,vperp)
    delta_nupar = integral(((vpa,vperp) -> vpa),Cei_result,vpa,vperp)
    if !ci_test
        println("Test evaluation")
        println("Evaluation error: $max_err_Cei")
        println("delta_density: $delta_density")
        println("delta_nupar: $delta_nupar")
        println("delta_energy: $delta_energy")
    end

    # test time evolution
    Finitial = zeros(vpa.n,vperp.n)
    Fnew = zeros(vpa.n,vperp.n)
    Fexact = zeros(vpa.n,vperp.n)
    Ferr = zeros(vpa.n,vperp.n)
    for ivperp in 1:vperp.n
        for ivpa in 1:vpa.n
            Finitial[ivpa,ivperp] = F_Maxwellian(n_e, u_e, c_e, vpa.grid[ivpa], vperp.grid[ivperp])
            Fexact[ivpa,ivperp] = F_Maxwellian(n_e, u_i, c_s, vpa.grid[ivpa], vperp.grid[ivperp])
        end
    end
    density_initial = integral(((vpa,vperp) -> 1.0),Finitial,vpa,vperp)
    # normalise Finitial to n_e
    @. Finitial *= n_e/density_initial
    density_initial = integral(((vpa,vperp) -> 1.0),Finitial,vpa,vperp)
    # make sure that Fexact has the exact same density as Finitial
    normfac = integral(((vpa,vperp) -> 1.0),Fexact,vpa,vperp)
    @. Fexact *= density_initial/normfac

    function time_advance_1st_order(Fnew)
        Fc = vec(Fnew)
        rhsc = vec(rhsvpavperp)
        # form the RHS of the weak form
        mul!(rhsc,mass_matrix_2D,Fc)
        # solve the linear system
        ldiv!(Fc,lu_time_advance_1st_order,rhsc)
        return nothing
    end
    density_t = zeros(ntime+1)
    nupar_t = zeros(ntime+1)
    energy_t = zeros(ntime+1)
    time = zeros(ntime+1)
    # set the initial condition
    @. Fnew = Finitial
    density_t[1], nupar_t[1], energy_t[1] = moment_diagnostics(Fnew,vpa,vperp,0,m_e,nwrite,ci_test)
    time[1] = 0.0
    for it in 1:ntime
        time[it+1] = delta_t*it
        time_advance_1st_order(Fnew)
        density_t[it+1], nupar_t[it+1], energy_t[it+1] = moment_diagnostics(Fnew,vpa,vperp,it,m_e,nwrite,ci_test)
    end
    # test the result
    @. Ferr = abs(Fnew - Fexact)
    max_err_Fexact = maximum(Ferr)
    # test moments
    density_initial = integral(((vpa,vperp) -> 1.0),Finitial,vpa,vperp)
    density_final = integral(((vpa,vperp) -> 1.0),Fnew,vpa,vperp)
    nupar_initial = integral(((vpa,vperp) -> vpa),Finitial,vpa,vperp)
    nupar_final = integral(((vpa,vperp) -> vpa),Fnew,vpa,vperp)
    energy_initial = integral(((vpa,vperp) -> 0.5*m_e*(vpa^2+vperp^2)),Finitial,vpa,vperp)
    energy_final = integral(((vpa,vperp) -> 0.5*m_e*(vpa^2+vperp^2)),Fnew,vpa,vperp)
    energy_expected = integral(((vpa,vperp) -> 0.5*m_e*(vpa^2+vperp^2)),Fexact,vpa,vperp)# 0.75*m_e*c_s^2
    density_expected = density_initial
    nupar_expected = integral(((vpa,vperp) -> vpa),Fexact,vpa,vperp) # n_e*u_i
    if !ci_test
        println("Test time advance, delta t = $delta_t ntime = $ntime")
        println("deviation from exact result: $max_err_Fexact")
        #println("deviation from isotropy: $max_err_Fisotropic")
        println("density(t=0): $density_initial (t=end): $density_final delta: $(density_final - density_initial)")
        println("nupar(t=0): $nupar_initial (t=end): $nupar_final delta: $(nupar_final - nupar_initial)")
        println("energy(t=0): $energy_initial (t=end): $energy_final delta: $(energy_final - energy_initial)")
        println("deviation from expected density: $(density_final - density_expected)")
        println("deviation from expected nupar: $(nupar_final - nupar_expected)")
        println("deviation from expected energy: $(energy_final - energy_expected)")
    else
        @testset "Cei(vpa,vperp) Tests: Time Evolution m_e=$m_e m_i=$m_i" begin
            println("Cei(vpa,vperp) Tests: Time Evolution m_e=$m_e m_i=$m_i")
            @test max_err_Fexact < rtol_exact*maximum(Fexact)
            @test abs(delta_density) < atol_delta_density
            @test abs(delta_nupar) < atol_delta_nupar
            @test abs(delta_energy) < atol_delta_energy
            @test abs(density_initial - n_e) < atol_density
            @test abs(nupar_initial - n_e*u_e) < atol_nupar
            @test abs(energy_initial - 0.75*m_e*n_e*c_e^2) < atol_energy
            @test abs(density_final - density_expected) < atol_density
            @test abs(nupar_final - nupar_expected) < atol_nupar
            @test abs(energy_final - energy_expected) < atol_energy
        end
    end
    return Cei_vpa_vperp_diagnostics(vpa.grid, vperp.grid,
            # evaluation tests
            Cei_result, Cei_exact, Cei_err, max_err_Cei, delta_density, delta_nupar, delta_energy,
            # time advance tests
            Finitial, Fnew, Fexact, Ferr, max_err_Fexact,
            density_initial, density_final, density_expected, density_t,
            nupar_initial, nupar_final, nupar_expected, nupar_t,
            energy_initial, energy_final, energy_expected, energy_t,
            time)
end

end #TestParticleCollisionsVpaVperp

using .TestParticleCollisionsVpaVperp
TestParticleCollisionsVpaVperp.runtests(ci_test=true,nwrite=500,
                                rtol_exact=1.0e-4, atol_delta_density=1.0e-9,
                                atol_delta_nupar=1.0e-13, atol_delta_energy=1.0e-6,
                                atol_density=1.0e-9, atol_nupar=1.0e-13, atol_energy=1.0e-4)