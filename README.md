# FiniteElementAssembly

A package for computing assembled matrix operators appearing in $`C^0`$ continuous Galerkin finite element models using tensor product basis functions on a tensor product coordinate set. Developed from [moment_kinetics](https://github.com/mabarnes/moment_kinetics) and [FokkerPlanck.jl](https://github.com/moment-kinetics/FokkerPlanck.jl).

# Creating a coordinate

The data for each individual 1D coordinate may be created by a call to `FiniteElementCoordinate()`.
Each 1D-coordinate has common information that is independent of the operators that we may need to assemble in a multi-dimensional system. This information includes the grid points of the assembled 1D coordinate `grid`, integration weights `wgts`, and maps between the assembled grid and the unassembled elements.
For example, to create the struct for a radial domain $`r \in (0,1]`$ in a cylinder we would write the following code.
```
# create the coordinate struct 'radial'
using FiniteElementAssembly: FiniteElementCoordinate, ScalarCoordinateInputs, exclude_lower_boundary_point
r_min = 0.0
r_max = 1.0
boundary_point_option = exclude_lower_boundary_point
radial = FiniteElementCoordinate("radial", ScalarCoordinateInputs(ngrid,nelement,
                                r_min,r_max,boundary_point_option),
                                weight_function=((r)-> 2.0*pi*r))

```
Here `ngrid` Is the number of grid points per element, `nelement` is the number of elements, `r_min` is the minimum value taken by the coordinate in the domain, and `r_max` is the maximum value in the domain. `boundary_point_option` must match one of the `@enum` values given below.
```
@enum BoundaryPointType begin
    # do not include end points in the grid
    exclude_boundary_points
    # include both end points in the grid
    include_boundary_points
    # exclude lower endpoint
    exclude_lower_boundary_point
    # exclude upper endpoint
    exclude_upper_boundary_point
end
```
The weight function `Function` type argument is used to include the Jacobian in the `radial.wgts` variable for integration with the convenience function `integral()`. N.B. The `wgts` array uses the same quadrature as the underlying co-location points in `grid`, integrals taken with `integral()` are only exact for sufficiently low-order polynomials.

# Operations on a 1D coordinate

To support pre-existing features of the [moment_kinetics](https://github.com/mabarnes/moment_kinetics) and [FokkerPlanck.jl](https://github.com/moment-kinetics/FokkerPlanck.jl) packages, `FiniteElementAssembly.jl` provides convenience functions for taking first derivatives and carrying out definite integrals in 1D. The syntax for a first derivative is given below.

```
using FiniteElementAssembly
# Create a coordinate x
x = FiniteElementCoordinate("coord", ScalarCoordinateInputs(ngrid,
                                      nelement, coord_min, coord_max, boundary_points))
# create array for the function f(x) to be differentiated/integrated
f = Array{Float64,1}(undef, x.n)
# create array for the derivative df/dx
df = similar(f)
# differentiate f in place
first_derivative!(df, f, x)
```

To integrate in 1D, we then use the following syntax
```
# integrate df/dx
intdf = integral((x->1.0),df, x)
```
where the `Function` type argument permits arbitrary 1D function kernels. Note that the quadrature points `x.wgts` are not adapted (and include the `weight_function` by multiplication), so care must be taken to test convergence in `ngrid` and `nelement` on the user side.

To interpolate a function in 1D, we use the following syntax
```
f_interpolant = interpolate_1D(xj, fdata, x)
```
where `xj` is the value of $`x`$ where we require the interpolated value of a function approximated by `fdata`. The array `fdata` should be of type `<: AbstractArray` with `length(fdata) == x.n`.

# Assembling sparse matrix operators

Once we have created a set of 1D coordinates that represent a N-D physical system, we now may wish to create matrix representations of physical operators in the coordinate space. We provide the function `assemble_operator()` for this purpose. We presently support 1D and 2D cases, see below.
```
# method for assembling operators in one coordinate
assemble_operator(weak_form::Array{Float64,3},
                x::FiniteElementCoordinate,
                boundary_condition_x::BoundaryConditionType)
# method for assembling operators in two coordinates
assemble_operator(weak_form::Function,
                    x::FiniteElementCoordinate,
                    y::FiniteElementCoordinate,
                    boundary_condition_x::BoundaryConditionType,
                    boundary_condition_y::BoundaryConditionType)
```
The returned value is of type `AbstractSparseArray{Float64,Int64,2}`.

The `weak_form` input variable represents the unassembled weak form of the matrix operator. In one dimension the weak form has type `Array{Float64,3}` and must be of size
```
weak_form = Array{Float64,3}(undef,x.ngrid,x.ngrid,x.nelement)
```
The first index represents the source index on the local element (`ixp`), the second index represents the field index on the local element (`ix`) and the third index represents the index labelling the element such that a local vector `v` on the `ie`th element could be constructed from the local vector `u` as follows.
```
u = ones(x.ngrid)
v = zeros(x.ngrid)
for ix in 1:x.ngrid
    for ixp in 1:x.ngrid
        v[ix] = weak_form[ixp,ix,ie]*u[ixp]
    end
end
```
As this example implies, this choice is to allow the local weak form matrices to be used for quick summation, allowing for non-linear operators to be constructed from the same local weak forms as used to construct linear matrix operators through `assemble_operator()`.

For 2D matrices the `weak_form` input variable must be of the type `Function`, returning a `Float64` and have the following arguments
```
weak_form(ixp_local,ix_local,ielement_x,
        iyp_local,iy_local,ielement_y)::Float64
```
where `ixp_local` is the source index on the local `x` element, `ix_local` is the field index on the local `x` element, `ielement_x` is the element index in `x`, `iyp_local` is the source index on the local `y` element, `iy_local` is the field index on the local `y` element, and `ielement_y` is the element index in `y`.

The matrices constructed using `assemble_operator()` may be used as any other `AbstractSparseArray`. E.g., one can calculate the LU decomposition and use this decomposition to solve a matrix problem. Given the weak form function `Laplacian_weak_form()` for a Laplacian operator in cylindrical coordinates `radial`, `theta`, we could write
```
using LinearAlgebra: lu
Laplacian_Dirichlet_bc = assemble_operator(Laplacian_weak_form,
                            radial, theta, DirichletBC(), PeriodicBC())
lu_Laplacian_Dirichlet_bc = lu(Laplacian_Dirichlet_bc)
```
and use `lu_Laplacian_Dirichlet_bc` with `ldiv!` as usual. See `test/PoissonSolverRadialThetaTests.jl` for more details.

The internally supported boundary conditions are
```
abstract type BoundaryConditionType end
struct NaturalBC <: BoundaryConditionType end
struct DirichletBC <: BoundaryConditionType end
struct PeriodicBC <: BoundaryConditionType end
```
The user can define their own boundary condition types and extend the boundary condition function `impose_boundary_condition()` as required. Note that we explicitly supply the boundary condition choice for each call to `assemble_operator()`. The option to pass a boundary condition instance to the coordinate struct is given, e.g., from `test/PeriodicBcTests.jl`,
```
x = FiniteElementCoordinate("coord", ScalarCoordinateInputs(ngrid,
                            nelement, coord_min, coord_max, include_boundary_points),
                            bc = PeriodicBC())
```
to permit `first_derivative!(df, f, x)` to respect boundary conditions, and for the user to store a single boundary condition for passing around the program (see [FokkerPlanck.jl](https://github.com/moment-kinetics/FokkerPlanck.jl)). However, note that not all operators use the same boundary condition, even within the one model so in `assemble_operator()` we do not assume that the boundary condition passed in the coordinate struct.

# Tests

This package comes with numerous tests of diffusion and Fokker--Planck type operators. They are as follows.
 - `test/CalculusTests.jl`: test first derivatives and integral convenience functions in one dimension.
 - `test/InterpolationTests.jl`: test interpolation convenience functions.
 - `test/InterfaceTests.jl`: test the implementation of the FiniteElementCoordinate interface allowing for scalar coordinate inputs (number of grid points per element, number of elements, et cetera) for uniformly spaced element boundaries, and a more general constructor function where the element collocation points and scale and shift factors are specified directly.
 - `test/PeriodicBcTests.jl`: test differentiation in 1D with periodic boundary conditions.
 - `test/PoissonSolverRadialThetaTests.jl`: test the solution of Poisson's equation in cylindrical $`(r,\theta)`$  coordinates with periodic boundary conditions in $`\theta`$.
 - `test/PoissonSolverThetaZedTests.jl`: test the solution of Poisson's equation in $`(\theta,z)`$ coordinates with
periodic boundary conditions in $`\theta`$.
 - `test/PoissonSolverZedRadialTests.jl`: test the solution of Poisson's equation
in cylindrical $`(z,r)`$ coordinates.
 - `test/SlowingDownTest.jl`: test the implementation of the slowing down operator for a trace distribution of $`\alpha`$ particles colliding with a background of ions and electrons. We solve the equation
 ```math
 \frac{\partial F_{\alpha}}{\partial t} = C_{\alpha}[F_{\alpha}] + S_{\alpha}.
 ```
 - `test/ElectronIonCollisionsTest.jl`: test the implementation of electron-ion collisions in a speed, pitch-angle $`(v,\xi)`$ coordinate system. The ions are assumed to have a Maxwellian distribution function characterised by density
and temperature. We solve the equation
 ```math
 \frac{\partial F_{e}}{\partial t} = C_{ei}[F_e,F_{i,M}].
 ```
 - `test/TestParticleCollisionsVpaVperp.jl`: test the implementation of electron-ion collisions in a $`(v_{\|},v_\perp)`$ coordinate system. The ions are assumed to have a Maxwellian distribution function characterised by density, mean flow, and temperature.


