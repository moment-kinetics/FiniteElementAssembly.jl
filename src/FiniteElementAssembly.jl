module FiniteElementAssembly

export FiniteElementCoordinate,
    ScalarCoordinateInputs,
    # choice of boundary points on grid
    exclude_boundary_points, include_boundary_points,
    exclude_lower_boundary_point, exclude_upper_boundary_point,
    BoundaryPointType,
    impose_boundary_condition,
    AbstractBoundaryConditionType, NaturalBC, DirichletBC, PeriodicBC,
    assemble_operator,
    # testing
    set_element_boundaries,
    set_element_scale_and_shift

using LinearAlgebra
using SparseArrays: sparse, AbstractSparseArray
using SuiteSparse
using LagrangePolynomials: LagrangePolyData, lagrange_poly
using FiniteElementMatrices: ElementCoordinates,
                             lagrange_x,
                             d_lagrange_dx,
                             finite_element_matrix
using FastGaussQuadrature: gausslegendre, gausslobatto, gaussradau

abstract type AbstractBoundaryConditionType end
struct NaturalBC <: AbstractBoundaryConditionType end
struct DirichletBC <: AbstractBoundaryConditionType end
struct PeriodicBC <: AbstractBoundaryConditionType end

"""
struct containing information for first derivatives
"""
struct FirstDerivativeData{TFloat <: Real, TMatrix <: AbstractSparseArray{TFloat,Int64,2}}
    # Local mass matrix = \int phi_i(x) phi_j(x) dx
    MM::Array{Float64,3}
    # Local first derivative matrix \int phi_i(x) phi'_j(x) dx
    PP::Array{Float64,3}
    # dummy array for storing intermediate results
    dummy_rhs::Array{Float64,1}
    # dummy array for storing solution
    dummy_df::Array{Float64,1}
    # Assembled 1D mass matrix
    MM1D::TMatrix
    # Assembled 1D first derivative matrix
    PP1D::TMatrix
    # LU object for mass matrix solve
    lu_MM1D::SuiteSparse.UMFPACK.UmfpackLU{Float64,Int64}
    """
    Function to initialise FirstDerivativeData
    from information from the grid.
    """
    function FirstDerivativeData(ngrid::Int64,
                                        nelement::Int64,
                                        n::Int64,
                                        igrid_full::Array{Int64,2},
                                        element_data::Array{ElementCoordinates,1},
                                        boundary_condition::Tbc) where Tbc <: AbstractBoundaryConditionType
        # Local mass matrix MM[j,i] = \int phi_i(x) phi_j(x) dx
        MM = Array{Float64,3}(undef,ngrid,ngrid,nelement)
        # PP[j,i] = \int phi_i(x) phi'_j(x) dx
        PP = Array{Float64,3}(undef,ngrid,ngrid,nelement)
        # get local matrices on each element
        for ielement in 1:nelement
            xdata = element_data[ielement]
            @views MM[:,:,ielement] = finite_element_matrix(lagrange_x,lagrange_x,0,xdata)
            @views PP[:,:,ielement] = finite_element_matrix(d_lagrange_dx,lagrange_x,0,xdata)
        end
        # dummy arrays
        dummy_rhs = Array{Float64,1,}(undef,n)
        dummy_df = Array{Float64,1,}(undef,n)

        # assemble the global matrices with a sparse construction
        MM1D = assemble_1D_operator(MM, ngrid, nelement,
                            n, igrid_full, boundary_condition)
        PP1D = assemble_1D_operator(PP, ngrid, nelement,
                            n, igrid_full, boundary_condition)
        MM1D_sparse = sparse(MM1D)
        PP1D_sparse = sparse(PP1D)
        lu_MM1D = lu(MM1D_sparse)
        return new{Float64,typeof(MM1D)}(MM,PP,dummy_rhs,dummy_df,
                MM1D_sparse,PP1D_sparse,lu_MM1D)
    end
end

"""
Compound index for the sparse assembly of a 1D
finite element matrix. Note that this compound
index is different to the one used internally
by the object created by `SparseArrays.sparse()`.
The maximum value is
 `nsparse_1D = (ngrid^2 - 1)*(nelement - 1) + ngrid^2`.
"""
function icsc1D(ixp_local::Int64,
            ix_local::Int64,
            ielement::Int64,
            ngrid::Int64)
    icsc = 1 + ((ixp_local - 1) + (ix_local - 1)*ngrid +
                (ielement - 1)*(ngrid^2 - 1))
    return icsc
end
"""
Function to assemble 1D operators within the coordinate struct constructor.
"""
function assemble_1D_operator(QQ1D_local::Array{Float64,3},
                            ngrid::Int64,
                            nelement::Int64,
                            n::Int64,
                            igrid_full::Array{Int64,2},
                            boundary_condition::Tbc) where Tbc <: AbstractBoundaryConditionType
    # create the 1D constructor arrays
    nsparse = (nelement - 1)*(ngrid^2 - 1) + ngrid^2
    II = zeros(Int64,nsparse)
    JJ = zeros(Int64,nsparse)
    VV = zeros(Float64,nsparse)
    # loop over elements
    for ielement in 1:nelement
        @views QQ1D = QQ1D_local[:,:,ielement]
        @views igrid = igrid_full[:,ielement]
        # loop over points within elements
        for ix in 1:ngrid
            for ixp in 1:ngrid
                # convert local indices within
                # elements to global indices and
                # the constructor compound index
                ix_global = igrid[ix]
                ixp_global = igrid[ixp]
                isparse = icsc1D(ixp,ix,ielement,ngrid)
                # assign data
                II[isparse] = ix_global
                JJ[isparse] = ixp_global
                VV[isparse] += QQ1D[ixp,ix]
            end
        end
    end
    # convert constructors to sparse matrix
    QQ1D_global = sparse(II,JJ,VV)
    # impose BC on the assembled sparse matrix
    impose_boundary_condition_x(boundary_condition, QQ1D_global, n, ngrid, QQ1D_local)
    return QQ1D_global
end

"""
Function that returns the sparse matrix index
used to directly construct the nonzero entries
of a 2D assembled sparse matrix.
The maximum value is
 `nsparse_2D = ((ngrid_x^2 - 1)*(nelement_x - 1) + ngrid_x^2)*
    ((ngrid_y^2 - 1)*(nelement_y - 1) + ngrid_y^2)`.
"""
function icsc2D(ixp_local::Int64,ix_local::Int64,ielement_x::Int64,ngrid_x::Int64,nelement_x::Int64,
                iyp_local::Int64,iy_local::Int64,ielement_y::Int64,ngrid_y::Int64)
    # maximum number of x is the same as the maximum icsc1D index
    ntot_x = icsc1D(ngrid_x,ngrid_x,nelement_x,ngrid_x)
    # icsc indexes for x and y
    icsc_x = icsc1D(ixp_local,ix_local,ielement_x,ngrid_x)
    icsc_y = icsc1D(iyp_local,iy_local,ielement_y,ngrid_y)
    # compound index for xy
    icsc = icsc_x + ntot_x*(icsc_y - 1)
    return icsc
end

"""
    ixy_func(ix::Int64,iy::Int64,nx::Int64)

Get the 'linear index' corresponding to `ix` and `iy`. Defined so that the linear
index corresponds to the underlying layout in memory of a 2d array indexed by
`[ix,iy]`, i.e. for a 2d array `f2d`:
* `size(f2d) == (x.n, y.n)`
* For a reference to `f2d` that is reshaped to a vector (a 1d array) `f1d = vec(f2d)` than
  for any `ix` and `iy` it is true that `f1d[ixy_func(ix,iy)] ==
  f2d[ix,iy]`.
"""
function ixy_func(ix::Int64,iy::Int64,nx::Int64)
    return ix + nx*(iy-1)
end

"""
enum for determining whether or not to include
points on the boundary in the grid.
"""
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
"""
This struct encapsulates the unique information
required to create an instance of
```
    Array{ElementCoordinates,1}
```
to permit an interface with a single function
for initialising the arrays for the finite element
problem, whilst retaining the convenience of
not having to construct the `ElementCoordinates`
types manually in all tests.
"""
struct ScalarCoordinateInputs
    # ngrid is number of grid points per element
    ngrid::Int64
    # nelement is the number of elements in total
    nelement::Int64
    # coord_min is the minimum value of this coordinate
    coord_min::Float64
    # coord_max is the minimum value of this coordinate
    coord_max::Float64
    # enum to determine whether or not boundary points
    # are included in the grid
    boundary_points::BoundaryPointType
end

"""
structure containing basic information related to coordinates
"""
struct FiniteElementCoordinate{Tbc <: AbstractBoundaryConditionType}
    # name is the name of the variable associated with this coordiante
    name::String
    # n is the total number of grid points associated with this coordinate
    n::Int64
    # ngrid is the number of grid points per element in this coordinate
    ngrid::Int64
    # nelement is the number of elements associated with this coordinate
    nelement::Int64
    # L is the box length in this coordinate
    L::Float64
    # grid is the location of the grid points
    grid::Array{Float64,1}
    # igrid contains the grid point index within the element
    igrid::Array{Int64,1}
    # ielement contains the element index
    ielement::Array{Int64,1}
    # imin[j] contains the minimum index on the full grid for element j
    imin::Array{Int64,1}
    # imax[j] contains the maximum index on the full grid for element j
    imax::Array{Int64,1}
    # igrid_full[i,j] contains the index of the full grid for the elemental grid point i, on element j
    igrid_full::Array{Int64,2}
    # bc is the boundary condition option stored in this coordinate for convenience.
    # and used automatically in the application of first derivatives only.
    # Note that the boundary condition must be separately supplied in the construction
    # of other operators, since not all operators have the same boundary conditions on
    # the same coordinates.
    bc::Tbc
    # wgts contains the integration weights associated with each grid point
    wgts::Array{Float64,1}
    # scale for each element
    element_scale::Array{Float64,1}
    # shift for each element
    element_shift::Array{Float64,1}
    # list of element boundaries
    element_boundaries::Array{Float64,1}
    # Lagrange Polynomial data for each element
    lpoly_data::Union{Array{LagrangePolyData,1},Nothing}
    # Coordinate data for each element
    element_data::Union{Array{ElementCoordinates,1},Nothing}
    # data required to take a first derivative (only)
    derivative_data::Union{FirstDerivativeData,Nothing}
    # flag to determine if lower endpoint is on the grid
    include_lower_boundary::Bool
    # flag to determine if upper endpoint is on the grid
    include_upper_boundary::Bool
    """
    This internal constructor for `FiniteElementCoordinate`
    takes `scalar_input = ScalarCoordinateInputs(ngrid,nelement,coord_min,coord_max)` to construct
    the `element_data` struct for the fundamental constructor.
    """
    function FiniteElementCoordinate(
        # name of coordinate
        name::String,
        # the inputs used to construct a FEM grid
        scalar_input::ScalarCoordinateInputs;
        # the 1D Jacobian or weight function rho(x) that
        # appears in 1D integrals as \int (.) rho(x) d x
        weight_function::TF=((x)-> 1.0 ),
        # which boundary condition to store and use for first derivatives
        bc::AbstractBoundaryConditionType=NaturalBC()
        ) where TF <: Function
        ngrid = scalar_input.ngrid
        nelement = scalar_input.nelement
        # initialise the data used to construct the grid
        # boundaries for each element
        element_boundaries = set_element_boundaries(scalar_input)
        # shift and scale factors for each local element
        element_scale, element_shift =
            set_element_scale_and_shift(element_boundaries)
        if ngrid > 1
            # get the nodes on [-1,1] for each element
            reference_nodes = reference_grids(scalar_input)
            element_data = Array{ElementCoordinates,1}(undef,nelement)
            for ielement in 1:nelement
                # get the reference nodes defined on [-1,1] (or (-1,1] on radau elements))
                scale = element_scale[ielement]
                shift = element_shift[ielement]
                @views x_nodes = reference_nodes[:,ielement]
                element_data[ielement] = ElementCoordinates(x_nodes,
                                                        scale,
                                                        shift)
            end
        else
            element_data = nothing
        end
        return FiniteElementCoordinate(name, element_data, weight_function=weight_function, bc=bc)
    end
    """
    This is the fundamental internal constructor
    for `FiniteElementCoordinate`, which takes
    `element_data::Union{Array{ElementCoordinates,1},Nothing}`
    as an argument to define the grid.

    The option to pass a value with type `Nothing` is
    required to permit a trivial coordinate of 1 point
    to be constructed with this function.
    """
    function FiniteElementCoordinate(
        # name of coordinate
        name::String,
        # array containing data defining element grids
        # from which the coordinate struct can be created
        element_data::Union{Array{ElementCoordinates,1},Nothing};
        # the 1D Jacobian or weight function rho(x) that
        # appears in 1D integrals as \int (.) rho(x) d x
        weight_function::TF=((x)-> 1.0 ),
        # which boundary condition to store and use in first derivatives
        bc::AbstractBoundaryConditionType=NaturalBC()
        ) where TF <: Function
        if typeof(element_data) == Nothing
            # this is a trivial coordinate of length 1
            nelement = 1
            ngrid = 1
            element_scale = ones(Float64,1)
            element_shift = zeros(Float64,1)
            element_boundaries = zeros(Float64,2)
            element_boundaries[1] = -1.0
            element_boundaries[2] = 1.0
        else
            # number of elements
            nelement = length(element_data)
            # number of grid points per element
            ngrid = length(element_data[1].lpoly_data.x_nodes)
            # check ngrid the same for each element
            for j in 2:nelement
                if !(length(element_data[j].lpoly_data.x_nodes) == ngrid)
                    error("length(element_data[j].lpoly_data.x_nodes) /= ngrid \n Number of nodes in reference grid must be the same for each element")
                end
            end
            # extract shift, scale, and boundary values
            element_scale = Array{Float64,1}(undef,nelement)
            element_shift = Array{Float64,1}(undef,nelement)
            element_boundaries = Array{Float64,1}(undef,nelement+1)
            for j in 1:nelement
                element_scale[j] = element_data[j].scale
                element_shift[j] = element_data[j].shift
                element_boundaries[j] = element_shift[j] - element_scale[j]
            end
            element_boundaries[nelement+1] = element_scale[nelement] + element_shift[nelement]
        end
        # total number of grid points is ngrid for the first element
        # plus ngrid-1 unique points for each additional element due
        # to the repetition of a point at the element boundary
        n_global = (ngrid-1)*nelement + 1
        # obtain index mapping from full (local) grid to the
        # grid within each element (igrid, ielement)
        igrid, ielement = full_to_elemental_grid_map(ngrid,
                            nelement, n_global)
        # obtain (local) index mapping from the grid within each element
        # to the full grid
        imin, imax, igrid_full = elemental_to_full_grid_map(ngrid,
                                                            nelement)
        # initialize the grid and the integration weights associated with the grid
        grid = Array{Float64,1}(undef,n_global)
        wgts = zeros(Float64,n_global)
        if n_global > 1
            nquad = 2*ngrid
            zz, wz = gausslegendre(nquad)
            k = 1
            for j in 1:nelement
                # extract reference nodes
                x_nodes = element_data[j].lpoly_data.x_nodes
                # calculate weights on the reference nodes
                wgts_nodes = Array{Float64,1}(undef,ngrid)
                for i in 1:ngrid
                    ith_lpoly_data = element_data[j].lpoly_data.lpoly_data[i]
                    result = 0.0
                    for l in 1:nquad
                        result += wz[l]*lagrange_poly(ith_lpoly_data,zz[l])
                    end
                    wgts_nodes[i] = result
                end
                # put this data into the global arrays
                scale = element_scale[j]
                shift = element_shift[j]
                @. grid[igrid_full[k,j]:igrid_full[ngrid,j]] = x_nodes[k:ngrid]*scale + shift
                @. wgts[igrid_full[1,j]:igrid_full[ngrid,j]] += wgts_nodes[1:ngrid]*scale
                k = 2
            end
            # include Jacobian factor consistent with original grid
            for i in 1:n_global
                wgts[i] *= weight_function(grid[i])::Float64
            end
        else
            grid[1] = 0.0
            wgts[1] = 1.0
        end
        if ngrid > 1
            lpoly_data = Array{LagrangePolyData,1}(undef,nelement)
            for ielement in 1:nelement
                # get the local grid in global coord system
                grid_local = grid[igrid_full[1,ielement]:igrid_full[ngrid,ielement]]
                # get Lagrange Poly data for interpolating in global coordinates
                lpoly_data[ielement] = LagrangePolyData(grid_local)
            end
        else
            lpoly_data = nothing
        end
        if typeof(element_data) == Nothing
            derivative_data = nothing
        else
            derivative_data = FirstDerivativeData(ngrid, nelement,
                                        n_global, igrid_full, element_data,
                                        bc)
        end
        domainLength = element_boundaries[end] - element_boundaries[1]
        tolerance = 1.0e-13
        include_lower_boundary = (abs(element_boundaries[1] - grid[1]) < tolerance)
        include_upper_boundary = (abs(element_boundaries[end] - grid[end]) < tolerance)
        if typeof(bc) == PeriodicBC && (!include_lower_boundary || !include_upper_boundary)
            error("Must include upper and lower boundaries on the grid for typeof(bc) = $(typeof(bc))")
        end
        return new{typeof(bc)}(name, n_global, ngrid,
            nelement, domainLength, grid, igrid, ielement, imin, imax,
            igrid_full, bc, wgts,
            element_scale, element_shift, element_boundaries,
            lpoly_data, element_data, derivative_data,
            include_lower_boundary, include_upper_boundary)
    end
end

function set_element_boundaries(scalar_input::ScalarCoordinateInputs)
    nelement_global = scalar_input.nelement
    coord_min = scalar_input.coord_min
    coord_max = scalar_input.coord_max
    Lcoord = coord_max - coord_min
    # set global element boundaries between [-L/2,L/2]
    element_boundaries = Array{Float64,1}(undef,nelement_global+1)
    for j in 1:nelement_global+1
        element_boundaries[j] = Lcoord*((j-1)/(nelement_global)) + coord_min
    end
    return element_boundaries
end

function set_element_scale_and_shift(element_boundaries::Array{Float64,1})
    nelement = length(element_boundaries) - 1
    element_scale = Array{Float64,1}(undef,nelement)
    element_shift = Array{Float64,1}(undef,nelement)
    for j in 1:nelement
        upper_boundary = element_boundaries[j+1]
        lower_boundary = element_boundaries[j]
        element_scale[j] = 0.5*(upper_boundary-lower_boundary)
        element_shift[j] = 0.5*(upper_boundary+lower_boundary)
    end
    return element_scale, element_shift
end

"""
Makes an array `reference_nodes` of shape `(ngrid,nelement)`
with `reference_nodes[:,j]` the nodes of the jth element
defined on the grid that goes from -1,1,
"""
function reference_grids(scalar_input::ScalarCoordinateInputs)
    ngrid=scalar_input.ngrid
    nelement=scalar_input.nelement
    boundary_points=scalar_input.boundary_points
    reference_nodes = Array{Float64,2}(undef,ngrid,nelement)
    # get Gauss-Legendre-Lobatto points and weights on [-1,1]
    x_lob, w_lob = gausslobatto(ngrid)
    # get Gauss-Legendre-Radau points and weights on [-1,1)
    x_rad, w_rad = gaussradau(ngrid)
    # transform to a Gauss-Legendre-Radau grid on (-1,1]
    x_rad_exclude_lower, w_rad_exclude_lower = -reverse(x_rad), reverse(w_rad)

    if nelement > 1
        # multiple elements, so upper and lower boundary endpoints
        # appear in different sets of nodes
        exclude_lower = ((boundary_points == exclude_lower_boundary_point) ||
                     (boundary_points == exclude_boundary_points))
        exclude_upper = ((boundary_points == exclude_upper_boundary_point) ||
                     (boundary_points == exclude_boundary_points))
        # lowest element
        if exclude_lower
            # Gauss-Legendre-Radau (-1,1] to exclude lower boundary
            # but include interior element endpoints for continuity
            reference_nodes[:,1] .= x_rad_exclude_lower
        else
            reference_nodes[:,1] .= x_lob
        end
        # interior elements, always Gauss-Legendre-Lobatto [-1,1]
        # to enforce continuity at interior element boundaries
        for j in 2:nelement-1
            reference_nodes[:,j] .= x_lob
        end
        # uppermost element
        if exclude_upper
            # Gauss-Legendre-Radau [-1,1) to exclude upper boundary
            # but include interior boundary endpoints for continuity
            reference_nodes[:,nelement] .= x_rad
        else
            reference_nodes[:,nelement] .= x_lob
        end
    else # only a single element
        if boundary_points == exclude_boundary_points
            # exclude all boundary points by using (-1,1)
            x, w = gausslegendre(ngrid)
            reference_nodes[:,1] .= x
        elseif boundary_points == exclude_lower_boundary_point
            reference_nodes[:,1] .= x_rad_exclude_lower
        elseif boundary_points == exclude_upper_boundary_point
            reference_nodes[:,1] .= x_rad
        else
            reference_nodes[:,1] .= x_lob
        end
    end
    return reference_nodes
end

"""
setup arrays containing a map from the unpacked grid point indices
to the element index and the grid point index within each element
"""
function full_to_elemental_grid_map(ngrid::Int64,
                        nelement::Int64, n::Int64)
    igrid = Array{Int64,1}(undef, n)
    ielement = Array{Int64,1}(undef, n)
    k = 1
    for i ∈ 1:ngrid
        ielement[k] = 1
        igrid[k] = i
        k += 1
    end
    if nelement > 1
        for j ∈ 2:nelement
            # avoid double-counting overlapping point
            # at boundary between elements
            for i ∈ 2:ngrid
                ielement[k] = j
                igrid[k] = i
                k += 1
            end
        end
    end
    return igrid, ielement
end

"""
returns imin and imax, which contain the minimum and maximum
indices on the full grid for each element
"""
function elemental_to_full_grid_map(ngrid::Int64, nelement::Int64)
    imin = Array{Int64,1}(undef, nelement)
    imax = Array{Int64,1}(undef, nelement)
    igrid_full = Array{Int64,2}(undef, ngrid, nelement)
    @inbounds begin
        # the first element contains ngrid entries
        imin[1] = 1
        imax[1] = ngrid
        # each additional element contributes ngrid-1 unique entries
        # due to repetition of one grid point at the boundary
        if nelement > 1
            for i ∈ 2:nelement
                imin[i] = imax[i-1] + 1
                imax[i] = imin[i] + ngrid - 2
            end
        end

        for j in 1:nelement
            for i in 1:ngrid
                igrid_full[i,j] = i + (j - 1)*(ngrid - 1)
            end
        end
    end
    return imin, imax, igrid_full
end

"""
    ixy_compound_index(x,y,ielement_x,ielement_y,ix_local,iy_local)

For local (within the single element specified by `ielement_x` and `ielement_y`)
indices `ix_local` and `iy_local`, get the global index in the 'linear-indexed' 2d
space of size `(x.n, y.n)` (as returned by [`ixy_func`](@ref)).
"""
function ixy_compound_index(x::FiniteElementCoordinate,
                    y::FiniteElementCoordinate,
                    ielement_x::Int64,ielement_y::Int64,
                    ix_local::Int64,iy_local::Int64)
    # global indices on the grids
    ix_global = x.igrid_full[ix_local,ielement_x]
    iy_global = y.igrid_full[iy_local,ielement_y]
    # global compound index
    ixy_global = ixy_func(ix_global,iy_global,x.n)
    return ixy_global
end

"""
Methods for imposing boundary conditions on 1D operators
"""
function impose_boundary_condition_x(boundary_condition::Union{NaturalBC,DirichletBC},
        operator_sparse::AbstractSparseArray{Float64,Int64,2},
        n::Int64, ngrid::Int64, QQ::AbstractArray{Float64,3})
    # do nothing
    return nothing
end
function impose_boundary_condition_x(boundary_condition::PeriodicBC,
        operator_sparse::AbstractSparseArray{Float64,Int64,2},
        n::Int64, ngrid::Int64, QQ::AbstractArray{Float64,3})
    # periodic BC contributions
    # set lower row to (1, 0, ..., 0, -1)
    operator_sparse[1,:] .= 0.0
    operator_sparse[1,1] = 1.0
    operator_sparse[1,n] = -1.0
    # assemble lower row to upper row
    for ixp in 1:ngrid
        operator_sparse[n,ixp] += QQ[ixp,1,1]
    end
    return nothing
end
"""
Methods for imposing boundary conditions on 2D operators
"""
function impose_boundary_condition_x(boundary_condition::NaturalBC,
            operator_sparse::AbstractSparseArray{Float64,Int64,2},
            x::FiniteElementCoordinate,y::FiniteElementCoordinate,
            weak_form::TF) where TF <: Function
    # do nothing
    return nothing
end
function impose_boundary_condition_x(boundary_condition::DirichletBC,
            operator_sparse::AbstractSparseArray{Float64,Int64,2},
            x::FiniteElementCoordinate,y::FiniteElementCoordinate,
            weak_form::TF) where TF <: Function
    if x.include_lower_boundary
        # set row to (1, 0, 0, ..., 0)
        for iy in 1:y.n
            ixy = ixy_func(1,iy,x.n)
            operator_sparse[ixy,:] .= 0.0
            operator_sparse[ixy,ixy] = 1.0
        end
    end
    if x.include_upper_boundary
        # set row to (1, 0, 0, ..., 0)
        for iy in 1:y.n
            ixy = ixy_func(x.n,iy,x.n)
            operator_sparse[ixy,:] .= 0.0
            operator_sparse[ixy,ixy] = 1.0
        end
    end
    return nothing
end
function impose_boundary_condition_x(boundary_condition::PeriodicBC,
            operator_sparse::AbstractSparseArray{Float64,Int64,2},
            x::FiniteElementCoordinate,y::FiniteElementCoordinate,
            weak_form::TF) where TF <: Function
    if x.include_lower_boundary
        # set row to (1, 0, 0, ..., 0, -1)
        for iy in 1:y.n
            ixy = ixy_func(1,iy,x.n)
            ixyp = ixy_func(x.n,iy,x.n)
            operator_sparse[ixy,:] .= 0.0
            operator_sparse[ixy,ixy] = 1.0
            operator_sparse[ixy,ixyp] = -1.0
        end
    else
        error("impose_boundary_condition_x(): x.include_lower_boundary=true is required for boundary_condition_x=PeriodicBC()")
    end
    if x.include_upper_boundary
        # assemble lower row contribution to upper row
        ielement_x = 1
        # x index of row assembled to
        ix = x.n
        # x local index of row assembled from
        ix_local = 1
        for ielement_y in 1:y.nelement
            for iy_local in 1:y.ngrid
                iy = y.igrid_full[iy_local,ielement_y]
                # compound index for global ix, iy
                ixy = ixy_func(ix,iy,x.n)
                for iyp_local in 1:y.ngrid
                    iyp = y.igrid_full[iyp_local,ielement_y]
                    for ixp_local in 1:x.ngrid
                        # x' index corresponds to lower boundary element
                        ixp = x.igrid_full[ixp_local,ielement_x]
                        # compound index for global ix', iy'
                        ixyp = ixy_func(ixp,iyp,x.n)
                        # add the contribution
                        operator_sparse[ixy,ixyp] += weak_form(ixp_local, ix_local, ielement_x,
                                                        iyp_local, iy_local, ielement_y)::Float64
                    end
                end
            end
        end
    else
        error("impose_boundary_condition_x(): x.include_upper_boundary=true is required for boundary_condition_x=PeriodicBC()")
    end
    return nothing
end

function impose_boundary_condition_y(boundary_condition::NaturalBC,
            operator_sparse::AbstractSparseArray{Float64,Int64,2},
            x::FiniteElementCoordinate,y::FiniteElementCoordinate,
            weak_form::TF) where TF <: Function
    # do nothing
    return nothing
end
function impose_boundary_condition_y(boundary_condition::DirichletBC,
            operator_sparse::AbstractSparseArray{Float64,Int64,2},
            x::FiniteElementCoordinate,y::FiniteElementCoordinate,
            weak_form::TF) where TF <: Function
    if y.include_lower_boundary
        # set row to (1, 0, 0, ..., 0)
        for ix in 1:x.n
            ixy = ixy_func(ix,1,x.n)
            operator_sparse[ixy,:] .= 0.0
            operator_sparse[ixy,ixy] = 1.0
        end
    end
    if y.include_upper_boundary
        # set row to (1, 0, 0, ..., 0)
        for ix in 1:x.n
            ixy = ixy_func(ix,y.n,x.n)
            operator_sparse[ixy,:] .= 0.0
            operator_sparse[ixy,ixy] = 1.0
        end
    end
    return nothing
end
function impose_boundary_condition_y(boundary_condition::PeriodicBC,
            operator_sparse::AbstractSparseArray{Float64,Int64,2},
            x::FiniteElementCoordinate,y::FiniteElementCoordinate,
            weak_form::TF) where TF <: Function
    if y.include_lower_boundary
        # set row to (1, 0, 0, ..., 0, -1)
        for ix in 1:x.n
            ixy = ixy_func(ix,1,x.n)
            ixyp = ixy_func(ix,y.n,x.n)
            operator_sparse[ixy,:] .= 0.0
            operator_sparse[ixy,ixy] = 1.0
            operator_sparse[ixy,ixyp] = -1.0
        end
    else
        error("impose_boundary_condition_y(): y.include_lower_boundary=true is required for boundary_condition_y=PeriodicBC()")
    end
    if y.include_upper_boundary
        # assemble lower row contribution to upper row
        ielement_y = 1
        # y index of row assembled to
        iy = y.n
        # local y index of row assembled from
        iy_local = 1
        for ielement_x in 1:x.nelement
            for ix_local in 1:x.ngrid
                ix = x.igrid_full[ix_local,ielement_x]
                # compound index for global ix, iy
                ixy = ixy_func(ix,iy,x.n)
                for iyp_local in 1:y.ngrid
                    # y' index corresponds to lower boundary element
                    iyp = y.igrid_full[iyp_local,ielement_y]
                    for ixp_local in 1:x.ngrid
                        ixp = x.igrid_full[ixp_local,ielement_x]
                        # compound index for global ix', iy'
                        ixyp = ixy_func(ixp,iyp,x.n)
                        # add the contribution
                        operator_sparse[ixy,ixyp] += weak_form(ixp_local, ix_local, ielement_x,
                                                        iyp_local, iy_local, ielement_y)::Float64
                    end
                end
            end
        end
    else
        error("impose_boundary_condition_y(): y.include_upper_boundary=true is required for boundary_condition_y=PeriodicBC()")
    end
    return nothing
end

function impose_boundary_condition(
            boundary_condition_x::PeriodicBC,boundary_condition_y::Union{NaturalBC,DirichletBC},
            operator_sparse::AbstractSparseArray{Float64,Int64,2},
            x::FiniteElementCoordinate,y::FiniteElementCoordinate,
            weak_form::TF) where TF <: Function
    # impose periodic BC first to ensure Dirichlet BC is imposed last
    impose_boundary_condition_x(boundary_condition_x,operator_sparse,x,y,weak_form)
    impose_boundary_condition_y(boundary_condition_y,operator_sparse,x,y,weak_form)
    #impose_boundary_condition_x(boundary_condition_x,operator_sparse,x,y,weak_form)
    return nothing
end
function impose_boundary_condition(
            boundary_condition_x::Union{NaturalBC,DirichletBC},boundary_condition_y::PeriodicBC,
            operator_sparse::AbstractSparseArray{Float64,Int64,2},
            x::FiniteElementCoordinate,y::FiniteElementCoordinate,
            weak_form::TF) where TF <: Function
    # impose periodic BC first to ensure Dirichlet BC is imposed last
    impose_boundary_condition_y(boundary_condition_y,operator_sparse,x,y,weak_form)
    impose_boundary_condition_x(boundary_condition_x,operator_sparse,x,y,weak_form)
    return nothing
end
function impose_boundary_condition(
            boundary_condition_x::Union{NaturalBC,DirichletBC},boundary_condition_y::Union{NaturalBC,DirichletBC},
            operator_sparse::AbstractSparseArray{Float64,Int64,2},
            x::FiniteElementCoordinate,y::FiniteElementCoordinate,
            weak_form::TF) where TF <: Function
    # order of imposition is unimportant
    impose_boundary_condition_x(boundary_condition_x,operator_sparse,x,y,weak_form)
    impose_boundary_condition_y(boundary_condition_y,operator_sparse,x,y,weak_form)
    return nothing
end
"""
External method for assembling operators in one coordinate
"""
function assemble_operator(weak_form::Array{Float64,3},
                            x::FiniteElementCoordinate,
                            boundary_condition_x::AbstractBoundaryConditionType)
    return assemble_1D_operator(weak_form,
            x.ngrid, x.nelement, x.n, x.igrid_full,
            boundary_condition_x)
end

"""
Method for assembling operators in two coordinates
"""
function assemble_operator(weak_form::TF,
                    x::FiniteElementCoordinate,
                    y::FiniteElementCoordinate,
                    boundary_condition_x::AbstractBoundaryConditionType,
                    boundary_condition_y::AbstractBoundaryConditionType
                    ) where TF <: Function
    # Assemble a 2D mass matrix in the global compound coordinate
    # total number of non-zero element is the maximum index of
    # the sparse matrix index, by construction
    nsparse = icsc2D(x.ngrid,x.ngrid,x.nelement,x.ngrid,x.nelement,
                    y.ngrid,y.ngrid,y.nelement,y.ngrid)
    # data required to make a sparse matrix V_{IJ},
    # with I, J the row and column indices and V the value
    II = zeros(Int64,nsparse)
    JJ = zeros(Int64,nsparse)
    VV = zeros(Float64,nsparse)
    @inbounds begin
        for ielement_y in 1:y.nelement
            for ielement_x in 1:x.nelement
                for iy_local in 1:y.ngrid
                    for ix_local in 1:x.ngrid
                        for iyp_local in 1:y.ngrid
                            for ixp_local in 1:x.ngrid
                                ixy_global = ixy_compound_index(x,y,ielement_x,ielement_y,ix_local,iy_local)
                                ixyp_global = ixy_compound_index(x,y,ielement_x,ielement_y,ixp_local,iyp_local)
                                icsc = icsc2D(ixp_local,ix_local,ielement_x,x.ngrid,x.nelement,
                                            iyp_local,iy_local,ielement_y,y.ngrid)
                                # assign indices
                                II[icsc] = ixy_global
                                JJ[icsc] = ixyp_global
                                # assemble matrix values
                                VV[icsc] += weak_form(ixp_local,ix_local,ielement_x,
                                            iyp_local,iy_local,ielement_y)::Float64
                            end
                        end
                    end
                end
            end
        end
    end
    operator_sparse = sparse(II,JJ,VV)
    impose_boundary_condition(boundary_condition_x,boundary_condition_y,
        operator_sparse,x,y,weak_form)
    return operator_sparse
end

include("Calculus.jl")
include("Interpolation.jl")

end # module FiniteElementAssembly
