export value_in_coordinate_domain, 
    get_ielement,
    interpolate_1D

"""
Function to determine whether or not `xval` is within
the domain covered by the coordinate grid `xcoord`.
"""
function value_in_coordinate_domain(xval::Float64, xcoord::FiniteElementCoordinate)
    xebs = xcoord.element_boundaries
    tolerance = 1.0e-14
    # internal point
    if (xval - xebs[1])*(xebs[end] - xval) > tolerance
        in_domain = true
    # boundary points
    elseif (abs(xval-xebs[1]) < 100*tolerance) || (abs(xval-xebs[end]) < 100*tolerance)
        in_domain = true
    else 
        in_domain = false
    end
    return in_domain
end

"""
Function to find the element in which the value xval sits in the
assembled grid of the coordinate xcoord.
"""
function get_ielement(xval::Float64,
            xcoord::FiniteElementCoordinate)
    @inbounds begin
        xebs = xcoord.element_boundaries
        nelement = xcoord.nelement
        tolerance = 1.0e-14
        ielement = -1
        # find the element
        for j in 1:nelement
            # check for internal points
            if (xval - xebs[j])*(xebs[j+1] - xval) > tolerance
                ielement = j
                break
            # check for boundary points
            elseif (abs(xval-xebs[j]) < 100*tolerance) || (abs(xval-xebs[j+1]) < 100*tolerance && j == nelement)
                ielement = j
                break
            end
        end
        if ielement < 1
            error("xval=$xval is not within the coordinate $(xcoord.name)")
        end
        return ielement
    end
end

"""
Function for calculating the interpolated value at `x=xval`
for the input data `xfunction` in the coordinate `xcoord`.
"""
function interpolate_1D(xval::Float64, xfunction::AbstractArray{Float64,1}, xcoord::FiniteElementCoordinate)
    @boundscheck length(xfunction) == xcoord.n || throw(BoundsError(xfunction))
    ielement = get_ielement(xval, xcoord)
    # get data for interpolation
    lpoly_data = xcoord.lpoly_data[ielement]
    x_igrid_full = @view xcoord.igrid_full[:,ielement]
    xfunction_local = @view xfunction[x_igrid_full[1]:x_igrid_full[end]]
    result = 0.0
    for ix in 1:xcoord.ngrid
        # index for referencing pdf_in on orginal grid
        ix_lpoly_data = lpoly_data.lpoly_data[ix]
        # interpolating polynomial value at ix for interpolation
        poly = lagrange_poly(ix_lpoly_data,xval)
        result += poly*xfunction_local[ix]
    end
    return result
end