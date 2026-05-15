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
# linear search algorithm
# function get_ielement(xval::Float64,
#             xcoord::FiniteElementCoordinate)
#     @inbounds begin
#         xebs = xcoord.element_boundaries
#         nelement = xcoord.nelement
#         tolerance = 1.0e-14
#         ielement = -1
#         # find the element
#         for j in 1:nelement
#             # check for internal points
#             if (xval - xebs[j])*(xebs[j+1] - xval) > tolerance
#                 ielement = j
#                 break
#             # check for boundary points
#             elseif (abs(xval-xebs[j]) < 100*tolerance) || (abs(xval-xebs[j+1]) < 100*tolerance && j == nelement)
#                 ielement = j
#                 break
#             end
#         end
#         if ielement < 1
#             error("xval=$xval is not within the coordinate $(xcoord.name)")
#         end
#         return ielement
#     end
# end
function get_ielement(xval::Float64,
            xcoord::FiniteElementCoordinate)
    # search by bisection
    @inbounds begin
        xebs = xcoord.element_boundaries
        nelement = xcoord.nelement
        tolerance = eps(Float64)
        i = 1 # lower limit
        k = nelement+1 # upper limit
        # check extreme limits before main bisection loop
        xi = xebs[i] - xval
        xk = xebs[k] - xval
        if abs(xi) < tolerance
            # xval is xebs[i], so in element i
            return i
        elseif abs(xk) < tolerance
            # xval is xebs[k], so in element k - 1
            return k - 1 # ielement range is 1:nelement, but we have nelement+1 boundaries
        elseif xk*xi > tolerance # positive if root not in [xebs[i],xebs[k]]
            error("xval=$xval is not within the coordinate $(xcoord.name)")
        end
        j = i + div(k - i, 2) # midpoint
        ielement = i
        # find the element by bisection
        # note that because the element range i 1:nelement, but we have
        # nelement+1 element boundaries, we default to returning the
        # lower limit index in the loops below
        while k - i > 1
            xj = xebs[j] - xval
            if abs(xj) < tolerance
                # ielement = j - div(j , nelement + 1)
                # j is always < nelement + 1, so no need for additional div(j, nelement + 1) term
                ielement = j
                break
            elseif xj*(xval - xebs[k]) > tolerance
                # root between j and k
                i = j # set lower limit index to j
                j = i + div(k - i, 2) # set new midpoint < k
            else
                # root between i and j
                k = j # set upper limit to j
                j = i + div(k - i, 2) # set new midpoint =< i
            end
            ielement = i
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