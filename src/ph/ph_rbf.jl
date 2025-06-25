using Statistics

#=============================================================================
    RBF INTERPOLATION MODULE
    
    Radial Basis Function interpolation for irregular PES data:
    - RBF basis functions (multiquadric)
    - Grid generation for 1D-4D data
    - Weight computation and evaluation
    - Complete interpolation pipeline
=============================================================================#

#=============================================================================
    RBF BASIS FUNCTIONS
=============================================================================#

function rbf_multiquadric(r::Float64, epsilon::Float64=1.0)
    """Multiquadric radial basis function."""
    return sqrt(1 + (epsilon * r)^2)
end

#=============================================================================
    RBF WEIGHT COMPUTATION AND EVALUATION
=============================================================================#

function compute_rbf_weights(coordinates::Matrix{Float64}, values::Vector{Float64}, epsilon::Float64=1.0)
    """Compute RBF interpolation weights for given data points."""
    n_points = size(coordinates, 1)
    
    # Build RBF matrix
    A = zeros(Float64, n_points, n_points)
    for i in 1:n_points
        for j in 1:n_points
            if i == j
                A[i, j] = rbf_multiquadric(0.0, epsilon)
            else
                r = sqrt(sum((coordinates[i, :] - coordinates[j, :]).^2))
                A[i, j] = rbf_multiquadric(r, epsilon)
            end
        end
    end
    
    # Solve for weights: A * weights = values
    weights = A \ values
    return weights
end

function evaluate_rbf_interpolation(query_points::Matrix{Float64}, 
                                  data_coordinates::Matrix{Float64}, 
                                  weights::Vector{Float64}, epsilon::Float64=1.0)
    """Evaluate RBF interpolation at query points."""
    n_query = size(query_points, 1)
    n_data = size(data_coordinates, 1)
    
    interpolated_values = zeros(Float64, n_query)
    
    for i in 1:n_query
        value = 0.0
        for j in 1:n_data
            r = sqrt(sum((query_points[i, :] - data_coordinates[j, :]).^2))
            value += weights[j] * rbf_multiquadric(r, epsilon)
        end
        interpolated_values[i] = value
    end
    
    return interpolated_values
end

#=============================================================================
    GRID GENERATION
=============================================================================#

function create_regular_grid(coordinates::Matrix{Float64}, grid_resolution::Vector{Int})
    """Create regular grid covering the coordinate space."""
    n_dims = size(coordinates, 2)
    @assert length(grid_resolution) == n_dims "Grid resolution must match coordinate dimensions"
    
    # Get coordinate bounds with small padding
    coord_mins = minimum(coordinates, dims=1)[:]
    coord_maxs = maximum(coordinates, dims=1)[:]
    padding = 0.05 * (coord_maxs - coord_mins)  # 5% padding
    coord_mins = coord_mins .- padding
    coord_maxs = coord_maxs .+ padding
    
    # Create grid points for each dimension
    grid_axes = []
    for dim in 1:n_dims
        push!(grid_axes, range(coord_mins[dim], coord_maxs[dim], length=grid_resolution[dim]))
    end
    
    # Generate all grid point combinations
    if n_dims == 1
        grid_points = reshape(collect(grid_axes[1]), :, 1)
    elseif n_dims == 2
        x_grid = repeat(collect(grid_axes[1]), 1, grid_resolution[2])
        y_grid = repeat(collect(grid_axes[2])', grid_resolution[1], 1)
        grid_points = hcat(vec(x_grid), vec(y_grid))
    elseif n_dims == 3
        n_total = prod(grid_resolution)
        grid_points = zeros(Float64, n_total, 3)
        idx = 1
        for k in grid_axes[3], j in grid_axes[2], i in grid_axes[1]
            grid_points[idx, :] = [i, j, k]
            idx += 1
        end
    elseif n_dims == 4
        n_total = prod(grid_resolution)
        grid_points = zeros(Float64, n_total, 4)
        idx = 1
        for l in grid_axes[4], k in grid_axes[3], j in grid_axes[2], i in grid_axes[1]
            grid_points[idx, :] = [i, j, k, l]
            idx += 1
        end
    else
        error("Grid creation not implemented for $(n_dims)D data")
    end
    
    return grid_points, grid_axes, tuple(grid_resolution...)
end

function auto_tune_epsilon(coordinates::Matrix{Float64})
    """Automatically tune epsilon parameter based on typical inter-point distances."""
    n_points = size(coordinates, 1)
    distances = []
    sample_size = min(100, n_points)
    sample_indices = rand(1:n_points, sample_size)
    
    for i in sample_indices
        for j in sample_indices
            if i != j
                r = sqrt(sum((coordinates[i, :] - coordinates[j, :]).^2))
                push!(distances, r)
            end
        end
    end
    
    epsilon = 1.0 / median(distances)  # Typical heuristic
    return epsilon
end

#=============================================================================
    COMPLETE INTERPOLATION PIPELINE
=============================================================================#

function interpolate_to_regular_grid(coordinates::Matrix{Float64}, rel_energies::Vector{Float64},
                                   grid_resolution::Union{Vector{Int}, Nothing}=nothing)
    """Interpolate irregular PES data to regular grid using RBF interpolation."""
    n_dims = size(coordinates, 2)
    n_points = length(rel_energies)
    
    println("Interpolating $(n_points) irregular points to regular $(n_dims)D grid...")
    
    # Default grid resolutions based on dimensionality and data density
    if grid_resolution === nothing
        if n_dims == 1
            grid_resolution = [min(200, max(50, n_points))]
        elseif n_dims == 2
            base_res = min(100, max(20, Int(round(sqrt(n_points)))))
            grid_resolution = [base_res, base_res]
        elseif n_dims == 3
            base_res = min(50, max(15, Int(round(n_points^(1/3)))))
            grid_resolution = [base_res, base_res, base_res]
        elseif n_dims == 4
            base_res = min(25, max(10, Int(round(n_points^(1/4)))))
            grid_resolution = [base_res, base_res, base_res, base_res]
        else
            error("Automatic grid resolution not supported for $(n_dims)D")
        end
    end
    
    println("  Grid resolution: $(grid_resolution)")
    
    # Auto-tune epsilon based on typical distances
    epsilon = auto_tune_epsilon(coordinates)
    println("  Auto-tuned epsilon: $(round(epsilon, digits=4))")
    
    # Create regular grid
    grid_points, grid_axes, grid_shape = create_regular_grid(coordinates, grid_resolution)
    
    # Compute RBF weights and interpolate
    println("  Computing RBF interpolation...")
    weights = compute_rbf_weights(coordinates, rel_energies, epsilon)
    interpolated_energies = evaluate_rbf_interpolation(grid_points, coordinates, weights, epsilon)
    
    # Reshape to grid structure
    energy_grid = reshape(interpolated_energies, grid_shape)
    
    println("  Interpolation complete!")
    println("  Grid energy range: $(round(minimum(energy_grid), digits=6)) to $(round(maximum(energy_grid), digits=6))")
    
    return energy_grid, grid_axes, grid_points
end 
