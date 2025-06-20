using Ripserer, DelimitedFiles

#=============================================================================
    CORE PERSISTENCE HOMOLOGY COMPUTATION MODULE
    
    This module contains only the essential persistence computation functions:
    - Data loading from PES files
    - Persistence homology computation
    - Basic data projection methods
=============================================================================#

#=============================================================================
    DATA LOADING FUNCTIONS
=============================================================================#

function load_pes_data(filename::String; coordinate_names::Union{Vector{String}, Nothing} = nothing, 
                       coordinate_units::Union{Vector{String}, Nothing} = nothing)
    """Load PES data from file with fast comment filtering."""
    if !isfile(filename)
        println("Error: File '$filename' not found.")
        return nothing, nothing, nothing
    end
    
    try
        # Fast file reading - read all lines and filter comments manually
        lines = readlines(filename)
        data_lines = filter(line -> !startswith(strip(line), "#") && !isempty(strip(line)), lines)
        
        # Parse numeric data more efficiently
        data = zeros(Float64, length(data_lines), 0)
        if !isempty(data_lines)
            # Parse first line to determine number of columns
            first_vals = parse.(Float64, split(strip(data_lines[1])))
            n_cols = length(first_vals)
            data = zeros(Float64, length(data_lines), n_cols)
            data[1, :] = first_vals
            
            # Parse remaining lines
            for (i, line) in enumerate(data_lines[2:end])
                data[i+1, :] = parse.(Float64, split(strip(line)))
            end
        end
    catch e
        println("Error loading data from $filename: $e")
        return nothing, nothing, nothing
    end
    
    if ndims(data) == 1
        data = reshape(data, 1, :)
    end
    
    n_points, n_columns = size(data)
    @assert n_columns >= 2 "Data must have at least 2 columns (coordinate + energy)"
    
    coordinates = data[:, 1:end-1]
    energies = data[:, end]
    n_dimensions = size(coordinates, 2)
    
    rel_energies = energies .- minimum(energies)
    
    if coordinate_names === nothing
        coordinate_names = ["coord_$(i)" for i in 1:n_dimensions]
    end
    if coordinate_units === nothing
        coordinate_units = ["unknown" for _ in 1:n_dimensions]
    end
    
    metadata = Dict(
        "n_points" => n_points,
        "n_dimensions" => n_dimensions,
        "coordinate_names" => coordinate_names,
        "coordinate_units" => coordinate_units,
        "filename" => filename
    )
    
    println("Loaded $(n_dimensions)D PES data: $n_points points")
    println("Energy range: $(round(minimum(rel_energies), digits=6)) to $(round(maximum(rel_energies), digits=6)) Hartree")
    
    return coordinates, rel_energies, metadata
end

#=============================================================================
    PERSISTENCE COMPUTATION FUNCTIONS
=============================================================================#

function compute_cubical_persistence(energies::Union{Vector{Float64}, Matrix{Float64}}, description::String="")
    """Core function to compute persistence homology using Cubical complex."""
    result = ripserer(Cubical(energies), reps=true)
    return result
end

function compute_rips_persistence(coordinates::Matrix{Float64}, rel_energies::Vector{Float64}, description::String="")
    """Compute persistence homology using Rips complex for irregular point clouds."""
    println("Computing Rips complex persistence for irregular data...")
    
    # For Rips complex, we need distance matrix between points
    # Add energy as an extra dimension (scaled appropriately)
    energy_scale = 0.1  # Scale factor for energy relative to coordinates
    scaled_energies = rel_energies * energy_scale
    
    # Combine coordinates with scaled energy
    augmented_coords = hcat(coordinates, scaled_energies)
    
    # Compute Rips complex
    result = ripserer(augmented_coords, dim_max=2, reps=true)
    
    println("  $(description)")
    println("  Points: $(size(coordinates, 1))")
    println("  Dimensions: $(size(coordinates, 2)) + energy")
    
    return result
end

function compute_alpha_persistence(coordinates::Matrix{Float64}, rel_energies::Vector{Float64}, description::String="")
    """Compute persistence homology using Alpha complex for irregular point clouds."""
    println("Computing Alpha complex persistence for irregular data...")
    
    # Alpha complexes work better for geometric data
    # But Ripserer may not have direct Alpha support - use Rips as fallback
    return compute_rips_persistence(coordinates, rel_energies, description)
end

function compute_sublevel_filtration(coordinates::Matrix{Float64}, rel_energies::Vector{Float64}, description::String="")
    """Compute sublevel filtration by energy threshold."""
    println("Computing sublevel filtration for irregular data...")
    
    # Sort energies to create filtration sequence
    sorted_indices = sortperm(rel_energies)
    sorted_energies = rel_energies[sorted_indices]
    sorted_coords = coordinates[sorted_indices, :]
    
    # Build point cloud with energy-based filtration
    # This creates a sequence of point clouds as energy threshold increases
    
    # For now, use Rips complex as a proxy
    # In full implementation, you'd build proper sublevel sets
    result = compute_rips_persistence(sorted_coords, sorted_energies, description)
    
    return result
end

function project_to_1d_ordered(coordinates::Matrix{Float64}, rel_energies::Vector{Float64}, method::String="coordinate_sum")
    """Project multi-dimensional data to 1D with proper ordering."""
    if method == "coordinate_sum"
        # Order by sum of coordinates (good for general multi-D data)
        sorted_indices = sortperm(vec(sum(coordinates, dims=2)))
    elseif method == "first_coordinate"
        # Order by first coordinate (good for 1D data)
        sorted_indices = sortperm(coordinates[:, 1])
    else
        error("Unknown projection method: $method")
    end
    
    return rel_energies[sorted_indices], sorted_indices
end

function compute_2d_grid_persistence(coordinates::Matrix{Float64}, rel_energies::Vector{Float64})
    """Compute persistence for 2D data - try grid first, fallback to point cloud methods."""
    println("Analyzing 2D data structure...")
    
    # Detect grid structure
    coord1_vals = sort(unique(coordinates[:, 1]))  
    coord2_vals = sort(unique(coordinates[:, 2]))
    
    if length(coord1_vals) * length(coord2_vals) == length(rel_energies)
        println("Regular 2D grid detected: $(length(coord1_vals)) × $(length(coord2_vals))")
        
        # Create mapping from coordinates to grid indices
        coord1_map = Dict(val => i for (i, val) in enumerate(coord1_vals))
        coord2_map = Dict(val => i for (i, val) in enumerate(coord2_vals))
        
        # Build 2D energy grid
        energy_grid = zeros(Float64, length(coord1_vals), length(coord2_vals))
        for i in 1:length(rel_energies)
            row = coord1_map[coordinates[i, 1]]
            col = coord2_map[coordinates[i, 2]]
            energy_grid[row, col] = rel_energies[i]
        end
        
        result = compute_cubical_persistence(energy_grid, "2D grid")
        return result, "2d_grid"
    else
        println("Irregular 2D data - using Rips complex for point cloud")
        result = compute_rips_persistence(coordinates, rel_energies, "2D irregular point cloud")
        return result, "2d_rips"
    end
end

function compute_persistence(coordinates::Matrix{Float64}, rel_energies::Vector{Float64})
    """Main persistence computation dispatcher - handles regular grids and irregular point clouds."""
    n_dimensions = size(coordinates, 2)
    n_points = size(coordinates, 1)
    
    println("Data analysis:")
    println("  Points: $n_points")
    println("  Dimensions: $n_dimensions")
    
    if n_dimensions == 1
        # For 1D data, check if regularly spaced
        coord_diffs = diff(sort(coordinates[:, 1]))
        is_regular = all(abs.(coord_diffs .- coord_diffs[1]) .< 1e-6)
        
        if is_regular
            println("Regular 1D spacing detected - using cubical complex")
            ordered_energies, _ = project_to_1d_ordered(coordinates, rel_energies, "first_coordinate")
            result = compute_cubical_persistence(ordered_energies, "1D coordinate-ordered")
            return result, "1d"
        else
            println("Irregular 1D spacing - using Rips complex")
            result = compute_rips_persistence(coordinates, rel_energies, "1D irregular")
            return result, "1d_rips"
        end
        
    elseif n_dimensions == 2
        # For 2D, try grid structure first, then point cloud methods
        result, grid_type = compute_2d_grid_persistence(coordinates, rel_energies)
        return result, grid_type
        
    else
        # For higher dimensions, always use point cloud methods for irregular data
        println("Multi-dimensional data ($(n_dimensions)D) - using Rips complex")
        result = compute_rips_persistence(coordinates, rel_energies, "$(n_dimensions)D point cloud")
        return result, "multid_rips"
    end
end 
