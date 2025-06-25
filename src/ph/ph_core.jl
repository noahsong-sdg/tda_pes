using Ripserer, DelimitedFiles, Statistics
include("ph_rbf.jl")

#=============================================================================
    CORE PERSISTENCE HOMOLOGY COMPUTATION MODULE
    
    Streamlined persistence computation for potential energy surfaces:
    - Data loading from PES files
    - Two main methods: Cubical (direct/interpolated) and Rips complex
    - Automatic method selection based on data structure
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
    UTILITY FUNCTIONS
=============================================================================#

function project_to_1d_ordered(coordinates::Matrix{Float64}, rel_energies::Vector{Float64}, method::String="first_coordinate")
    """Project multi-dimensional data to 1D with proper ordering."""
    if method == "first_coordinate"
        # Order by first coordinate (good for 1D data)
        sorted_indices = sortperm(coordinates[:, 1])
    else
        error("Unknown projection method: $method")
    end
    
    return rel_energies[sorted_indices], sorted_indices
end

function is_regular_grid(coordinates::Matrix{Float64})
    """Check if coordinates form a regular grid structure."""
    n_dims = size(coordinates, 2)
    
    if n_dims == 1
        # For 1D: check if regularly spaced
        coord_diffs = diff(sort(coordinates[:, 1]))
        return all(abs.(coord_diffs .- coord_diffs[1]) .< 1e-6)
    elseif n_dims == 2
        # For 2D: check if forms complete grid
        coord1_vals = sort(unique(coordinates[:, 1]))
        coord2_vals = sort(unique(coordinates[:, 2]))
        return length(coord1_vals) * length(coord2_vals) == size(coordinates, 1)
    else
        # For higher dimensions: assume irregular (could be extended)
        return false
    end
end

function create_grid_from_coordinates(coordinates::Matrix{Float64}, rel_energies::Vector{Float64})
    """Convert regular coordinate data to grid format for cubical complex."""
    n_dims = size(coordinates, 2)
    
    if n_dims == 1
        # 1D: sort by coordinate
        ordered_energies, _ = project_to_1d_ordered(coordinates, rel_energies)
        return ordered_energies
    elseif n_dims == 2
        # 2D: create energy grid
        coord1_vals = sort(unique(coordinates[:, 1]))
        coord2_vals = sort(unique(coordinates[:, 2]))
        
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
        
        return energy_grid
    else
        error("Grid creation only supported for 1D and 2D regular data")
    end
end

#=============================================================================
    CORE PERSISTENCE COMPUTATION FUNCTIONS
=============================================================================#

function compute_cubical_persistence(energies::Union{Vector{Float64}, Matrix{Float64}}, description::String="")
    """Core function to compute persistence homology using Cubical complex."""
    println("Computing cubical persistence with representatives...")
    result = ripserer(Cubical(energies), reps=true)
    println("  $(description)")
    return result
end

function compute_rips_persistence(coordinates::Matrix{Float64}, rel_energies::Vector{Float64}, description::String="")
    """Compute persistence homology using Rips complex with energy augmentation."""
    println("Computing Rips complex persistence...")
    
    # Add energy as an extra dimension (scaled appropriately)
    energy_scale = 0.1  # Scale factor for energy relative to coordinates
    scaled_energies = rel_energies * energy_scale
    
    # Combine coordinates with scaled energy
    augmented_coords = hcat(coordinates, scaled_energies)
    
    # Compute Rips complex
    result = ripserer(augmented_coords, dim_max=2, reps=true)
    
    println("  $(description)")
    println("  Points: $(size(coordinates, 1)), Dimensions: $(size(coordinates, 2)) + energy")
    
    return result
end

function compute_interpolated_cubical_persistence(coordinates::Matrix{Float64}, rel_energies::Vector{Float64},
                                                grid_resolution::Union{Vector{Int}, Nothing}=nothing)
    """Compute cubical persistence on interpolated regular grid from irregular data."""
    
    println("Computing interpolated cubical persistence...")
    
    # Interpolate to regular grid using RBF module
    energy_grid, grid_axes, grid_points = interpolate_to_regular_grid(coordinates, rel_energies, grid_resolution)
    
    # Compute cubical persistence on the interpolated grid
    result = compute_cubical_persistence(energy_grid, "Interpolated $(size(coordinates,2))D grid")
    
    return result, energy_grid, grid_axes, grid_points
end

#=============================================================================
    MAIN PERSISTENCE COMPUTATION DISPATCHER
=============================================================================#

function compute_persistence(coordinates::Matrix{Float64}, rel_energies::Vector{Float64}, 
                           method::String="auto")
    """Main persistence computation dispatcher.
    
    Methods:
    - 'auto': Automatic method selection (recommended)
    - 'interpolated_cubical': RBF interpolation + cubical complex 
    - 'cubical': Direct cubical complex (only for regular grids)
    - 'rips': Rips complex with energy augmentation
    """
    n_dimensions = size(coordinates, 2)
    n_points = size(coordinates, 1)
    
    println("Data analysis:")
    println("  Points: $n_points, Dimensions: $n_dimensions, Method: $method")
    
    if method == "interpolated_cubical"
        # Use RBF interpolation + cubical complex for any data
        result, energy_grid, grid_axes, grid_points = compute_interpolated_cubical_persistence(coordinates, rel_energies)
        return result, "interpolated_cubical"
        
    elseif method == "cubical"
        # Direct cubical - only for regular grids
        if !is_regular_grid(coordinates)
            error("Data is not regularly gridded - cannot use direct cubical method. Use 'interpolated_cubical' instead.")
        end
        
        energy_grid = create_grid_from_coordinates(coordinates, rel_energies)
        result = compute_cubical_persistence(energy_grid, "$(n_dimensions)D regular grid")
        return result, "cubical"
        
    elseif method == "rips"
        # Rips complex with energy augmentation
        result = compute_rips_persistence(coordinates, rel_energies, "$(n_dimensions)D Rips complex")
        return result, "rips"
        
    else  # method == "auto"
        # Automatic method selection
        if is_regular_grid(coordinates)
            println("Regular grid detected - using direct cubical complex")
            energy_grid = create_grid_from_coordinates(coordinates, rel_energies)
            result = compute_cubical_persistence(energy_grid, "$(n_dimensions)D regular grid")
            return result, "cubical"
        else
            println("Irregular data detected - using interpolated cubical complex")
            result, energy_grid, grid_axes, grid_points = compute_interpolated_cubical_persistence(coordinates, rel_energies)
            return result, "interpolated_cubical"
        end
    end
end 
