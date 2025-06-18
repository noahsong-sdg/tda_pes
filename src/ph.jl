using Ripserer, Images, Plots, DelimitedFiles

# Set non-interactive backend to avoid hanging
ENV["GKSwstype"] = "100"  # For headless environments
gr()  # Use GR backend which is more stable

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
    """Compute persistence for 2D data - try grid first, fallback to ordered 1D."""
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
        println("Irregular 2D data - using coordinate-ordered 1D projection")
        ordered_energies, _ = project_to_1d_ordered(coordinates, rel_energies, "coordinate_sum")
        result = compute_cubical_persistence(ordered_energies, "2D→1D projection")
        return result, "1d_projection"
    end
end

function compute_persistence(coordinates::Matrix{Float64}, rel_energies::Vector{Float64})
    """Main persistence computation dispatcher - handles all dimensionalities."""
    n_dimensions = size(coordinates, 2)
    
    if n_dimensions == 1
        # For 1D data, order by coordinate value  
        ordered_energies, _ = project_to_1d_ordered(coordinates, rel_energies, "first_coordinate")
        result = compute_cubical_persistence(ordered_energies, "1D coordinate-ordered")
        return result, "1d"
        
    elseif n_dimensions == 2
        # For 2D, try grid structure first
        result, grid_type = compute_2d_grid_persistence(coordinates, rel_energies)
        return result, grid_type
        
    else
        # For higher dimensions, project to 1D using coordinate sum
        println("Multi-dimensional data ($(n_dimensions)D) - projecting to 1D")
        ordered_energies, _ = project_to_1d_ordered(coordinates, rel_energies, "coordinate_sum") 
        result = compute_cubical_persistence(ordered_energies, "$(n_dimensions)D→1D projection")
        return result, "multid_projection"
    end
end

#=============================================================================
    PLOTTING FUNCTIONS
=============================================================================#

function create_1d_plots(coordinates::Matrix{Float64}, rel_energies::Vector{Float64}, 
                         result, output_name::String="jl_test_1d")
    """Create plots for 1D data analysis with generators visualization."""
    println("Creating 1D plots with generators...")
    
    # We need to work with the ordered data for proper indexing
    ordered_energies, sorted_indices = project_to_1d_ordered(coordinates, rel_energies, "first_coordinate")
    ordered_coords = coordinates[sorted_indices, 1]
    
    # Energy profile plot
    p1 = plot(ordered_coords, ordered_energies, 
              xlabel="Coordinate", ylabel="Relative Energy (Hartree)", 
              title="1D Energy Profile with Representatives", 
              linewidth=2, color=:black, alpha=0.7,
              show=false, legend=:topright)
    
    # Plot representatives for each H0 interval using official Ripserer API
    colors = [:red, :blue, :green, :orange, :purple, :cyan]
    
    # Get the infinite interval (essential component)
    infinite_interval = nothing
    finite_intervals = []
    
    for interval in result[1]
        if !isfinite(interval)
            infinite_interval = interval
        else
            push!(finite_intervals, interval)
        end
    end
    #=
    # Plot the infinite interval representative (if it exists)
    if infinite_interval !== nothing
        try
            plot!(p1, representative(infinite_interval), ordered_energies; 
                  seriestype=:path, color=colors[1], linewidth=3, alpha=0.8,
                  label="Essential Component", show=false)
        catch e
            println("Could not plot infinite representative: $e")
        end
    end
    
    # Plot finite intervals representatives  
    for (i, interval) in enumerate(finite_intervals)
        color = colors[((i+1) % length(colors)) + 1]  # +1 because infinite used colors[1]
        
        try
            # Plot the representative path
            plot!(p1, interval, ordered_energies; 
                  seriestype=:path, color=color, linewidth=2, alpha=0.8,
                  label="Component $(i+1) Rep", show=false)
        catch e
            println("Could not plot representative for interval $i: $e")
        end
    end=#
    
    # Plot birth points (local minima) using birth_simplex
    try
        birth_indices = [only(Ripserer.birth_simplex(interval)) for interval in result[1]]
        birth_coords = ordered_coords[birth_indices]
        birth_energies = ordered_energies[birth_indices]
        
        scatter!(p1, birth_coords, birth_energies; 
                color=1:length(birth_indices), markersize=8, markershape=:star5,
                label="Birth Points (Minima)", show=false)
    catch e
        println("Could not plot birth points: $e")
    end
    
    # Add energy threshold lines
    births = [interval.birth for interval in result[1]]
    deaths = [interval.death for interval in result[1] if isfinite(interval)]
    thresholds = sort(unique(vcat(births, deaths)))
    
    for (i, threshold) in enumerate(thresholds[1:min(3, length(thresholds))])
        hline!(p1, [threshold]; color=:gray, linestyle=:dash, alpha=0.5, 
               linewidth=1, label="Threshold $(round(threshold, digits=4))", show=false)
    end
    
    # Persistence diagram
    p2 = plot(result, title="Persistence Diagram", 
              markercolor=1:length(result[1]), markeralpha=1, show=false)
    
    # Check if we have H1 features to determine layout
    h1_count = length(result[2])
    
    if h1_count > 0
        # Create 4-plot layout for H0 and H1 features
        println("H1 features detected - creating 4-plot layout")
        
        # H0 barcode
        p3 = plot(result[1], title="H0 Barcode", plottype=:barcode, show=false)
        
        # H1 barcode and generators
        p4 = plot(result[2], title="H1 Barcode", plottype=:barcode, show=false)
        
        # Combine all 4 plots: Energy profile with generators, H0 persistence, H0 barcode, H1 barcode
        combined_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1400, 800), show=false)
    else
        # Standard 2-plot layout
        combined_plot = plot(p1, p2, layout=(1,2), size=(1300, 650), show=false)
    end
    
    # Save plot
    filename = "figures/$(output_name).png"
    savefig(combined_plot, filename)
    println("Plot saved to figures/$filename")
    
    return combined_plot
end

function create_2d_plots(coordinates::Matrix{Float64}, rel_energies::Vector{Float64}, 
                         result, analysis_type::String, output_name::String="jl_test_2d")
    """Create plots for 2D data analysis."""
    println("Creating 2D plots...")
    
    if analysis_type == "2d_grid"
        # For regular 2D grid data
        coord1_vals = sort(unique(coordinates[:, 1]))
        coord2_vals = sort(unique(coordinates[:, 2]))
        
        # Create energy grid for heatmap
        coord1_map = Dict(val => i for (i, val) in enumerate(coord1_vals))
        coord2_map = Dict(val => i for (i, val) in enumerate(coord2_vals))
        
        energy_grid = fill(NaN, length(coord1_vals), length(coord2_vals))
        for i in 1:length(rel_energies)
            row = coord1_map[coordinates[i, 1]]
            col = coord2_map[coordinates[i, 2]]
            energy_grid[row, col] = rel_energies[i]
        end
        
        # Create heatmap
        p1 = heatmap(coord1_vals, coord2_vals, energy_grid', 
                     xlabel="Coordinate 1", ylabel="Coordinate 2", 
                     title="2D Energy Surface", 
                     color=:viridis, show=false)
    else
        # For irregular 2D data - use scatter plot
        p1 = scatter(coordinates[:, 1], coordinates[:, 2], 
                     marker_z=rel_energies, c=:viridis,
                     xlabel="Coordinate 1", ylabel="Coordinate 2", 
                     title="2D Energy Scatter", 
                     colorbar_title="Energy (Hartree)", 
                     markersize=4, show=false)
    end
    
    # Persistence diagram
    p2 = plot(result, title="Persistence Diagram", show=false)
    
    # Check if we have H1 features to determine layout
    h1_count = length(result[2])
    
    if h1_count > 0
        # Create 4-plot layout for H0 and H1 features
        println("H1 features detected - creating 4-plot layout")
        
        # H0 barcode
        p3 = plot(result[1], title="H0 Barcode", plottype=:barcode, show=false)
        
        # H1 barcode and generators
        p4 = plot(result[2], title="H1 Barcode", plottype=:barcode, show=false)
        
        # Combine all 4 plots: Energy surface, H0 persistence, H0 barcode, H1 barcode
        combined_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1400, 800), show=false)
    else
        # Standard 2-plot layout
        combined_plot = plot(p1, p2, layout=(1,2), size=(1200, 500), show=false)
    end
    
    # Save plot
    filename = "figures/$(output_name).png"
    savefig(combined_plot, filename)
    println("Plot saved to figures/$filename")
    
    return combined_plot
end

function create_multid_plots(coordinates::Matrix{Float64}, rel_energies::Vector{Float64}, 
                            result, output_name::String="jl_test_multid")
    """Create plots for multi-dimensional data analysis."""
    println("Creating multi-dimensional plots...")
    
    # Energy vs point index
    p1 = plot(1:length(rel_energies), rel_energies, 
              xlabel="Point Index", ylabel="Relative Energy (Hartree)", 
              title="Multi-D Energy Profile", 
              linewidth=2, marker=:circle, markersize=3,
              show=false)
    
    # Persistence diagram
    p2 = plot(result, title="Persistence Diagram", show=false)
    
    # Check if we have H1 features to determine layout
    h1_count = length(result[2])
    
    if h1_count > 0
        # Create 4-plot layout for H0 and H1 features
        println("H1 features detected - creating 4-plot layout")
        
        # H0 barcode
        p3 = plot(result[1], title="H0 Barcode", plottype=:barcode, show=false)
        
        # H1 barcode and generators
        p4 = plot(result[2], title="H1 Barcode", plottype=:barcode, show=false)
        
        # Combine all 4 plots: Energy profile, H0 persistence, H0 barcode, H1 barcode
        combined_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1400, 800), show=false)
    else
        # Standard 2-plot layout
        combined_plot = plot(p1, p2, layout=(1,2), size=(1200, 500), show=false)
    end
    
    # Save plot
    filename = "figures/$(output_name).png"
    savefig(combined_plot, filename)
    println("Plot saved to figures/$filename")
    
    return combined_plot
end

function print_generators(result)
    """Extract and print generator information from persistence result."""
    println("\n" * "="^50)
    println("GENERATOR INFORMATION")
    println("="^50)
    
    # H0 generators (connected components)
    if length(result[1]) > 0
        println("\nH0 Generators (Connected Components):")
        for (i, interval) in enumerate(result[1])
            birth = round(interval.birth, digits=6)
            death = interval.death == Inf ? "∞" : round(interval.death, digits=6)
            
            # Access generator from meta field
            if hasfield(typeof(interval), :meta) && !isnothing(interval.meta)
                meta = interval.meta
                println("  Component $i (Birth=$birth, Death=$death):")
                println("    Generator type: $(typeof(meta))")
                println("    Generator: $meta")
            else
                println("  Component $i (Birth=$birth, Death=$death): No generator info in meta")
            end
        end
    end
    
    # H1 generators (cycles)  
    if length(result[2]) > 0
        println("\nH1 Generators (Cycles):")
        for (i, interval) in enumerate(result[2])
            birth = round(interval.birth, digits=6)
            death = interval.death == Inf ? "∞" : round(interval.death, digits=6)
            
            # Access generator from meta field
            if hasfield(typeof(interval), :meta) && !isnothing(interval.meta)
                meta = interval.meta
                println("  Cycle $i (Birth=$birth, Death=$death):")
                println("    Generator type: $(typeof(meta))")
                println("    Generator: $meta")
            else
                println("  Cycle $i (Birth=$birth, Death=$death): No generator info in meta")
            end
        end
    else
        println("\nNo H1 features (cycles) detected.")
    end
    
    println("="^50)
end

function print_persistence_summary(result, coordinates::Matrix{Float64}, rel_energies::Vector{Float64})
    """Print summary of persistence homology results."""
    println("\n" * "="^50)
    println("PERSISTENCE HOMOLOGY SUMMARY")
    println("="^50)
    
    # Count features by dimension
    h0_count = length(result[1])  # 0-dimensional features
    h1_count = length(result[2])  # 1-dimensional features  
    
    println("Dataset Info:")
    println("  Points: $(length(rel_energies))")
    println("  Dimensions: $(size(coordinates, 2))")
    println("  Energy range: $(round(minimum(rel_energies), digits=6)) - $(round(maximum(rel_energies), digits=6)) Hartree")
    
    println("\nTopological Features:")
    println("  H0 (Connected Components): $h0_count")
    println("  H1 (Cycles/Loops): $h1_count")
    
    if h0_count > 0
        println("\nH0 Features (Connected Components):")
        for (i, interval) in enumerate(result[1])
            birth = interval.birth
            death = interval.death == Inf ? "∞" : round(interval.death, digits=6)
            println("  Component $i: Birth=$(round(birth, digits=6)), Death=$death")
        end
    end
    
    if h1_count > 0
        println("\nH1 Features (Cycles):")
        for (i, interval) in enumerate(result[2])
            birth = interval.birth
            death = interval.death == Inf ? "∞" : round(interval.death, digits=6)
            persistence = interval.death == Inf ? "∞" : round(interval.death - birth, digits=6)
            println("  Cycle $i: Birth=$(round(birth, digits=6)), Death=$death, Persistence=$persistence")
        end
    end
    
    println("="^50)
    
    # Print generator information
    print_generators(result)
end

#=============================================================================
    MAIN EXECUTION FUNCTION
=============================================================================#

function analyze_pes_file(filename::String; output_prefix::String="pes_analysis")
    """Main function to analyze PES file with persistence homology."""
    println("Starting PES Topological Data Analysis")
    println("Input file: $filename")
    println("Output prefix: $output_prefix")
    println("-"^40)
    
    # Load data
    coordinates, rel_energies, metadata = load_pes_data(filename)
    
    if coordinates === nothing
        println("Failed to load data. Exiting.")
        return nothing
    end
    
    # Compute persistence
    result, analysis_type = compute_persistence(coordinates, rel_energies)
    
    # Create visualizations based on analysis type
    if analysis_type == "1d"
        create_1d_plots(coordinates, rel_energies, result, output_prefix)
    elseif analysis_type in ["2d_grid", "1d_projection"]
        create_2d_plots(coordinates, rel_energies, result, analysis_type, output_prefix)
    else
        create_multid_plots(coordinates, rel_energies, result, output_prefix)
    end
    
    # Print summary
    print_persistence_summary(result, coordinates, rel_energies)
    
    println("\nAnalysis complete!")
    return result, coordinates, rel_energies, metadata
end

#=============================================================================
    SCRIPT EXECUTION
=============================================================================#

# Main execution when script is run directly
if length(ARGS) >= 1
    filename = ARGS[1]
    output_prefix = length(ARGS) >= 2 ? ARGS[2] : "pes_analysis"
    analyze_pes_file(filename, output_prefix=output_prefix)
else
    println("Usage: julia ph.jl <input_file> [output_prefix]")
    println("Example: julia ph.jl data/butane_pes.dat butane_analysis")
end 
