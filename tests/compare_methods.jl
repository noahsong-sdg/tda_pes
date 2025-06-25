# Compare Cubical vs Rips Complex Methods on NH3 PES Data
include("src/ph/ph_core.jl")
include("src/sublevel_advanced.jl")
using Plots

function compare_methods(filename::String)
    println("="^60)
    println("COMPARING CUBICAL vs RIPS COMPLEX METHODS")
    println("Data file: $filename")
    println("="^60)
    
    # Load the data
    coordinates, rel_energies, metadata = load_pes_data(filename)
    
    if coordinates === nothing
        println("Failed to load data")
        return
    end
    
    println("\nData overview:")
    println("  Dimensions: $(metadata["n_dimensions"])")
    println("  Points: $(metadata["n_points"])")
    println("  Energy range: $(round(minimum(rel_energies), digits=6)) to $(round(maximum(rel_energies), digits=6)) Hartree")
    
    # Method 1: Cubical Complex (for regular 1D data)
    println("\n" * "="^40)
    println("METHOD 1: CUBICAL COMPLEX")
    println("="^40)
    
    try
        # Order the data by coordinate for cubical complex
        ordered_energies, sorted_indices = project_to_1d_ordered(coordinates, rel_energies, "first_coordinate")
        result_cubical = compute_cubical_persistence(ordered_energies, "1D cubical complex")
        
        println("Cubical Complex Results:")
        for (dim, intervals) in enumerate(result_cubical)
            dim_idx = dim - 1  # Convert to 0-indexed
            println("  H$dim_idx (dim $dim_idx features): $(length(intervals))")
            if length(intervals) > 0 && length(intervals) <= 10
                for (i, interval) in enumerate(intervals)
                    birth = interval.birth
                    death = interval.death == Inf ? "∞" : round(interval.death, digits=6)
                    persistence = interval.death == Inf ? "∞" : round(interval.death - interval.birth, digits=6)
                    println("    Feature $i: birth=$(round(birth, digits=6)), death=$death, persistence=$persistence")
                end
            elseif length(intervals) > 10
                println("    ($(length(intervals)) features - showing first 5)")
                for (i, interval) in enumerate(intervals[1:5])
                    birth = interval.birth
                    death = interval.death == Inf ? "∞" : round(interval.death, digits=6)
                    persistence = interval.death == Inf ? "∞" : round(interval.death - interval.birth, digits=6)
                    println("    Feature $i: birth=$(round(birth, digits=6)), death=$death, persistence=$persistence")
                end
            end
        end
        
        # Plot persistence diagram for Cubical
        plot_persistence_diagram(result_cubical, "Cubical Complex")
    catch e
        println("Error with cubical complex: $e")
    end
    
    # Method 2: Rips Complex (for point cloud data)
    println("\n" * "="^40)
    println("METHOD 2: RIPS COMPLEX")
    println("="^40)
    
    try
        # Fix Rips complex for 1D data by computing proper distance matrix
        n_points = size(coordinates, 1)
        println("Computing Rips complex for $n_points points...")
        
        if size(coordinates, 2) == 1
            # For 1D data, create distance matrix manually but limit max distance
            distances = zeros(n_points, n_points)
            println("Building distance matrix...")
            for i in 1:n_points
                for j in i+1:n_points
                    dist = abs(coordinates[i, 1] - coordinates[j, 1])
                    distances[i, j] = distances[j, i] = dist
                end
            end
            println("Distance matrix built. Computing Rips with max_dim=1...")
            # Use lower max dimension to avoid hanging
            result_rips = ripserer(distances, dim_max=1, reps=false)
            println("Rips computation complete.")
        else
            result_rips = compute_rips_persistence(coordinates, rel_energies, "Multi-D Rips complex")
        end
        
        println("Rips Complex Results:")
        for (dim, intervals) in enumerate(result_rips)
            dim_idx = dim - 1  # Convert to 0-indexed
            println("  H$dim_idx (dim $dim_idx features): $(length(intervals))")
            if length(intervals) > 0 && length(intervals) <= 10
                for (i, interval) in enumerate(intervals)
                    birth = interval.birth
                    death = interval.death == Inf ? "∞" : round(interval.death, digits=6)
                    persistence = interval.death == Inf ? "∞" : round(interval.death - interval.birth, digits=6)
                    println("    Feature $i: birth=$(round(birth, digits=6)), death=$death, persistence=$persistence")
                end
            elseif length(intervals) > 10
                println("    ($(length(intervals)) features - showing first 5)")
                for (i, interval) in enumerate(intervals[1:5])
                    birth = interval.birth
                    death = interval.death == Inf ? "∞" : round(interval.death, digits=6)
                    persistence = interval.death == Inf ? "∞" : round(interval.death - interval.birth, digits=6)
                    println("    Feature $i: birth=$(round(birth, digits=6)), death=$death, persistence=$persistence")
                end
            end
        end
        
        # Plot persistence diagram for Rips
        plot_persistence_diagram(result_rips, "Rips Complex")
        
    catch e
        println("Error with Rips complex: $e")
        result_rips = nothing
    end
    
    # Method 3: Energy-Augmented Rips (Sublevel-like)
    println("\n" * "="^40)
    println("METHOD 3: ENERGY-AUGMENTED RIPS")
    println("="^40)
    
    try
        println("Computing Rips with energy as extra dimension...")
        energy_scale = 0.1  # Scale factor for energy relative to coordinates
        augmented_coords = hcat(coordinates, rel_energies * energy_scale)
        result_energy_rips = ripserer(augmented_coords, dim_max=2, reps=true)
        
        println("Energy-Augmented Rips Results:")
        for (dim, intervals) in enumerate(result_energy_rips)
            dim_idx = dim - 1
            println("  H$dim_idx: $(length(intervals)) features")
            if length(intervals) > 0 && length(intervals) <= 10
                for (i, interval) in enumerate(intervals)
                    birth = interval.birth
                    death = interval.death == Inf ? "∞" : round(interval.death, digits=6)
                    persistence = interval.death == Inf ? "∞" : round(interval.death - interval.birth, digits=6)
                    println("    Feature $i: birth=$(round(birth, digits=6)), death=$death, persistence=$persistence")
                end
            elseif length(intervals) > 10
                println("    ($(length(intervals)) features - showing first 5)")
                for (i, interval) in enumerate(intervals[1:5])
                    birth = interval.birth
                    death = interval.death == Inf ? "∞" : round(interval.death, digits=6)
                    persistence = interval.death == Inf ? "∞" : round(interval.death - interval.birth, digits=6)
                    println("    Feature $i: birth=$(round(birth, digits=6)), death=$death, persistence=$persistence")
                end
            end
        end
        
        # Plot persistence diagram
        plot_persistence_diagram(result_energy_rips, "Energy-Augmented Rips")
        
    catch e
        println("Error with energy-augmented Rips: $e")
    end
    
    # Method 4: NEW Interpolated Cubical Complex
    println("\n" * "="^40)
    println("METHOD 4: INTERPOLATED CUBICAL COMPLEX")
    println("="^40)
    
    try
        println("Computing RBF interpolation + cubical complex...")
        result_interpolated, energy_grid, grid_axes, grid_points = compute_interpolated_cubical_persistence(coordinates, rel_energies)
        
        println("Interpolated Cubical Complex Results:")
        for (dim, intervals) in enumerate(result_interpolated)
            dim_idx = dim - 1
            println("  H$dim_idx (dim $dim_idx features): $(length(intervals))")
            if length(intervals) > 0 && length(intervals) <= 10
                for (i, interval) in enumerate(intervals)
                    birth = interval.birth
                    death = interval.death == Inf ? "∞" : round(interval.death, digits=6)
                    persistence = interval.death == Inf ? "∞" : round(interval.death - interval.birth, digits=6)
                    println("    Feature $i: birth=$(round(birth, digits=6)), death=$death, persistence=$persistence")
                end
            elseif length(intervals) > 10
                println("    ($(length(intervals)) features - showing first 5)")
                for (i, interval) in enumerate(intervals[1:5])
                    birth = interval.birth
                    death = interval.death == Inf ? "∞" : round(interval.death, digits=6)
                    persistence = interval.death == Inf ? "∞" : round(interval.death - interval.birth, digits=6)
                    println("    Feature $i: birth=$(round(birth, digits=6)), death=$death, persistence=$persistence")
                end
            end
        end
        
        # Plot persistence diagram
        plot_persistence_diagram(result_interpolated, "Interpolated Cubical Complex")
        
    catch e
        println("Error with interpolated cubical complex: $e")
    end
    
    println("\n" * "="^60)
    println("COMPARISON SUMMARY")
    println("="^60)
    println("NH3 shows a characteristic double-well potential due to inversion.")
    println("The minimum at ~247° corresponds to the planar configuration.")
    println("\nMethods compared:")
    println("  1. Cubical Complex: Best for regular grid data")
    println("  2. Standard Rips: General point cloud method (geometric proximity)")
    println("  3. Energy-Augmented Rips: Coordinates + scaled energy dimension")
    println("  4. Interpolated Cubical: RBF interpolation + cubical (RECOMMENDED for irregular data)")
    println("\nExpected features:")
    println("  - H0: Connected components (energy basins)")
    println("  - H1: Loops/cycles (transition pathways)")
    println("Differences reflect how each method handles the energy landscape structure.")
end

function plot_persistence_diagram(result, title::String)
    """Plot persistence diagram for a given persistence result."""
    try
        # Extract birth and death times for different dimensions
        colors = [:red, :blue, :green, :orange, :purple]
        p = plot(title=title, xlabel="Birth", ylabel="Death", legend=:bottomright)
        
        for (dim, intervals) in enumerate(result)
            if length(intervals) > 0
                dim_idx = dim - 1
                births = [interval.birth for interval in intervals]
                deaths = [interval.death == Inf ? maximum(births) * 1.2 : interval.death for interval in intervals]
                
                # Plot points
                scatter!(p, births, deaths, 
                        label="H$dim_idx ($(length(intervals)) features)", 
                        color=colors[min(dim, length(colors))],
                        markersize=4)
            end
        end
        
        # Add diagonal line y=x
        max_val = maximum([xlims(p)[2], ylims(p)[2]])
        plot!(p, [0, max_val], [0, max_val], 
              linestyle=:dash, color=:black, alpha=0.5, label="y=x")
        
        # Save plot
        filename = "figures/persistence_$(replace(lowercase(title), " " => "_")).png"
        savefig(p, filename)
        println("  Persistence diagram saved to: $filename")
        
    catch e
        println("  Error plotting persistence diagram for $title: $e")
    end
end

# Run the comparison
compare_methods("data/nh3_pes.dat") 
