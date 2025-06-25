using Plots

# Set non-interactive backend to avoid hanging
ENV["GKSwstype"] = "100"  # For headless environments
gr()  # Use GR backend which is more stable

#=============================================================================
    STREAMLINED PERSISTENCE HOMOLOGY VISUALIZATION MODULE
    
    Consolidated visualization functions with eliminated redundancies:
    - Common persistence diagram and barcode plotting
    - Unified layout and saving logic
    - Specialized plots for 1D, 2D, and multi-dimensional data
=============================================================================#

#=============================================================================
    CORE HELPER FUNCTIONS (ELIMINATE REDUNDANCIES)
=============================================================================#

function create_persistence_diagram(result, rel_energies::Vector{Float64})
    """Create standard persistence diagram with H0/H1 features."""
    p = plot(title="Persistence Diagram", xlabel="Birth", ylabel="Death", 
             legend=:bottomright, show=false)
    
    # Plot H0 features (connected components) in red
    if !isempty(result[1])
        h0_births = [interval.birth for interval in result[1]]
        h0_deaths = [isfinite(interval.death) ? interval.death : maximum(rel_energies) * 1.1 for interval in result[1]]
        scatter!(p, h0_births, h0_deaths, color=:red, markersize=4, alpha=0.7, label="H0 (Components)")
    end
    
    # Plot H1 features (cycles) in blue
    h1_count = length(result) > 1 ? length(result[2]) : 0
    if h1_count > 0
        h1_births = [interval.birth for interval in result[2]]
        h1_deaths = [isfinite(interval.death) ? interval.death : maximum(rel_energies) * 1.1 for interval in result[2]]
        scatter!(p, h1_births, h1_deaths, color=:blue, markersize=4, alpha=0.7, label="H1 (Cycles)")
    end
    
    # Add diagonal line y=x
    max_val = maximum([maximum(rel_energies), 
                      !isempty(result[1]) ? maximum([interval.birth for interval in result[1]]) : 0])
    plot!(p, [0, max_val], [0, max_val], color=:gray, linestyle=:dash, alpha=0.5, label="y=x")
    
    return p
end

function create_barcode_plots(result)
    """Create H0 and H1 barcode plots manually to avoid plotting recipe issues."""
    h1_count = length(result) > 1 ? length(result[2]) : 0
    
    # Manual H0 barcode
    p_h0 = plot(title="H0 Barcode (Components)", 
                xlabel="Energy Threshold", ylabel="Feature Index", 
                legend=false, show=false)
    
    for (i, interval) in enumerate(result[1])
        birth = interval.birth
        death = isfinite(interval.death) ? interval.death : interval.birth * 1.2 + 0.1
        plot!(p_h0, [birth, death], [i, i], linewidth=3, color=:red, alpha=0.8)
    end
    
    if h1_count > 0
        # Manual H1 barcode
        p_h1 = plot(title="H1 Barcode (Cycles)", 
                    xlabel="Energy Threshold", ylabel="Feature Index", 
                    legend=false, show=false)
        
        for (i, interval) in enumerate(result[2])
            birth = interval.birth
            death = isfinite(interval.death) ? interval.death : interval.birth * 1.2 + 0.1
            plot!(p_h1, [birth, death], [i, i], linewidth=3, color=:blue, alpha=0.8)
        end
        return p_h0, p_h1, h1_count
    else
        return p_h0, nothing, h1_count
    end
end

function combine_and_save_plots(plots::Vector, layout_type::String, output_name::String, 
                               output_dir::String="../../figures")
    """Combine plots with appropriate layout and save to file."""
    
    if layout_type == "1d"
        if length(plots) == 4
            # 4-plot: Energy profile, Persistence diagram, H0 barcode, H1 barcode
            combined_plot = plot(plots[1], plots[2], plots[3], plots[4], 
                               layout=(2,2), size=(1400, 800), show=false)
            println("4-plot layout: Energy profile + persistence diagram + H0/H1 barcodes")
        else
            # 3-plot: Energy profile, Persistence diagram, H0 barcode
            combined_plot = plot(plots[1], plots[2], plots[3], 
                               layout=(2,2), size=(1300, 750), show=false)
            println("3-plot layout: Energy profile + persistence diagram + H0 barcode")
        end
        
    elseif layout_type == "2d"
        if length(plots) == 5
            # 5-plot: Large 3D manifold, 2D view, persistence diagram, H0 barcode, H1 barcode
            l = @layout([a{0.5w,0.6h} b{0.5w,0.6h}; c{0.33w,0.4h} d{0.33w,0.4h} e{0.33w,0.4h}])
            combined_plot = plot(plots[1], plots[2], plots[3], plots[4], plots[5], 
                               layout=l, size=(1800, 1200), show=false)
            println("5-plot layout: 3D manifold + 2D view + persistence diagram + H0/H1 barcodes")
        else
            # 4-plot: Large 3D manifold, 2D view, persistence diagram, H0 barcode
            l = @layout([a{0.5w,0.6h} b{0.5w,0.6h}; c{0.5w,0.4h} d{0.5w,0.4h}])
            combined_plot = plot(plots[1], plots[2], plots[3], plots[4], 
                               layout=l, size=(1600, 1000), show=false)
            println("4-plot layout: 3D manifold + 2D view + persistence diagram + H0 barcode")
        end
        
    else # multid
        if length(plots) == 4
            # 4-plot: Energy profile, Persistence diagram, H0 barcode, H1 barcode
            combined_plot = plot(plots[1], plots[2], plots[3], plots[4], 
                               layout=(2,2), size=(1400, 800), show=false)
            println("4-plot layout: Energy profile + persistence diagram + H0/H1 barcodes")
        else
            # 3-plot: Energy profile, Persistence diagram, H0 barcode
            combined_plot = plot(plots[1], plots[2], plots[3], 
                               layout=(2,2), size=(1200, 600), show=false)
            println("3-plot layout: Energy profile + persistence diagram + H0 barcode")
        end
    end
    
    # Save plot to main figures directory
    mkdir_if_not_exists(output_dir)
    filename = "$(output_dir)/$(output_name).png"
    savefig(combined_plot, filename)
    println("Plot saved to $filename")
    
    return combined_plot
end

function mkdir_if_not_exists(dir_name::String)
    """Create directory if it doesn't exist."""
    if !isdir(dir_name)
        mkdir(dir_name)
    end
end

#=============================================================================
    SPECIALIZED PLOTTING FUNCTIONS
=============================================================================#

function create_1d_plots(coordinates::Matrix{Float64}, rel_energies::Vector{Float64}, 
                         result, output_name::String="jl_test_1d")
    """Create plots for 1D data analysis with representatives visualization."""
    println("Creating 1D plots with representatives...")
    
    # We need to work with the ordered data for proper indexing
    ordered_energies, sorted_indices = project_to_1d_ordered(coordinates, rel_energies, "first_coordinate")
    ordered_coords = coordinates[sorted_indices, 1]
    
    # Energy profile plot
    p1 = plot(ordered_coords, ordered_energies, 
              xlabel="Coordinate", ylabel="Relative Energy (Hartree)", 
              title="1D Energy Profile with Representatives", 
              linewidth=2, color=:black, alpha=0.7,
              show=false, legend=:topright)
    
    # Plot representatives and critical points
    colors = [:red, :blue, :green, :orange, :purple, :cyan]
    
    # Plot critical points (birth points = local minima)
    try
        min_indices = [only(birth_simplex(interval)) for interval in result[1]]
        min_coords = ordered_coords[min_indices]
        min_energies = ordered_energies[min_indices]
        
        scatter!(p1, min_coords, min_energies; 
                color=1:length(min_indices), markersize=8, markershape=:star5,
                label="Critical Points (Minima)", show=false)
        
        println("  Found $(length(min_indices)) critical points")
    catch e
        # Fallback: mark global minimum
        min_idx = argmin(ordered_energies)
        scatter!(p1, [ordered_coords[min_idx]], [ordered_energies[min_idx]]; 
                color=:red, markersize=8, markershape=:star5,
                label="Global Minimum", show=false)
    end
    
    # Simplified representatives plotting - just mark birth/death points
    try
        for (i, interval) in enumerate(result[1])
            birth_energy = interval.birth
            death_energy = isfinite(interval.death) ? interval.death : maximum(ordered_energies)
            
            # Find closest coordinates for birth/death energies
            birth_idx = argmin(abs.(ordered_energies .- birth_energy))
            birth_coord = ordered_coords[birth_idx]
            
            # Mark birth point
            scatter!(p1, [birth_coord], [birth_energy]; 
                    color=colors[((i-1) % length(colors)) + 1], 
                    markersize=6, markershape=:circle, alpha=0.8,
                    label=i==1 ? "Birth Points" : "", show=false)
            
            # Mark death point if finite
            if isfinite(interval.death)
                death_idx = argmin(abs.(ordered_energies .- death_energy))
                death_coord = ordered_coords[death_idx]
                scatter!(p1, [death_coord], [death_energy]; 
                        color=colors[((i-1) % length(colors)) + 1], 
                        markersize=4, markershape=:x, alpha=0.6,
                        label=i==1 ? "Death Points" : "", show=false)
            end
        end
    catch e
        println("Could not plot birth/death points: $e")
    end
    
    # Plot H1 cycles if they exist
    h1_count = length(result) > 1 ? length(result[2]) : 0
    if h1_count > 0
        try
            h1_births = [interval.birth for interval in result[2]]
            h1_coords = [ordered_coords[argmin(abs.(ordered_energies .- energy))] for energy in h1_births]
            
            scatter!(p1, h1_coords, h1_births; 
                    color=:orange, markersize=6, markershape=:dtriangle,
                    label="H1 Birth Points (Cycles)", show=false)
        catch e
            println("Could not plot H1 birth points: $e")
        end
    end
    
    # Create persistence diagram and barcodes using helper functions
    p2 = create_persistence_diagram(result, rel_energies)
    p3, p4, h1_count = create_barcode_plots(result)
    
    # Combine and save
    plots = h1_count > 0 ? [p1, p2, p3, p4] : [p1, p2, p3]
    return combine_and_save_plots(plots, "1d", output_name)
end

function create_2d_plots(coordinates::Matrix{Float64}, rel_energies::Vector{Float64}, 
                         result, analysis_type::String, output_name::String="jl_test_2d")
    """Create plots for 2D data analysis with 3D manifold surface."""
    println("Creating 2D plots with 3D manifold surface...")
    
    # Get coordinate ranges and create interpolation grid
    coord1_vals = sort(unique(coordinates[:, 1]))
    coord2_vals = sort(unique(coordinates[:, 2]))
    n_interp = 50
    coord1_fine = range(minimum(coord1_vals), maximum(coord1_vals), length=n_interp)
    coord2_fine = range(minimum(coord2_vals), maximum(coord2_vals), length=n_interp)
    
    # Create energy surface and 2D view
    if analysis_type == "1d_projection" || length(coord1_vals) * length(coord2_vals) != length(rel_energies)
        # Irregular data: use distance-weighted interpolation
        energy_surface = zeros(Float64, n_interp, n_interp)
        for (i, x) in enumerate(coord1_fine), (j, y) in enumerate(coord2_fine)
            weights = 1.0 ./ (sqrt.((coordinates[:, 1] .- x).^2 + (coordinates[:, 2] .- y).^2) .+ 1e-6)
            weights = weights ./ sum(weights)
            energy_surface[i, j] = sum(weights .* rel_energies)
        end
        
        # 2D scatter plot  
        p2 = scatter(coordinates[:, 1], coordinates[:, 2], 
                     marker_z=rel_energies, c=:viridis,
                     xlabel="Dihedral 1 (degrees)", ylabel="Dihedral 2 (degrees)", 
                     title="2D Energy Scatter", colorbar_title="Energy (Hartree)", 
                     markersize=4, show=false)
    else
        # Regular grid data
        coord1_map = Dict(val => i for (i, val) in enumerate(coord1_vals))
        coord2_map = Dict(val => i for (i, val) in enumerate(coord2_vals))
        
        energy_grid = fill(NaN, length(coord1_vals), length(coord2_vals))
        for i in 1:length(rel_energies)
            row = coord1_map[coordinates[i, 1]]
            col = coord2_map[coordinates[i, 2]]
            energy_grid[row, col] = rel_energies[i]
        end
        
        # Simple interpolation to finer grid
        energy_surface = zeros(Float64, n_interp, n_interp)
        for (i, x) in enumerate(coord1_fine), (j, y) in enumerate(coord2_fine)
            x_idx = argmin(abs.(coord1_vals .- x))
            y_idx = argmin(abs.(coord2_vals .- y))
            energy_surface[i, j] = energy_grid[x_idx, y_idx]
        end
        
        # 2D heatmap
        p2 = heatmap(coord1_vals, coord2_vals, energy_grid', 
                     xlabel="Dihedral 1 (degrees)", ylabel="Dihedral 2 (degrees)", 
                     title="2D Energy Heatmap", color=:viridis, show=false)
    end
    
    # Create 3D surface plot
    p1 = surface(coord1_fine, coord2_fine, energy_surface', 
                 xlabel="Dihedral 1 (degrees)", ylabel="Dihedral 2 (degrees)", 
                 zlabel="Energy (Hartree)", title="3D Energy Manifold", 
                 color=:viridis, show=false, camera=(45, 60))
    
    # Add original data points to 3D plot
    scatter!(p1, coordinates[:, 1], coordinates[:, 2], rel_energies,
             markersize=3, color=:red, alpha=0.8, label="Data Points", show=false)
    
    # Create persistence diagram and barcodes using helper functions
    p3 = create_persistence_diagram(result, rel_energies)
    p4, p5, h1_count = create_barcode_plots(result)
    
    # Combine and save
    plots = h1_count > 0 ? [p1, p2, p3, p4, p5] : [p1, p2, p3, p4]
    return combine_and_save_plots(plots, "2d", output_name)
end

function create_multid_plots(coordinates::Matrix{Float64}, rel_energies::Vector{Float64}, 
                            result, output_name::String="jl_test_multid")
    """Create plots for multi-dimensional data analysis."""
    println("Creating multi-dimensional plots...")
    
    # Energy vs point index
    p1 = plot(1:length(rel_energies), rel_energies, 
              xlabel="Point Index", ylabel="Relative Energy (Hartree)", 
              title="Multi-D Energy Profile", 
              linewidth=2, marker=:circle, markersize=3, show=false)
    
    # Create persistence diagram and barcodes using helper functions
    p2 = create_persistence_diagram(result, rel_energies)
    p3, p4, h1_count = create_barcode_plots(result)
    
    # Combine and save
    plots = h1_count > 0 ? [p1, p2, p3, p4] : [p1, p2, p3]
    return combine_and_save_plots(plots, "multid", output_name)
end
