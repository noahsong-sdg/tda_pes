using Plots

# Set non-interactive backend to avoid hanging
ENV["GKSwstype"] = "100"  # For headless environments
gr()  # Use GR backend which is more stable

#=============================================================================
    PERSISTENCE HOMOLOGY VISUALIZATION MODULE
    
    This module contains all plotting and visualization functions:
    - 1D, 2D, and multi-dimensional plotting
    - Persistence diagrams and barcodes
    - 3D manifold visualization
    - Saddle point marking
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
    
    # Plot saddle points (H0 death events and H1 birth events)
    try
        # H0 death events (saddle points where components merge)
        h0_deaths = [interval.death for interval in result[1] if isfinite(interval.death)]
        if !isempty(h0_deaths)
            # Find approximate coordinates for saddle points (interpolation needed for exact location)
            saddle_energies = h0_deaths
            saddle_coords = [ordered_coords[argmin(abs.(ordered_energies .- energy))] for energy in saddle_energies]
            
            scatter!(p1, saddle_coords, saddle_energies; 
                    color=:red, markersize=6, markershape=:diamond,
                    label="H0 Saddle Points", show=false)
        end
        
        # H1 birth events (saddle points where cycles form)
        h1_births = [interval.birth for interval in result[2]]
        if !isempty(h1_births)
            saddle_energies = h1_births
            saddle_coords = [ordered_coords[argmin(abs.(ordered_energies .- energy))] for energy in saddle_energies]
            
            scatter!(p1, saddle_coords, saddle_energies; 
                    color=:orange, markersize=6, markershape=:dtriangle,
                    label="H1 Saddle Points", show=false)
        end
    catch e
        println("Could not plot saddle points: $e")
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
        combined_plot = plot(p1, p2, layout=(1,2), size=(1300, 750), show=false)
    end
    
    # Save plot
    mkdir_if_not_exists("../figures")
    filename = "../figures/$(output_name).png"
    savefig(combined_plot, filename)
    println("Plot saved to $filename")
    
    return combined_plot
end

function create_2d_plots(coordinates::Matrix{Float64}, rel_energies::Vector{Float64}, 
                         result, analysis_type::String, output_name::String="jl_test_2d")
    """Create plots for 2D data analysis with 3D manifold surface."""
    println("Creating 2D plots with 3D manifold surface...")
    
    # Create 3D manifold surface using interpolation
    println("Fitting 2D manifold to 3D points...")
    
    # Get coordinate ranges
    coord1_vals = sort(unique(coordinates[:, 1]))
    coord2_vals = sort(unique(coordinates[:, 2]))
    
    # Create a finer grid for smooth surface
    n_interp = 50  # Resolution of interpolated surface
    coord1_fine = range(minimum(coord1_vals), maximum(coord1_vals), length=n_interp)
    coord2_fine = range(minimum(coord2_vals), maximum(coord2_vals), length=n_interp)
    
    # For irregular data, use scattered interpolation
    if analysis_type == "1d_projection" || length(coord1_vals) * length(coord2_vals) != length(rel_energies)
        # Use radial basis function interpolation for irregular data
        println("Using RBF interpolation for irregular 2D data")
        
        # Simple distance-weighted interpolation
        energy_surface = zeros(Float64, n_interp, n_interp)
        
        for (i, x) in enumerate(coord1_fine), (j, y) in enumerate(coord2_fine)
            # Distance-weighted interpolation
            weights = 1.0 ./ (sqrt.((coordinates[:, 1] .- x).^2 + (coordinates[:, 2] .- y).^2) .+ 1e-6)
            weights = weights ./ sum(weights)
            energy_surface[i, j] = sum(weights .* rel_energies)
        end
        
        # 2D scatter plot  
        p1 = scatter(coordinates[:, 1], coordinates[:, 2], 
                     marker_z=rel_energies, c=:viridis,
                     xlabel="Dihedral 1 (degrees)", ylabel="Dihedral 2 (degrees)", 
                     title="2D Energy Scatter", 
                     colorbar_title="Energy (Hartree)", 
                     markersize=4, show=false)
    else
        # Regular grid data
        println("Using grid interpolation for regular 2D data")
        
        # Create energy grid for existing data
        coord1_map = Dict(val => i for (i, val) in enumerate(coord1_vals))
        coord2_map = Dict(val => i for (i, val) in enumerate(coord2_vals))
        
        energy_grid = fill(NaN, length(coord1_vals), length(coord2_vals))
        for i in 1:length(rel_energies)
            row = coord1_map[coordinates[i, 1]]
            col = coord2_map[coordinates[i, 2]]
            energy_grid[row, col] = rel_energies[i]
        end
        
        # Interpolate to finer grid (simple bilinear for now)
        energy_surface = zeros(Float64, n_interp, n_interp)
        for (i, x) in enumerate(coord1_fine), (j, y) in enumerate(coord2_fine)
            # Find closest grid points and interpolate
            x_idx = argmin(abs.(coord1_vals .- x))
            y_idx = argmin(abs.(coord2_vals .- y))
            energy_surface[i, j] = energy_grid[x_idx, y_idx]
        end
        
        # 2D heatmap
        p1 = heatmap(coord1_vals, coord2_vals, energy_grid', 
                     xlabel="Dihedral 1 (degrees)", ylabel="Dihedral 2 (degrees)", 
                     title="2D Energy Heatmap", 
                     color=:viridis, show=false)
    end
    
    # Create 3D surface plot (the manifold!)
    p2 = surface(coord1_fine, coord2_fine, energy_surface', 
                 xlabel="Dihedral 1 (degrees)", ylabel="Dihedral 2 (degrees)", 
                 zlabel="Energy (Hartree)", 
                 title="3D Energy Manifold", 
                 color=:viridis, show=false, camera=(45, 60))
    
    # Add original data points to 3D plot
    scatter!(p2, coordinates[:, 1], coordinates[:, 2], rel_energies,
             markersize=3, color=:red, alpha=0.8, label="Data Points", show=false)
    
    # Persistence diagram
    p3 = plot(result, title="Persistence Diagram", show=false)
    
    # Check if we have H1 features to determine layout
    h1_count = length(result[2])
    
    if h1_count > 0
        # Create 4-plot layout for H0 and H1 features with manifold
        println("H1 features detected - creating 4-plot layout with manifold")
        
        # H0 barcode
        p4 = plot(result[1], title="H0 Barcode", plottype=:barcode, show=false)
        
        # Custom layout: Large 3D manifold on top, smaller plots below
        l = @layout([a{0.6w,0.6h} b{0.4w,0.6h}; c{0.5w,0.4h} d{0.5w,0.4h}])
        combined_plot = plot(p2, p1, p3, p4, layout=l, size=(1600, 1200), show=false)
    else
        # Custom layout: Large 3D manifold with smaller 2D and persistence plots
        println("Creating layout with large 3D manifold")
        l = @layout([a{0.6w} [b{0.5h}; c{0.5h}]])
        combined_plot = plot(p2, p1, p3, layout=l, size=(1600, 800), show=false)
    end
    
    # Save plot
    mkdir_if_not_exists("figures")
    filename = "figures/$(output_name).png"
    savefig(combined_plot, filename)
    println("Plot saved to $filename")
    
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
    mkdir_if_not_exists("figures")
    filename = "figures/$(output_name).png"
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
