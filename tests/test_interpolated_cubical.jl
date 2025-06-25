include("ph_core.jl")
include("ph_viz.jl")
include("ph_analysis.jl")

"""
Test script for the new interpolated cubical complex method.
This demonstrates how RBF interpolation + cubical complex works on irregular PES data.
"""

function test_interpolated_cubical_method(filename::String; 
                                        grid_resolution::Union{Vector{Int}, Nothing}=nothing,
                                        rbf_type::String="multiquadric",
                                        epsilon::Float64=1.0)
    """Test the interpolated cubical complex method on PES data."""
    
    println("="^70)
    println("TESTING INTERPOLATED CUBICAL COMPLEX METHOD")
    println("="^70)
    
    # Load data
    coordinates, rel_energies, metadata = load_pes_data(filename)
    if coordinates === nothing
        println("Failed to load data")
        return nothing
    end
    
    n_dims = size(coordinates, 2)
    n_points = length(rel_energies)
    
    println("\nDataset Information:")
    println("  File: $filename")
    println("  Points: $n_points")
    println("  Dimensions: $(n_dims)D")
    println("  Energy range: $(round(minimum(rel_energies), digits=6)) to $(round(maximum(rel_energies), digits=6)) Hartree")
    
    # Test different methods for comparison
    println("\n" * "="^50)
    println("METHOD COMPARISON")
    println("="^50)
    
    # Method 1: Interpolated Cubical (our new method)
    println("\n1. INTERPOLATED CUBICAL COMPLEX:")
    println("-"^35)
    try
        result_interp, energy_grid, grid_axes, grid_points = compute_interpolated_cubical_persistence(
            coordinates, rel_energies, grid_resolution, rbf_type, epsilon)
        
        h0_count = length(result_interp[1])
        h1_count = length(result_interp) > 1 ? length(result_interp[2]) : 0
        h0_persistent = sum([!isfinite(interval.death) for interval in result_interp[1]])
        
        println("✓ Success!")
        println("  H0 features: $h0_count (persistent: $h0_persistent)")
        println("  H1 features: $h1_count")
        
        # Show persistence details
        if h0_count > 0
            births = [interval.birth for interval in result_interp[1]]
            deaths = [isfinite(interval.death) ? interval.death : Inf for interval in result_interp[1]]
            println("  Birth energies: $(round.(births[1:min(3,end)], digits=6))...")
            finite_deaths = [d for d in deaths if isfinite(d)]
            if !isempty(finite_deaths)
                println("  Death energies: $(round.(finite_deaths[1:min(3,end)], digits=6))...")
            end
        end
        
        if h1_count > 0
            h1_births = [interval.birth for interval in result_interp[2]]
            h1_deaths = [isfinite(interval.death) ? interval.death : Inf for interval in result_interp[2]]
            println("  H1 birth energies: $(round.(h1_births[1:min(3,end)], digits=6))...")
        end
        
    catch e
        println("✗ Error: $e")
        result_interp = nothing
    end
    
    # Method 2: Standard Rips for comparison
    println("\n2. STANDARD RIPS COMPLEX (for comparison):")
    println("-"^45)
    try
        result_rips = compute_rips_persistence(coordinates, rel_energies, "Standard Rips")
        
        h0_count = length(result_rips[1])
        h1_count = length(result_rips) > 1 ? length(result_rips[2]) : 0
        
        println("✓ Success!")
        println("  H0 features: $h0_count")
        println("  H1 features: $h1_count")
        
    catch e
        println("✗ Error: $e")
        result_rips = nothing
    end
    
    # Method 3: Direct cubical if data is regular
    println("\n3. DIRECT CUBICAL (if applicable):")
    println("-"^35)
    try
        if n_dims == 1
            coord_diffs = diff(sort(coordinates[:, 1]))
            is_regular = all(abs.(coord_diffs .- coord_diffs[1]) .< 1e-6)
            if is_regular
                ordered_energies, _ = project_to_1d_ordered(coordinates, rel_energies, "first_coordinate")
                result_direct = compute_cubical_persistence(ordered_energies, "Direct 1D cubical")
                
                h0_count = length(result_direct[1])
                h1_count = length(result_direct) > 1 ? length(result_direct[2]) : 0
                
                println("✓ Success! (Regular 1D data)")
                println("  H0 features: $h0_count")
                println("  H1 features: $h1_count")
            else
                println("⚠ Not applicable (irregular 1D data)")
            end
        elseif n_dims == 2
            coord1_vals = sort(unique(coordinates[:, 1]))
            coord2_vals = sort(unique(coordinates[:, 2]))
            if length(coord1_vals) * length(coord2_vals) == length(rel_energies)
                result_direct, grid_type = compute_2d_grid_persistence(coordinates, rel_energies)
                
                h0_count = length(result_direct[1])
                h1_count = length(result_direct) > 1 ? length(result_direct[2]) : 0
                
                println("✓ Success! (Regular 2D grid)")
                println("  H0 features: $h0_count")
                println("  H1 features: $h1_count")
            else
                println("⚠ Not applicable (irregular 2D data)")
            end
        else
            println("⚠ Not applicable ($(n_dims)D data)")
        end
    catch e
        println("✗ Error: $e")
    end
    
    # Analysis and interpretation
    if result_interp !== nothing
        println("\n" * "="^50)
        println("DETAILED ANALYSIS OF INTERPOLATED CUBICAL RESULT")
        println("="^50)
        
        # Detect saddle points using the new method
        saddle_points = detect_saddle_points(result_interp, coordinates, rel_energies)
        
        # Print persistence summary
        print_persistence_summary(result_interp, coordinates, rel_energies)
        
        # Chemical interpretation based on file type
        println("\nCHEMICAL INTERPRETATION:")
        println("-"^25)
        
        if contains(lowercase(filename), "nh3")
            println("• NH3 pyramidal inversion analysis")
            println("• H0 features represent conformational basins")
            println("• H1 features indicate ring-like transition pathways")
            min_idx = argmin(rel_energies)
            min_angle = coordinates[min_idx, 1]
            if abs(min_angle - 247) < 10
                println("• Global minimum near 247° indicates planar transition state preference")
            end
        elseif contains(lowercase(filename), "butane")
            println("• Butane conformational rotation analysis")
            println("• H0 features represent gauche/anti conformational wells")
            min_idx = argmin(rel_energies)
            min_angle = coordinates[min_idx, 1]
            if abs(min_angle - 180) < 20
                println("• Global minimum near $(round(min_angle, digits=1))° indicates anti conformation preference")
            else
                println("• Global minimum near $(round(min_angle, digits=1))° indicates gauche conformation preference")
            end
        else
            println("• General PES topological analysis")
            println("• H0 features represent energy basins/wells")
            println("• H1 features represent cyclic transition pathways")
        end
        
        # Create visualization
        println("\n" * "="^50)
        println("CREATING VISUALIZATION")
        println("="^50)
        
        try
            if n_dims == 1
                create_1d_plots(coordinates, rel_energies, result_interp, "interpolated_cubical_test")
            elseif n_dims == 2
                create_2d_plots(coordinates, rel_energies, result_interp, "interpolated_cubical", "interpolated_cubical_test_2d")
            else
                create_multid_plots(coordinates, rel_energies, result_interp, "interpolated_cubical_test_multid")
            end
            println("✓ Visualization created successfully!")
        catch e
            println("✗ Visualization error: $e")
        end
    end
    
    println("\n" * "="^70)
    println("INTERPOLATED CUBICAL COMPLEX TEST COMPLETE")
    println("="^70)
    
    return result_interp
end

# Test with different RBF types
function test_rbf_variants(filename::String)
    """Test different RBF interpolation variants."""
    
    println("="^70)
    println("TESTING DIFFERENT RBF INTERPOLATION VARIANTS")
    println("="^70)
    
    coordinates, rel_energies, metadata = load_pes_data(filename)
    if coordinates === nothing
        return nothing
    end
    
    rbf_types = ["gaussian", "multiquadric", "inverse_multiquadric"]
    
    for rbf_type in rbf_types
        println("\n" * "="^40)
        println("TESTING RBF TYPE: $(uppercase(rbf_type))")
        println("="^40)
        
        try
            result, energy_grid, grid_axes, grid_points = compute_interpolated_cubical_persistence(
                coordinates, rel_energies, nothing, rbf_type, 1.0)
            
            h0_count = length(result[1])
            h1_count = length(result) > 1 ? length(result[2]) : 0
            
            println("✓ Success with $rbf_type RBF!")
            println("  H0 features: $h0_count")
            println("  H1 features: $h1_count")
            
            # Compare energy preservation
            original_range = maximum(rel_energies) - minimum(rel_energies)
            grid_range = maximum(energy_grid) - minimum(energy_grid)
            println("  Energy range preservation: $(round(grid_range/original_range * 100, digits=1))%")
            
        catch e
            println("✗ Error with $rbf_type: $e")
        end
    end
end

# Example usage
if abspath(PROGRAM_FILE) == @__FILE__
    println("Testing interpolated cubical complex method...")
    println("Usage: julia test_interpolated_cubical.jl")
    println("Or call: test_interpolated_cubical_method(\"your_file.dat\")")
end 
