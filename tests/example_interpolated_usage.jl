include("ph.jl")

"""
EXAMPLE: Using the New Interpolated Cubical Complex Method

This demonstrates the proper way to analyze irregular PES data using
RBF interpolation + cubical complex persistence homology.
"""

function example_interpolated_analysis()
    """Example showing how to use the interpolated cubical method."""
    
    println("="^70)
    println("EXAMPLE: INTERPOLATED CUBICAL COMPLEX FOR PES ANALYSIS")
    println("="^70)
    
    # Example 1: Using the main interface with interpolated method
    println("\n1. USING MAIN INTERFACE WITH INTERPOLATED METHOD:")
    println("-"^50)
    
    # This will now default to interpolated cubical for irregular data
    if isfile("data/nh3_pes.dat")
        result = analyze_pes_file("data/nh3_pes.dat", 
                                 output_prefix="nh3_interpolated",
                                 method="interpolated_cubical")
        println("✓ NH3 analysis complete with interpolated cubical method")
    else
        println("⚠ NH3 data file not found")
    end
    
    # Example 2: Manual usage of core functions
    println("\n2. MANUAL USAGE OF CORE FUNCTIONS:")
    println("-"^35)
    
    if isfile("data/nh3_pes.dat")
        # Load data manually
        coordinates, rel_energies, metadata = load_pes_data("data/nh3_pes.dat")
        
        # Compute interpolated cubical persistence with custom parameters
        result, energy_grid, grid_axes, grid_points = compute_interpolated_cubical_persistence(
            coordinates, rel_energies,
            [100],  # Custom grid resolution for 1D
            "multiquadric",  # RBF type
            1.0  # Auto-tune epsilon
        )
        
        println("✓ Manual interpolated cubical computation complete")
        println("  Grid shape: $(size(energy_grid))")
        println("  H0 features: $(length(result[1]))")
        println("  H1 features: $(length(result) > 1 ? length(result[2]) : 0)")
        
        # Analyze results
        saddle_points = detect_saddle_points(result, coordinates, rel_energies)
        print_persistence_summary(result, coordinates, rel_energies)
    end
    
    # Example 3: Method comparison
    println("\n3. COMPARING METHODS:")
    println("-"^20)
    
    if isfile("data/nh3_pes.dat")
        println("Running method comparison...")
        # This will now include the interpolated cubical method
        include("compare_methods.jl")  # Runs the updated comparison
    end
    
    println("\n" * "="^70)
    println("KEY ADVANTAGES OF INTERPOLATED CUBICAL METHOD:")
    println("="^70)
    println("✓ Works on ANY irregular PES data (1D, 2D, 3D, 4D)")
    println("✓ Computes TRUE sublevel set filtration {x : f(x) ≤ t}")
    println("✓ Preserves energy landscape topology")
    println("✓ More stable than Rips-based methods")
    println("✓ Chemically meaningful H0/H1 features")
    println("✓ Automatic parameter tuning")
    
    println("\nUSAGE RECOMMENDATIONS:")
    println("-"^20)
    println("• Use 'interpolated_cubical' for irregular/sparse data")
    println("• Use 'auto' to let the system choose the best method")
    println("• Adjust grid_resolution for speed vs accuracy trade-off")
    println("• Try different RBF types for different energy landscapes")
    
    println("\n" * "="^70)
end

# Test different scenarios
function test_different_scenarios()
    """Test the method on different types of PES data."""
    
    test_files = [
        ("data/nh3_pes.dat", "NH3 pyramidal inversion"),
        ("data/butane_pes.dat", "Butane conformational rotation"),
        ("data/butane_v3.dat", "Butane variant")
    ]
    
    for (filename, description) in test_files
        if isfile(filename)
            println("\n" * "="^50)
            println("TESTING: $description")
            println("File: $filename")
            println("="^50)
            
            try
                # Test with automatic method selection (should choose interpolated for irregular data)
                result = analyze_pes_file(filename, 
                                        output_prefix="test_$(splitext(basename(filename))[1])",
                                        method="auto")
                println("✓ Analysis successful!")
            catch e
                println("✗ Error: $e")
            end
        else
            println("⚠ File not found: $filename")
        end
    end
end

# Run examples if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    example_interpolated_analysis()
    test_different_scenarios()
end 
