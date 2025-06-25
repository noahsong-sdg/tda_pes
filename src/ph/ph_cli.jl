# Import the modular components
include("ph_core.jl")      # Core persistence computation (includes ph_rbf.jl)
include("ph_viz.jl")       # Visualization functions
include("ph_analysis.jl")  # Chemical analysis and interpretation
include("ph_adv.jl")       # Advanced persistence methods

#=============================================================================
    MAIN PERSISTENCE HOMOLOGY ORCHESTRATOR
    
    This is the main interface that coordinates:
    - Core persistence computation (ph_core.jl + ph_rbf.jl)
    - Visualization (ph_viz.jl)
    - Chemical analysis (ph_analysis.jl)
    - Advanced methods (ph_adv.jl)
=============================================================================#

function analyze_pes_file(filename::String; output_prefix::String="pes_analysis", 
                         method::String="auto")
    """Main function to analyze PES file with persistence homology.
    
    Methods:
    - 'auto': Automatic method selection (default)
    - 'interpolated_cubical': RBF interpolation + cubical complex (recommended for irregular data)
    - 'cubical': Direct cubical complex (only for regular grids)
    - 'rips': Rips complex (geometric proximity)
    """
    println("Starting PES Topological Data Analysis")
    println("Input file: $filename")
    println("Output prefix: $output_prefix")
    println("Method: $method")
    println("-"^40)
    
    # Load data using core module
    coordinates, rel_energies, metadata = load_pes_data(filename)
    
    if coordinates === nothing
        println("Failed to load data. Exiting.")
        return nothing
    end
    
    # Compute persistence using core module
    result, analysis_type = compute_persistence(coordinates, rel_energies, method)
    
    # Create visualizations using viz module
    if analysis_type in ["1d", "1d_rips", "cubical"]
        create_1d_plots(coordinates, rel_energies, result, output_prefix)
    elseif analysis_type in ["2d_grid", "2d_rips", "1d_projection"]
        create_2d_plots(coordinates, rel_energies, result, analysis_type, output_prefix)
    else
        create_multid_plots(coordinates, rel_energies, result, output_prefix)
    end
    
    # Print summaries using analysis module
    print_persistence_summary(result, coordinates, rel_energies)
    
    # Detect and analyze saddle points using analysis module
    saddle_points = detect_saddle_points(result, coordinates, rel_energies)
    
    println("\nAnalysis complete!")
    return result, coordinates, rel_energies, metadata, saddle_points
end

#=============================================================================
    COMMAND LINE INTERFACE
=============================================================================#

# Main execution when script is run directly
if length(ARGS) >= 1
    filename = ARGS[1]
    output_prefix = length(ARGS) >= 2 ? ARGS[2] : "pes_analysis"
    method = length(ARGS) >= 3 ? ARGS[3] : "auto"
    analyze_pes_file(filename, output_prefix=output_prefix, method=method)
else
    println("Usage: julia ph_cli.jl <input_file> [output_prefix] [method]")
    println("Example: julia ph_cli.jl data/butane_pes.dat butane_analysis interpolated_cubical")
    println()
    println("Methods:")
    println("  auto               - Automatic method selection (default)")
    println("  interpolated_cubical - RBF interpolation + cubical complex")
    println("  cubical            - Direct cubical complex (regular grids only)")
    println("  rips               - Rips complex with energy augmentation")
    println()
    println("This script performs topological data analysis on potential energy surfaces:")
    println("  - Loads PES data from file")
    println("  - Computes persistence homology")
    println("  - Creates visualizations (1D/2D/3D plots)")
    println("  - Detects saddle points and energy barriers")
    println("  - Provides chemical interpretation")
end 
