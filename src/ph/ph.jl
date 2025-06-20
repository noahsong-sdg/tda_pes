# Import the modular components
include("ph_core.jl")
include("ph_viz.jl") 
include("ph_analysis.jl")
include("../../sublevel_advanced.jl")  # Advanced sublevel methods

#=============================================================================
    MAIN PERSISTENCE HOMOLOGY ORCHESTRATOR
    
    This is the main interface that coordinates:
    - Core persistence computation (ph_core.jl)
    - Visualization (ph_viz.jl)
    - Chemical analysis (ph_analysis.jl)
=============================================================================#

function analyze_pes_file(filename::String; output_prefix::String="pes_analysis")
    """Main function to analyze PES file with persistence homology."""
    println("Starting PES Topological Data Analysis")
    println("Input file: $filename")
    println("Output prefix: $output_prefix")
    println("-"^40)
    
    # Load data using core module
    coordinates, rel_energies, metadata = load_pes_data(filename)
    
    if coordinates === nothing
        println("Failed to load data. Exiting.")
        return nothing
    end
    
    # Compute persistence using core module
    result, analysis_type = compute_persistence(coordinates, rel_energies)
    
    # Create visualizations using viz module
    if analysis_type in ["1d", "1d_rips"]
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
    analyze_pes_file(filename, output_prefix=output_prefix)
else
    println("Usage: julia ph.jl <input_file> [output_prefix]")
    println("Example: julia ph.jl data/butane_pes.dat butane_analysis")
    println()
    println("This script performs topological data analysis on potential energy surfaces:")
    println("  - Loads PES data from file")
    println("  - Computes persistence homology")
    println("  - Creates visualizations (1D/2D/3D plots)")
    println("  - Detects saddle points and energy barriers")
    println("  - Provides chemical interpretation")
end 
