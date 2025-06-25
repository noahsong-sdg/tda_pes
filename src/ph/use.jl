# Import modular components
include("ph_core.jl")      # Core persistence computation (includes ph_rbf.jl)
include("ph_viz.jl")       # Visualization functions
include("ph_analysis.jl")  # Chemical analysis

# Load data (adjust path since we're in src/ph/ directory)
coordinates, rel_energies, metadata = load_pes_data("../../data/nh3_pes.dat")

# Check if data loaded successfully
if coordinates === nothing
    println("Failed to load data. Please check the file path.")
    exit(1)
end

# Compute persistence using interpolated cubical method
result, analysis_type = compute_persistence(coordinates, rel_energies, "interpolated_cubical")

# Create visualization
create_1d_plots(coordinates, rel_energies, result, "interpolated_cubical_test")

# Print analysis summary
print_persistence_summary(result, coordinates, rel_energies)
