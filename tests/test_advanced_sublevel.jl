include("../src/ph/ph_core.jl")
include("../src/sublevel_advanced.jl")

"""
Test script for advanced sublevel filtration methods.
This compares different approaches for irregular conformer data.
"""

function test_advanced_methods(filename::String)
    println("Testing advanced sublevel methods on: $filename")
    println("="^60)
    
    # Load data using existing function
    coordinates, rel_energies, metadata = load_pes_data(filename)
    
    if coordinates === nothing
        println("Failed to load data")
        return
    end
    
    n_points, n_dims = size(coordinates)
    println("Data: $n_points points, $n_dims dimensions")
    println("Energy range: $(minimum(rel_energies):.6f) to $(maximum(rel_energies):.6f) Hartree")
    
    # Test all methods
    println("\nComparing sublevel filtration methods:")
    println("-"^40)
    
    # Method comparison
    results = test_sublevel_methods(coordinates, rel_energies)
    
    println("\nMethod Comparison Summary:")
    println("1. Standard Rips: Basic geometric complex")
    println("2. Energy-augmented: Coordinates + energy as extra dimension") 
    println("3. Adaptive Rips: Energy-weighted neighborhoods")
    
    return results
end

# Command line interface
if length(ARGS) >= 1
    filename = ARGS[1]
    test_advanced_methods(filename)
else
    println("Usage: julia test_advanced_sublevel.jl <data_file>")
    println("Example: julia test_advanced_sublevel.jl data/butane_v3_pes.dat")
end 
