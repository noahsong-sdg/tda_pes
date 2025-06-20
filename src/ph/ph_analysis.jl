#=============================================================================
    PERSISTENCE HOMOLOGY ANALYSIS MODULE
    
    This module contains chemical interpretation and analysis functions:
    - Saddle point detection from persistence data
    - Energy barrier analysis
    - Statistical summaries
    - Chemical coordinate analysis
=============================================================================#

function detect_saddle_points(result, coordinates::Matrix{Float64}, rel_energies::Vector{Float64})
    """Detect saddle points from persistence homology results."""
    println("\n" * "="^50)
    println("SADDLE POINT ANALYSIS")
    println("="^50)
    
    saddle_points = []
    
    # Method 1: H0 death events (where components merge)
    h0_deaths = []
    for (i, interval) in enumerate(result[1])
        if isfinite(interval.death)
            # This is where two minima connect through a saddle point
            saddle_energy = interval.death
            birth_energy = interval.birth
            barrier_height = saddle_energy - birth_energy
            
            push!(h0_deaths, (
                type = "H0_death",
                saddle_energy = saddle_energy,
                birth_energy = birth_energy, 
                barrier_height = barrier_height,
                component_id = i
            ))
        end
    end
    
    # Method 2: H1 birth events (where cycles form around barriers)
    h1_births = []
    for (i, interval) in enumerate(result[2])
        # H1 birth corresponds to formation of a cycle around a barrier
        saddle_energy = interval.birth
        persistence = isfinite(interval.death) ? interval.death - interval.birth : Inf
        
        push!(h1_births, (
            type = "H1_birth",
            saddle_energy = saddle_energy,
            persistence = persistence,
            cycle_id = i
        ))
    end
    
    # Combine and sort by energy
    all_saddles = vcat(h0_deaths, h1_births)
    sort!(all_saddles, by = x -> x.saddle_energy)
    
    println("Detected Saddle Points:")
    println("  H0 deaths (basin merging): $(length(h0_deaths))")
    println("  H1 births (cycle formation): $(length(h1_births))")
    println("  Total potential saddle points: $(length(all_saddles))")
    
    if !isempty(all_saddles)
        println("\nSaddle Point Details:")
        for (i, saddle) in enumerate(all_saddles)
            energy_kcal = saddle.saddle_energy * 627.509  # Convert to kcal/mol
            
            if saddle.type == "H0_death"
                barrier_kcal = saddle.barrier_height * 627.509
                println("  Saddle $i ($(saddle.type)):")
                println("    Energy: $(round(saddle.saddle_energy, digits=6)) Hartree ($(round(energy_kcal, digits=3)) kcal/mol)")
                println("    Barrier height: $(round(saddle.barrier_height, digits=6)) Hartree ($(round(barrier_kcal, digits=3)) kcal/mol)")
                println("    Component ID: $(saddle.component_id)")
            else  # H1_birth
                pers_kcal = isfinite(saddle.persistence) ? saddle.persistence * 627.509 : Inf
                println("  Saddle $i ($(saddle.type)):")
                println("    Energy: $(round(saddle.saddle_energy, digits=6)) Hartree ($(round(energy_kcal, digits=3)) kcal/mol)")
                println("    Persistence: $(isfinite(saddle.persistence) ? round(saddle.persistence, digits=6) : "∞") Hartree ($(isfinite(pers_kcal) ? round(pers_kcal, digits=3) : "∞") kcal/mol)")
                println("    Cycle ID: $(saddle.cycle_id)")
            end
        end
    end
    
    # Statistical analysis
    if !isempty(all_saddles)
        saddle_energies = [s.saddle_energy for s in all_saddles]
        min_saddle = minimum(saddle_energies)
        max_saddle = maximum(saddle_energies)
        mean_saddle = sum(saddle_energies) / length(saddle_energies)
        
        println("\nSaddle Point Statistics:")
        println("  Lowest saddle energy: $(round(min_saddle, digits=6)) Hartree ($(round(min_saddle * 627.509, digits=3)) kcal/mol)")
        println("  Highest saddle energy: $(round(max_saddle, digits=6)) Hartree ($(round(max_saddle * 627.509, digits=3)) kcal/mol)")
        println("  Average saddle energy: $(round(mean_saddle, digits=6)) Hartree ($(round(mean_saddle * 627.509, digits=3)) kcal/mol)")
        
        # Barrier heights from H0 deaths only
        if !isempty(h0_deaths)
            barriers = [s.barrier_height for s in h0_deaths]
            min_barrier = minimum(barriers)
            max_barrier = maximum(barriers)
            mean_barrier = sum(barriers) / length(barriers)
            
            println("\nBarrier Height Statistics (H0 deaths only):")
            println("  Lowest barrier: $(round(min_barrier, digits=6)) Hartree ($(round(min_barrier * 627.509, digits=3)) kcal/mol)")
            println("  Highest barrier: $(round(max_barrier, digits=6)) Hartree ($(round(max_barrier * 627.509, digits=3)) kcal/mol)")
            println("  Average barrier: $(round(mean_barrier, digits=6)) Hartree ($(round(mean_barrier * 627.509, digits=3)) kcal/mol)")
        end
    end
    
    println("="^50)
    
    return all_saddles
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
