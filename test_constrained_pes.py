#!/usr/bin/env python3
"""
Comprehensive test script for the advanced constrained PES calculation.
This will test various methods and compare results.
"""

import os
import sys
import subprocess
import numpy as np

def create_test_template():
    """Create a butane Z-matrix template for testing."""
    template_dir = "templates"
    os.makedirs(template_dir, exist_ok=True)
    
    butane_template = """C
C 1 1.54
C 2 1.54 1 109.5
C 3 1.54 2 109.5 1 {dihedral}
H 1 1.09 2 109.5 3 120.0
H 1 1.09 2 109.5 3 -120.0
H 2 1.09 1 109.5 4 120.0
H 2 1.09 1 109.5 4 -120.0
H 3 1.09 2 109.5 1 120.0
H 3 1.09 2 109.5 1 -120.0
H 4 1.09 3 109.5 2 120.0
H 4 1.09 3 109.5 2 -120.0
H 4 1.09 3 109.5 2 0.0"""
    
    with open(f"{template_dir}/butane_test.zmat", "w") as f:
        f.write(butane_template)
    
    print(f"Created test template: {template_dir}/butane_test.zmat")
    return f"{template_dir}/butane_test.zmat"

def run_pes_test(template_file, test_name, method='b3lyp', basis='6-311g', 
                relaxed=False, points=18):
    """Run a PES calculation test."""
    
    print(f"\n{'='*60}")
    print(f"RUNNING TEST: {test_name}")
    print(f"Method: {method}/{basis}")
    print(f"Relaxed scan: {relaxed}")
    print(f"Points: {points}")
    print(f"{'='*60}")
    
    cmd = [
        'python', 'src/1_pes_constrained.py',
        '--template', template_file,
        '--molecule', 'butane',
        '--method', method,
        '--basis', basis,
        '--points', str(points),
        '--scan-range', '0', '360'
    ]
    
    if relaxed:
        cmd.append('--relaxed')
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print(f"✅ {test_name} completed successfully")
            print("STDOUT preview:")
            print(result.stdout[-500:])  # Last 500 characters
        else:
            print(f"❌ {test_name} failed")
            print("STDERR:")
            print(result.stderr)
            print("STDOUT:")
            print(result.stdout)
            
    except subprocess.TimeoutExpired:
        print(f"⏱️ {test_name} timed out (>10 minutes)")
    except Exception as e:
        print(f"💥 {test_name} crashed: {e}")

def test_direct_import():
    """Test importing and running the script directly."""
    print(f"\n{'='*60}")
    print("TESTING DIRECT IMPORT")
    print(f"{'='*60}")
    
    try:
        # Add src to path
        sys.path.insert(0, 'src')
        
        # Import the constrained PES module
        import importlib.util
        spec = importlib.util.spec_from_file_location("pes_constrained", "src/1_pes_constrained.py")
        pes_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pes_module)
        
        print("✅ Module imported successfully")
        
        # Test basic functions
        template_file = "butane_template.zmat"
        if os.path.exists(template_file):
            template, variables = pes_module.read_zmatrix_template(template_file)
            print(f"✅ Template read successfully: {len(variables)} variables found")
            
            # Test geometry creation
            variable_values = {variables[0]: 60.0}
            geometry = pes_module.create_geometry_from_template(template, variable_values)
            print(f"✅ Geometry creation successful")
            
            # Test single energy calculation
            energy = pes_module.calculate_energy_with_constraints(
                variable_values, template, "butane_test",
                level_of_theory='b3lyp', basis_set='6-31g'
            )
            
            if energy != float('inf'):
                print(f"✅ Single energy calculation successful: {energy:.6f} Hartree")
            else:
                print("❌ Single energy calculation failed")
        else:
            print(f"❌ Template file {template_file} not found")
            
    except Exception as e:
        print(f"❌ Direct import test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run comprehensive tests of the constrained PES script."""
    
    print("TESTING ADVANCED CONSTRAINED PES CALCULATION")
    print("="*60)
    
    # Create test template
    template_file = create_test_template()
    
    # Test 1: Direct import and function testing
    test_direct_import()
    
    # Test 2: Basic rigid scan with B3LYP/6-31G (should work)
    run_pes_test(template_file, "Basic B3LYP Rigid Scan", 
                method='b3lyp', basis='6-31g', relaxed=False, points=12)
    
    # Test 3: Try HF method (minimal test)
    run_pes_test(template_file, "HF Rigid Scan", 
                method='hf', basis='6-31g', relaxed=False, points=8)
    
    # Test 4: Test constrained optimization (may have issues but should not crash)
    run_pes_test(template_file, "B3LYP Constrained Scan", 
                method='b3lyp', basis='6-31g', relaxed=True, points=8)
    
    # Test 5: Try with existing butane template if available
    if os.path.exists("butane_template.zmat"):
        run_pes_test("butane_template.zmat", "Existing Template Test", 
                    method='b3lyp', basis='6-31g', relaxed=False, points=8)
    
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print("Check the data/ and figures/ directories for output files")
    print("Look for files like:")
    print("  - data/butane_*_pes.dat")  
    print("  - figures/butane_*_pes.png")
    
    # List generated files
    if os.path.exists("data"):
        data_files = [f for f in os.listdir("data") if f.startswith("butane") and "constrained" in f]
        if data_files:
            print(f"\nGenerated data files:")
            for f in data_files:
                print(f"  - data/{f}")
    
    if os.path.exists("figures"):
        fig_files = [f for f in os.listdir("figures") if f.startswith("butane") and "constrained" in f]
        if fig_files:
            print(f"\nGenerated figure files:")
            for f in fig_files:
                print(f"  - figures/{f}")

if __name__ == "__main__":
    main()
