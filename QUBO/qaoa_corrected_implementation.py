# CORRECTED QAOA IMPLEMENTATION
# ============================================================================
# Use this code in your Jupyter notebook to replace the weak penalty version
# ============================================================================

import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit_aer.primitives import SamplerV2

# ============================================================================
# STEP 1: Load Strong-Constraint QUBO
# ============================================================================

print("="*70)
print("QAOA WITH STRONG CARDINALITY CONSTRAINT")
print("="*70)

# Load the corrected QUBO matrix
Q = np.load("Q_matrix_strong_constraint.npy")
risk_scores = np.load("risk_scores.npy")

with open("phase2_ready_data.json", "r") as f:
    data = json.load(f)
    location_ids = data["location_ids"]

n = Q.shape[0]
k = 5  # Target sensors

print(f"\n✓ Loaded Q matrix: {n}x{n}")
print(f"✓ Penalty strength: Very strong (1000x)")
print(f"✓ Target sensors: {k}")

# ============================================================================
# STEP 2: Build QAOA Circuit (Same as before, but with new Q)
# ============================================================================

def build_qaoa_circuit(Q, theta, p):
    """
    Build QAOA circuit with p layers
    
    Args:
        Q: QUBO matrix (n x n)
        theta: Parameters [γ₁, β₁, γ₂, β₂, ..., γₚ, βₚ]
        p: Number of QAOA layers
    
    Returns:
        qc: Quantum circuit
    """
    n = Q.shape[0]
    qc = QuantumCircuit(n, n)
    
    # Initial state: Equal superposition
    qc.h(range(n))
    
    # QAOA layers
    for layer in range(p):
        gamma = theta[2*layer]
        beta = theta[2*layer + 1]
        
        # Problem Hamiltonian (cost layer)
        # Diagonal terms: RZ gates
        for i in range(n):
            qc.rz(2 * gamma * Q[i, i], i)
        
        # Off-diagonal terms: RZZ gates
        for i in range(n):
            for j in range(i+1, n):
                if Q[i, j] != 0:  # Only if interaction exists
                    qc.rzz(2 * gamma * Q[i, j], i, j)
        
        # Mixing Hamiltonian (exploration layer)
        for i in range(n):
            qc.rx(2 * beta, i)
    
    # Measurement
    qc.measure(range(n), range(n))
    
    return qc

# ============================================================================
# STEP 3: Objective Function for Optimization
# ============================================================================

def qaoa_objective(theta, Q, p, shots=4096):
    """
    Compute expected value <x|Q|x> from QAOA circuit
    
    This is what the classical optimizer minimizes
    """
    n = Q.shape[0]
    
    # Build circuit with current parameters
    qc = build_qaoa_circuit(Q, theta, p)
    
    # Sample from circuit
    sampler = SamplerV2(default_shots=shots)
    job = sampler.run([(qc)])
    counts = job.result()[0].data.meas.get_counts()
    
    # Compute expectation value
    expectation = 0.0
    total_counts = sum(counts.values())
    
    for bitstring, count in counts.items():
        # Convert bitstring to binary vector
        x = np.array([int(b) for b in bitstring])
        
        # Evaluate QUBO
        energy = x.T @ Q @ x
        
        # Weighted by probability
        prob = count / total_counts
        expectation += prob * energy
    
    return expectation

# ============================================================================
# STEP 4: Optimize Parameters
# ============================================================================

def optimize_qaoa(Q, p=2, shots=4096, maxiter=50):
    """
    Find optimal γ and β parameters using classical optimizer
    
    Args:
        Q: QUBO matrix
        p: Number of QAOA layers
        shots: Shots per circuit evaluation
        maxiter: Maximum optimizer iterations
    
    Returns:
        result: Optimization result
        history: Dictionary of optimization history
    """
    print(f"\n{'='*70}")
    print(f"🔧 OPTIMIZING QAOA PARAMETERS")
    print(f"{'='*70}")
    print(f"Layers (p):       {p}")
    print(f"Parameters:       {2*p} (γ₁, β₁, ..., γₚ, βₚ)")
    print(f"Shots per eval:   {shots}")
    print(f"Max iterations:   {maxiter}")
    
    # Initial guess: Random small values
    np.random.seed(42)
    theta_init = np.random.uniform(0, 2*np.pi, 2*p)
    
    # Track optimization history
    history = {
        'iteration': [],
        'energy': [],
        'gamma': [],
        'beta': []
    }
    
    iter_count = [0]  # Mutable counter for callback
    
    def callback(theta):
        """Track optimization progress"""
        iter_count[0] += 1
        energy = qaoa_objective(theta, Q, p, shots)
        
        history['iteration'].append(iter_count[0])
        history['energy'].append(energy)
        history['gamma'].append(theta[::2].tolist())
        history['beta'].append(theta[1::2].tolist())
        
        if iter_count[0] % 5 == 0:
            print(f"  Iter {iter_count[0]:3d}: Energy = {energy:.4f}")
    
    print(f"\nStarting optimization...")
    
    # Optimize using COBYLA
    result = minimize(
        lambda theta: qaoa_objective(theta, Q, p, shots),
        theta_init,
        method='COBYLA',
        callback=callback,
        options={'maxiter': maxiter, 'disp': False}
    )
    
    print(f"\n✓ Optimization complete!")
    print(f"  Final energy: {result.fun:.4f}")
    print(f"  Iterations: {iter_count[0]}")
    
    # Extract optimal parameters
    optimal_gamma = result.x[::2]
    optimal_beta = result.x[1::2]
    
    print(f"\nOptimal parameters:")
    for i in range(p):
        print(f"  Layer {i+1}: γ = {optimal_gamma[i]:.4f}, β = {optimal_beta[i]:.4f}")
    
    return result, history

# Run optimization
p = 2  # Start with 2 layers
result, opt_history = optimize_qaoa(Q, p=2, shots=4096, maxiter=30)

# ============================================================================
# STEP 5: Extract Best Solution with Constraint Checking
# ============================================================================

def extract_best_solution(result, Q, p, k=5, shots=8192):
    """
    Extract best FEASIBLE solution from optimized QAOA
    
    Key difference: With strong penalty, most top solutions should be feasible
    """
    print(f"\n{'='*70}")
    print(f"🔍 SOLUTION EXTRACTION")
    print(f"{'='*70}")
    print(f"Sampling shots:     {shots}")
    print(f"Target sensors:     {k}")
    
    # Build circuit with optimal parameters
    qc_optimal = build_qaoa_circuit(Q, result.x, p)
    
    # Sample many times
    sampler = SamplerV2(default_shots=shots)
    job = sampler.run([(qc_optimal)])
    counts = job.result()[0].data.meas.get_counts()
    
    print(f"Unique bitstrings:  {len(counts)}")
    
    # Sort by count
    sorted_results = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    # Display top 10
    print(f"\nTop 10 bitstrings:")
    print(f"  Rank   Bitstring            Count    Prob     #Sensors   Feasible?")
    print(f"  {'-'*70}")
    
    feasible_found = False
    best_feasible = None
    best_feasible_count = 0
    
    for rank, (bitstring, count) in enumerate(sorted_results[:10], 1):
        x = np.array([int(b) for b in bitstring])
        num_sensors = int(x.sum())
        prob = count / shots
        is_feasible = (num_sensors == k)
        
        status = "✓" if is_feasible else "✗"
        print(f"  {rank:<6} {bitstring:20s} {count:<8} {prob:<8.4f} {num_sensors:<10} {is_feasible}")
        
        if is_feasible and not feasible_found:
            best_feasible = bitstring
            best_feasible_count = count
            feasible_found = True
    
    # Find best feasible solution
    if not feasible_found:
        print(f"\n⚠️  WARNING: No feasible solutions in top 10!")
        print(f"   Searching all {len(counts)} bitstrings...")
        
        for bitstring, count in sorted_results:
            x = np.array([int(b) for b in bitstring])
            if x.sum() == k:
                best_feasible = bitstring
                best_feasible_count = count
                feasible_found = True
                break
    
    if feasible_found:
        print(f"\n✅ BEST FEASIBLE SOLUTION FOUND:")
        print(f"   Bitstring: {best_feasible}")
        print(f"   Count: {best_feasible_count} / {shots} ({best_feasible_count/shots*100:.1f}%)")
        
        # Convert to sensor selection
        x_best = np.array([int(b) for b in best_feasible])
        selected_indices = np.where(x_best == 1)[0]
        selected_sensors = [location_ids[i] for i in selected_indices]
        
        print(f"\n   Selected sensors:")
        for i, sensor in zip(selected_indices, selected_sensors):
            print(f"     {sensor}: Risk = {risk_scores[i]:.2f}")
        
        # Calculate metrics
        total_risk = sum(risk_scores[i] for i in selected_indices)
        obj_value = x_best.T @ Q @ x_best
        
        print(f"\n   Performance:")
        print(f"     Total risk coverage: {total_risk:.2f}")
        print(f"     QUBO objective: {obj_value:.4f}")
        
        return {
            'bitstring': best_feasible,
            'solution_vector': x_best,
            'selected_indices': selected_indices,
            'selected_sensors': selected_sensors,
            'count': best_feasible_count,
            'probability': best_feasible_count / shots,
            'risk_coverage': total_risk,
            'objective': obj_value,
            'feasible': True
        }
    else:
        print(f"\n❌ CRITICAL: NO FEASIBLE SOLUTIONS FOUND!")
        print(f"   This means penalty is STILL too weak")
        print(f"   Try increasing LAMBDA to 2000-5000 in fix_qaoa_cardinality.py")
        
        # Return best solution anyway
        best_bitstring = sorted_results[0][0]
        x_best = np.array([int(b) for b in best_bitstring])
        
        return {
            'bitstring': best_bitstring,
            'solution_vector': x_best,
            'selected_indices': np.where(x_best == 1)[0].tolist(),
            'count': sorted_results[0][1],
            'probability': sorted_results[0][1] / shots,
            'feasible': False
        }

# Extract solution
qaoa_solution = extract_best_solution(result, Q, p, k=5, shots=8192)

# ============================================================================
# STEP 6: Compare to Classical Baselines
# ============================================================================

print(f"\n{'='*70}")
print(f"📊 PERFORMANCE COMPARISON")
print(f"{'='*70}")

# Load classical baselines
with open("qubo_validation_results.json", "r") as f:
    classical = json.load(f)

# Display comparison
if qaoa_solution['feasible']:
    print(f"\n{'Method':<25} {'Risk Coverage':<15} {'Feasible?'}")
    print(f"{'-'*55}")
    print(f"{'Greedy':<25} {classical['greedy']['risk_coverage']:<15.2f} ✓")
    print(f"{'Simulated Annealing':<25} {classical['simulated_annealing']['risk_coverage']:<15.2f} ✓")
    print(f"{'QAOA (p={p})':<25} {qaoa_solution['risk_coverage']:<15.2f} ✓")
    
    best_classical = max(
        classical['greedy']['risk_coverage'],
        classical['simulated_annealing']['risk_coverage']
    )
    
    improvement = ((qaoa_solution['risk_coverage'] - best_classical) / best_classical) * 100
    
    if improvement > 0:
        print(f"\n🎉 QUANTUM ADVANTAGE!")
        print(f"   QAOA beats classical by {improvement:.1f}%")
    else:
        print(f"\n⚠️  QAOA: {improvement:.1f}% vs best classical")
        print(f"   Consider increasing p (layers) or shots")
else:
    print(f"\n⚠️  Cannot compare: QAOA solution not feasible")
    print(f"   Constraint penalty needs to be increased")

print(f"\n{'='*70}")
print(f"✅ QAOA ANALYSIS COMPLETE")
print(f"{'='*70}")
