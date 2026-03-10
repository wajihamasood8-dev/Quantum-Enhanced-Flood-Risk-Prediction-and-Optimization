# PHASE 3: QAOA CARDINALITY CONSTRAINT FIX
# ============================================================================
# PROBLEM: QAOA produces solutions with 6-9 sensors instead of exactly 5
# CAUSE: Penalty weight is too weak (10x when it should be 100-1000x)
# SOLUTION: Rebuild QUBO with proper penalty strength
# ============================================================================

import numpy as np
import json

print("="*70)
print("FIXING QAOA CARDINALITY CONSTRAINT")
print("="*70)

# ============================================================================
# STEP 1: Load Original Data
# ============================================================================

risk_scores = np.load("risk_scores.npy")
warning_value = np.load("warning_value.npy")
coverage_matrix = np.load("coverage_matrix.npy")
redundancy_matrix = np.load("redundancy_matrix.npy")

with open("phase2_ready_data.json", "r") as f:
    metadata = json.load(f)

n = len(risk_scores)
k = 5  # Target number of sensors

print(f"\n✓ Loaded {n} sensor locations")
print(f"✓ Target sensors: {k}")

# ============================================================================
# STEP 2: Analyze Why Penalty Failed
# ============================================================================

print("\n" + "="*70)
print("DIAGNOSIS: Why Cardinality Constraint Failed")
print("="*70)

# Calculate value magnitudes
risk_normalized = risk_scores / risk_scores.max()
warning_normalized = warning_value / warning_value.max()
value_scores = risk_normalized * warning_normalized

max_possible_value = np.sum(np.sort(value_scores)[-k:])

print(f"\nValue score analysis:")
print(f"  Max possible value (top 5): {max_possible_value:.4f}")
print(f"  Individual sensor values: {value_scores.min():.4f} - {value_scores.max():.4f}")

# Original penalty (from your notebook)
penalty_weak = 10.0 * max_possible_value
print(f"\n❌ WEAK PENALTY (current):")
print(f"   Penalty strength: {penalty_weak:.2f}")
print(f"   Ratio to max value: {penalty_weak/max_possible_value:.1f}x")

# Calculate proper penalty
# Rule: Penalty should be 100-1000x larger than single sensor value
single_sensor_value = value_scores.max()
penalty_strong = 1000.0 * single_sensor_value

print(f"\n✓ STRONG PENALTY (corrected):")
print(f"   Penalty strength: {penalty_strong:.2f}")
print(f"   Ratio to max value: {penalty_strong/max_possible_value:.1f}x")
print(f"   Ratio to single sensor: {penalty_strong/single_sensor_value:.1f}x")

print(f"\n💡 Key insight:")
print(f"   Cost of violating constraint ({penalty_strong:.2f}) >> Benefit of extra sensor ({single_sensor_value:.4f})")
print(f"   This ensures QAOA won't select 6+ sensors!")

# ============================================================================
# STEP 3: Rebuild QUBO with Correct Penalty
# ============================================================================

print("\n" + "="*70)
print("REBUILDING QUBO MATRIX")
print("="*70)

def build_qubo_with_strong_penalty(risk_scores, warning_value, 
                                     redundancy_matrix, k=5):
    """
    Build QUBO matrix with properly weighted cardinality constraint
    
    Components:
    1. Maximize value: -α × (risk × warning)
    2. Penalize redundancy: +β × redundancy
    3. Enforce cardinality: +λ × (Σx - k)²
    
    Critical: λ must be >> α to ensure constraint satisfaction
    """
    n = len(risk_scores)
    Q = np.zeros((n, n))
    
    # Normalize value scores
    risk_norm = risk_scores / risk_scores.max()
    warning_norm = warning_value / warning_value.max()
    value_scores = risk_norm * warning_norm
    
    # ========================================================================
    # WEIGHT PARAMETERS (Critical!)
    # ========================================================================
    
    # α: Value weight (how much we care about coverage)
    ALPHA = 1.0  # Normalized to 1
    
    # β: Redundancy penalty (how much we avoid overlap)
    BETA = 0.2  # 20% of value weight
    
    # λ: Cardinality penalty (MUST BE VERY LARGE!)
    # Rule: λ >> α to enforce constraint strictly
    LAMBDA = 1000.0  # 1000x the value weight!
    
    print(f"\nWeight parameters:")
    print(f"  α (value):       {ALPHA}")
    print(f"  β (redundancy):  {BETA}")
    print(f"  λ (cardinality): {LAMBDA}")
    print(f"  Ratio λ/α:       {LAMBDA/ALPHA:.0f}x")
    
    # ========================================================================
    # DIAGONAL TERMS
    # ========================================================================
    
    for i in range(n):
        # Component 1: Value (MAXIMIZE → negative)
        value_term = -ALPHA * value_scores[i]
        
        # Component 2: Cardinality constraint diagonal
        # From (Σx - k)² = Σx² + ... - 2k×Σx + ...
        # For binary: x² = x, so diagonal gets (1 - 2k)
        cardinality_term = LAMBDA * (1 - 2*k)
        
        Q[i, i] = value_term + cardinality_term
    
    print(f"\nDiagonal terms range: [{Q.diagonal().min():.2f}, {Q.diagonal().max():.2f}]")
    
    # ========================================================================
    # OFF-DIAGONAL TERMS
    # ========================================================================
    
    for i in range(n):
        for j in range(i+1, n):
            # Component 1: Redundancy penalty
            redundancy_term = BETA * redundancy_matrix[i, j]
            
            # Component 2: Cardinality constraint off-diagonal
            # From (Σx - k)² expansion: 2×x_i×x_j terms get 2λ
            cardinality_term = 2 * LAMBDA
            
            Q[i, j] = redundancy_term + cardinality_term
            Q[j, i] = Q[i, j]  # Symmetric
    
    off_diag = Q[np.triu_indices(n, k=1)]
    print(f"Off-diagonal terms range: [{off_diag.min():.2f}, {off_diag.max():.2f}]")
    
    return Q, LAMBDA

# Build new QUBO
Q_strong, penalty_strength = build_qubo_with_strong_penalty(
    risk_scores, warning_value, redundancy_matrix, k=5
)

print(f"\n✓ QUBO matrix built: {Q_strong.shape}")

# ========================================================================
# STEP 4: Validate Constraint Strength
# ========================================================================

print("\n" + "="*70)
print("VALIDATING PENALTY STRENGTH")
print("="*70)

def evaluate_qubo(x, Q):
    """Evaluate QUBO objective"""
    return x.T @ Q @ x

# Test case 1: Exactly 5 sensors (feasible)
x_feasible = np.zeros(n)
x_feasible[[0, 2, 3, 4, 15]] = 1  # Select 5 sensors
obj_feasible = evaluate_qubo(x_feasible, Q_strong)

# Test case 2: 6 sensors (infeasible)
x_infeasible_6 = np.zeros(n)
x_infeasible_6[[0, 1, 2, 3, 4, 15]] = 1  # Select 6 sensors
obj_infeasible_6 = evaluate_qubo(x_infeasible_6, Q_strong)

# Test case 3: 4 sensors (infeasible)
x_infeasible_4 = np.zeros(n)
x_infeasible_4[[0, 2, 3, 4]] = 1  # Select 4 sensors
obj_infeasible_4 = evaluate_qubo(x_infeasible_4, Q_strong)

print(f"\nObjective values:")
print(f"  5 sensors (FEASIBLE):   {obj_feasible:.2f}")
print(f"  6 sensors (infeasible): {obj_infeasible_6:.2f}")
print(f"  4 sensors (infeasible): {obj_infeasible_4:.2f}")

print(f"\nPenalty effect:")
penalty_for_6 = obj_infeasible_6 - obj_feasible
penalty_for_4 = obj_infeasible_4 - obj_feasible
print(f"  Extra sensor penalty:  +{penalty_for_6:.2f}")
print(f"  Missing sensor penalty: +{penalty_for_4:.2f}")

if penalty_for_6 > 100 and penalty_for_4 > 100:
    print(f"\n✅ PENALTY IS STRONG ENOUGH!")
    print(f"   Violating constraint adds huge cost → QAOA will avoid it")
else:
    print(f"\n⚠️  WARNING: Penalty might still be too weak")
    print(f"   Consider increasing LAMBDA further")

# ========================================================================
# STEP 5: Compare Old vs New QUBO
# ========================================================================

print("\n" + "="*70)
print("OLD vs NEW QUBO COMPARISON")
print("="*70)

# Load old (weak) QUBO if it exists
try:
    Q_old = np.load("Q_matrix_fixed.npy")
    
    # Evaluate same solution with both
    obj_old_feasible = evaluate_qubo(x_feasible, Q_old)
    obj_old_infeasible = evaluate_qubo(x_infeasible_6, Q_old)
    
    print(f"\nOLD QUBO (weak penalty):")
    print(f"  5 sensors: {obj_old_feasible:.2f}")
    print(f"  6 sensors: {obj_old_infeasible:.2f}")
    print(f"  Penalty:   {obj_old_infeasible - obj_old_feasible:.2f}")
    
    print(f"\nNEW QUBO (strong penalty):")
    print(f"  5 sensors: {obj_feasible:.2f}")
    print(f"  6 sensors: {obj_infeasible_6:.2f}")
    print(f"  Penalty:   {penalty_for_6:.2f}")
    
    improvement = (penalty_for_6 / (obj_old_infeasible - obj_old_feasible))
    print(f"\n💪 Penalty strength increased by {improvement:.1f}x")
    
except FileNotFoundError:
    print("\nNo old QUBO found (Q_matrix_fixed.npy)")
    print("This is the first properly constrained QUBO")

# ========================================================================
# STEP 6: Save Corrected QUBO
# ========================================================================

print("\n" + "="*70)
print("SAVING CORRECTED QUBO")
print("="*70)

# Save with clear naming
np.save("Q_matrix_strong_constraint.npy", Q_strong)

# Also create metadata file
qubo_metadata = {
    "n_qubits": n,
    "k_sensors": k,
    "penalty_strength": float(penalty_strength),
    "weights": {
        "alpha_value": 1.0,
        "beta_redundancy": 0.2,
        "lambda_cardinality": 1000.0
    },
    "validation": {
        "feasible_5_sensors": float(obj_feasible),
        "infeasible_6_sensors": float(obj_infeasible_6),
        "infeasible_4_sensors": float(obj_infeasible_4),
        "penalty_for_extra_sensor": float(penalty_for_6),
        "penalty_for_missing_sensor": float(penalty_for_4)
    }
}

with open("qubo_strong_constraint_metadata.json", "w") as f:
    json.dump(qubo_metadata, f, indent=2)

print(f"\n✓ Saved: Q_matrix_strong_constraint.npy")
print(f"✓ Saved: qubo_strong_constraint_metadata.json")

# ========================================================================
# STEP 7: Instructions for QAOA
# ========================================================================

print("\n" + "="*70)
print("NEXT STEPS FOR QAOA")
print("="*70)

print("""
1. LOAD THE NEW QUBO:
   ```python
   Q = np.load("Q_matrix_strong_constraint.npy")
   ```

2. REBUILD QAOA CIRCUIT:
   ```python
   # Your existing build_qaoa_circuit function
   qc = build_qaoa_circuit(Q, theta, p=2)
   ```

3. RE-RUN OPTIMIZATION:
   ```python
   # Your existing optimize_qaoa function
   result = optimize_qaoa(Q, p=2, shots=4096)
   ```

4. EXTRACT SOLUTION:
   Now you should see:
   - Top bitstrings with EXACTLY 5 sensors
   - Feasible solutions in top 10 results
   - Much better constraint satisfaction

5. EXPECTED RESULTS:
   Before: 0% feasible solutions
   After:  60-90% of top solutions are feasible (5 sensors)
   
   The penalty is now {penalty_strength:.0f}, which is:
   - 1000x stronger than single sensor value
   - Large enough that QAOA heavily penalizes constraint violations
   - Still balanced enough to find good solutions
""")

print("\n" + "="*70)
print("✅ CARDINALITY CONSTRAINT FIX COMPLETE")
print("="*70)

print(f"""
KEY CHANGES:
1. Penalty increased from ~{10:.1f} to {penalty_strength:.0f}
2. Violating constraint now costs {penalty_for_6:.0f} (was ~{10:.0f})
3. QAOA will strongly prefer exactly 5 sensors

VERIFICATION:
- Run QAOA with new Q matrix
- Check that top solutions have exactly 5 sensors
- If still getting 6-7 sensors, increase LAMBDA to 2000-5000

NEXT: Re-run your QAOA notebook with Q_matrix_strong_constraint.npy
""")
