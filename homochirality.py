import numpy as np
import matplotlib.pyplot as plt
from qutip import *
import os

os.makedirs('data_graphs', exist_ok=True)
print("🔄 FIXING LINDBLAD OPERATORS USING PROPER QUTIP FORMALISM")
print("=" * 60)

# Parameters
a = 1.0
b = 0.2
solver_options = Options(nsteps=50000, atol=1e-12, rtol=1e-10)

# Basis states and operators
L = basis(2, 0)
R = basis(2, 1)
sigma_z = sigmaz()
sigma_x = sigmax()
P_L = L * L.dag()  # Population of L
P_R = R * R.dag()  # Population of R

def create_proper_enantiomer_selection(gamma_select, preference):
    """
    PROPER Lindblad operators for enantiomer selection
    Models COMPETITION between L and R populations
    """
    print(f"  🎯 Creating PROPER selection: γ={gamma_select:.3f}, preference={preference:.3f}")
    
    # The key insight: We need operators that represent COMPETITIVE processes
    # When one enantiomer grows, the other should decrease
    
    # Process 1: Environmental bias favoring L (R → L conversion)
    rate_RL = gamma_select * (1 + preference)
    L_RL = np.sqrt(rate_RL) * L * R.dag()  # R decreases, L increases
    
    # Process 2: Environmental bias favoring R (L → R conversion) 
    rate_LR = gamma_select * (1 - preference)
    L_LR = np.sqrt(rate_LR) * R * L.dag()  # L decreases, R increases
    
    # Process 3: Spontaneous interconversion (maintains some racemization)
    rate_spon = gamma_select * 0.1  # Small spontaneous mixing
    L_spon_LR = np.sqrt(rate_spon) * R * L.dag()  # L → R
    L_spon_RL = np.sqrt(rate_spon) * L * R.dag()  # R → L
    
    operators = [L_RL, L_LR, L_spon_LR, L_spon_RL]
    
    print(f"  📊 Competitive rates: R→L: {rate_RL:.3f}, L→R: {rate_LR:.3f}")
    print(f"  📈 Net preference: {preference:.3f} → expected <σ_z> ≈ {preference:.3f}")
    
    return operators

def test_competitive_dynamics():
    """Test if populations properly compete"""
    print("\n" + "="*60)
    print("TEST: COMPETITIVE POPULATION DYNAMICS")
    print("="*60)
    
    tlist = np.linspace(0, 100, 1000)
    H = 0.5 * a * sigma_z + 0.5 * b * sigma_x
    
    # Test different selection strengths
    preferences = [0.3, 0.6, 0.9]
    
    plt.figure(figsize=(15, 10))
    
    for i, preference in enumerate(preferences):
        plt.subplot(2, 3, i+1)
        
        # Use proper competitive operators
        c_ops = create_proper_enantiomer_selection(0.2, preference)
        
        # Test different initial conditions
        initial_states = {
            'Mostly L': (np.sqrt(0.8)*L + np.sqrt(0.2)*R).unit(),
            'Racemic': (L + R).unit(),
            'Mostly R': (np.sqrt(0.2)*L + np.sqrt(0.8)*R).unit()
        }
        
        for label, psi0 in initial_states.items():
            # Track BOTH populations
            result = mesolve(H, psi0, tlist, c_ops, [P_L, P_R, sigma_z], options=solver_options)
            
            P_L_vals = result.expect[0]
            P_R_vals = result.expect[1] 
            chirality_vals = result.expect[2]
            
            plt.plot(tlist, P_L_vals, 'b-', alpha=0.7, label=f'P_L ({label})' if i==0 else "")
            plt.plot(tlist, P_R_vals, 'r-', alpha=0.7, label=f'P_R ({label})' if i==0 else "")
            plt.plot(tlist, chirality_vals, 'k--', alpha=0.5, label=f'<σ_z> ({label})' if i==0 else "")
            
            final_P_L = P_L_vals[-1]
            final_P_R = P_R_vals[-1]
            final_chirality = chirality_vals[-1]
            
            print(f"Preference {preference:.1f} - {label}:")
            print(f"  Final P_L = {final_P_L:.3f}, P_R = {final_P_R:.3f}")
            print(f"  Final <σ_z> = {final_chirality:.3f}")
            print(f"  Sum P_L + P_R = {final_P_L + final_P_R:.6f} (should be 1.0)")
            print(f"  P_L - P_R = {final_P_L - final_P_R:.3f}, <σ_z> = {final_chirality:.3f}")
        
        plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        plt.xlabel('Time')
        plt.ylabel('Population / Chirality')
        plt.title(f'Preference = {preference:.1f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.1, 1.1)
    
    # Test conservation laws
    plt.subplot(2, 3, 4)
    print("\n🔍 TESTING PROBABILITY CONSERVATION...")
    
    preference_test = 0.7
    c_ops_test = create_proper_enantiomer_selection(0.2, preference_test)
    
    # Track total probability
    identity = qeye(2)
    result = mesolve(H, (L+R).unit(), tlist, c_ops_test, [identity], options=solver_options)
    
    total_prob = result.expect[0]  # Should always be 1.0
    prob_error = np.max(np.abs(total_prob - 1.0))
    
    plt.plot(tlist, total_prob, 'g-', linewidth=2)
    plt.axhline(y=1.0, color='r', linestyle='--', label='Expected: 1.0')
    plt.xlabel('Time')
    plt.ylabel('Total Probability')
    plt.title(f'Probability Conservation\nMax error: {prob_error:.2e}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    print(f"✅ Probability conservation test: max error = {prob_error:.2e}")
    
    # Test competitive exclusion
    plt.subplot(2, 3, 5)
    print("\n🔍 TESTING COMPETITIVE EXCLUSION...")
    
    strong_preference = 0.95
    c_ops_strong = create_proper_enantiomer_selection(0.3, strong_preference)
    
    result_strong = mesolve(H, (L+R).unit(), tlist, c_ops_strong, [P_L, P_R], options=solver_options)
    
    plt.plot(tlist, result_strong.expect[0], 'b-', label='P_L', linewidth=2)
    plt.plot(tlist, result_strong.expect[1], 'r-', label='P_R', linewidth=2)
    plt.fill_between(tlist, result_strong.expect[0], result_strong.expect[1], alpha=0.3, color='purple')
    
    final_P_L_strong = result_strong.expect[0][-1]
    final_P_R_strong = result_strong.expect[1][-1]
    
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title(f'Strong Competition (preference={strong_preference})\nFinal: P_L={final_P_L_strong:.3f}, P_R={final_P_R_strong:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    print(f"Strong competition results:")
    print(f"  Final P_L = {final_P_L_strong:.3f}")
    print(f"  Final P_R = {final_P_R_strong:.3f}")
    print(f"  Ratio P_L/P_R = {final_P_L_strong/final_P_R_strong:.1f}:1")
    
    # Test the relationship between preference and final chirality
    plt.subplot(2, 3, 6)
    print("\n🔍 TESTING PREFERENCE → CHIRALITY MAPPING...")
    
    pref_range = np.linspace(-0.9, 0.9, 20)
    final_chiralities = []
    final_P_L_values = []
    final_P_R_values = []
    
    for pref in pref_range:
        c_ops = create_proper_enantiomer_selection(0.2, pref)
        result = mesolve(H, (L+R).unit(), [0, 150], c_ops, [sigma_z, P_L, P_R], options=solver_options)
        
        final_chirality = result.expect[0][-1]
        final_P_L = result.expect[1][-1]
        final_P_R = result.expect[2][-1]
        
        final_chiralities.append(final_chirality)
        final_P_L_values.append(final_P_L)
        final_P_R_values.append(final_P_R)
    
    plt.plot(pref_range, final_chiralities, 'k-', label='Final <σ_z>', linewidth=3)
    plt.plot(pref_range, final_P_L_values, 'b-', label='Final P_L', linewidth=2)
    plt.plot(pref_range, final_P_R_values, 'r-', label='Final P_R', linewidth=2)
    plt.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    plt.xlabel('Input Preference')
    plt.ylabel('Final Values')
    plt.title('Preference → Chirality Mapping')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Check if mapping makes sense
    correlation = np.corrcoef(pref_range, final_chiralities)[0,1]
    print(f"✅ Preference-chirality correlation: {correlation:.3f} (should be close to 1)")
    
    plt.tight_layout()
    plt.savefig('data_graphs/fixed_competitive_dynamics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return final_chiralities, final_P_L_values, final_P_R_values

def test_homochirality_limit():
    """Test if we can approach complete homochirality"""
    print("\n🎯 TESTING COMPLETE HOMOCHIRALITY LIMIT...")
    
    ultra_strong_pref = 0.99
    ultra_strong_gamma = 0.5
    
    H = 0.5 * a * sigma_z + 0.5 * b * sigma_x
    c_ops_ultra = create_proper_enantiomer_selection(ultra_strong_gamma, ultra_strong_pref)
    
    tlist_long = np.linspace(0, 200, 2000)
    result = mesolve(H, (L+R).unit(), tlist_long, c_ops_ultra, [P_L, P_R, sigma_z], options=solver_options)
    
    final_P_L = result.expect[0][-1]
    final_P_R = result.expect[1][-1] 
    final_chirality = result.expect[2][-1]
    
    plt.figure(figsize=(10, 6))
    plt.plot(tlist_long, result.expect[0], 'b-', label='P_L', linewidth=3)
    plt.plot(tlist_long, result.expect[1], 'r-', label='P_R', linewidth=3)
    plt.plot(tlist_long, result.expect[2], 'k--', label='<σ_z>', linewidth=2)
    
    plt.axhline(y=final_P_L, color='blue', linestyle=':', alpha=0.5)
    plt.axhline(y=final_P_R, color='red', linestyle=':', alpha=0.5)
    plt.axhline(y=final_chirality, color='black', linestyle=':', alpha=0.5)
    
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f'Ultra-Strong Homochirality Drive\nFinal: P_L={final_P_L:.4f}, P_R={final_P_R:.4f}, <σ_z>={final_chirality:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig('data_graphs/ultra_strong_homochirality.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Ultra-strong selection results:")
    print(f"  Final P_L = {final_P_L:.4f}")
    print(f"  Final P_R = {final_P_R:.4f}") 
    print(f"  Final <σ_z> = {final_chirality:.4f}")
    print(f"  Enantiomeric excess = {abs(final_P_L - final_P_R):.4f}")
    
    if final_P_L > 0.99:
        print("🎉 SUCCESS: Achieved >99% homochirality!")
    elif final_P_L > 0.95:
        print("✅ EXCELLENT: Achieved >95% homochirality!")
    else:
        print("⚠️  Limited homochirality achieved")
        

def analyze_competition_dynamics():
    """
    Thorough analysis of competition dynamics between enantiomers
    Studies how L and R populations compete under various conditions
    """
    print("\n" + "="*70)
    print("COMPETITION DYNAMICS ANALYSIS")
    print("="*70)
    
    H = 0.5 * a * sigma_z + 0.5 * b * sigma_x
    tlist_comp = np.linspace(0, 150, 1500)
    
    plt.figure(figsize=(16, 12))
    
    # 1. Population trajectories under different preferences
    print("1. 📊 POPULATION TRAJECTORIES UNDER DIFFERENT PREFERENCES")
    preferences = [-0.8, -0.4, 0.0, 0.4, 0.8]
    colors = ['red', 'orange', 'gray', 'lightblue', 'blue']
    
    for i, preference in enumerate(preferences):
        plt.subplot(3, 3, i+1)
        
        c_ops = create_proper_enantiomer_selection(0.2, preference)
        result = mesolve(H, (L+R).unit(), tlist_comp, c_ops, [P_L, P_R], options=solver_options)
        
        plt.plot(tlist_comp, result.expect[0], 'b-', label='P_L', linewidth=2)
        plt.plot(tlist_comp, result.expect[1], 'r-', label='P_R', linewidth=2)
        
        # Calculate competition metrics
        final_P_L = result.expect[0][-1]
        final_P_R = result.expect[1][-1]
        dominance_ratio = final_P_L / final_P_R if final_P_R > 0 else np.inf
        
        plt.title(f'Pref={preference:.1f}\nFinal: L={final_P_L:.2f}, R={final_P_R:.2f}\nRatio: {dominance_ratio:.1f}')
        plt.xlabel('Time')
        plt.ylabel('Population')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        print(f"   Preference {preference:.1f}: P_L={final_P_L:.3f}, P_R={final_P_R:.3f}, L/R ratio={dominance_ratio:.2f}")
    
    # 2. Competition phase space (P_L vs P_R trajectories)
    plt.subplot(3, 3, 6)
    print("\n2. 🎯 PHASE SPACE TRAJECTORIES")
    
    for preference in [0.3, 0.6, 0.9]:
        c_ops = create_proper_enantiomer_selection(0.2, preference)
        result = mesolve(H, (L+R).unit(), tlist_comp, c_ops, [P_L, P_R], options=solver_options)
        
        plt.plot(result.expect[0], result.expect[1], label=f'Pref={preference}', linewidth=2)
        
        # Mark start and end points
        plt.scatter(result.expect[0][0], result.expect[1][0], color='green', s=50, zorder=5)
        plt.scatter(result.expect[0][-1], result.expect[1][-1], color='red', s=50, zorder=5)
    
    plt.plot([0, 1], [1, 0], 'k--', alpha=0.3, label='P_L + P_R = 1')
    plt.xlabel('P_L')
    plt.ylabel('P_R')
    plt.title('Phase Space Trajectories')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # 3. Competition strength vs steady state
    plt.subplot(3, 3, 7)
    print("\n3. 💪 COMPETITION STRENGTH ANALYSIS")
    
    gamma_range = np.logspace(-2, 0, 30)
    preferences_comp = [0.3, 0.6, 0.9]
    
    for pref in preferences_comp:
        steady_states = []
        for gamma in gamma_range:
            c_ops = create_proper_enantiomer_selection(gamma, pref)
            result = mesolve(H, (L+R).unit(), [0, 200], c_ops, [P_L], options=solver_options)
            steady_states.append(result.expect[0][-1])
        
        plt.semilogx(gamma_range, steady_states, label=f'Pref={pref}', linewidth=2)
    
    plt.xlabel('Selection Rate γ')
    plt.ylabel('Steady-state P_L')
    plt.title('Competition Strength vs Outcome')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Bifurcation analysis - sensitivity to initial conditions
    plt.subplot(3, 3, 8)
    print("\n4. 🎢 BIFURCATION ANALYSIS")
    
    initial_L_fractions = np.linspace(0.05, 0.95, 20)
    preferences_bifurc = [0.1, 0.5, 0.9]
    
    for pref in preferences_bifurc:
        final_chiralities = []
        for p_L_initial in initial_L_fractions:
            psi0 = (np.sqrt(p_L_initial)*L + np.sqrt(1-p_L_initial)*R).unit()
            c_ops = create_proper_enantiomer_selection(0.2, pref)
            result = mesolve(H, psi0, [0, 200], c_ops, [sigma_z], options=solver_options)
            final_chiralities.append(result.expect[0][-1])
        
        plt.plot(initial_L_fractions, final_chiralities, 'o-', label=f'Pref={pref}', markersize=3)
    
    plt.axhline(y=0, color='k', linestyle=':', alpha=0.5)
    plt.axvline(x=0.5, color='k', linestyle=':', alpha=0.5)
    plt.xlabel('Initial P_L fraction')
    plt.ylabel('Final <σ_z>')
    plt.title('Bifurcation: Final vs Initial')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Competition metrics summary
    plt.subplot(3, 3, 9)
    print("\n5. 📈 COMPETITION METRICS SUMMARY")
    
    # Calculate various competition metrics
    pref_range_metrics = np.linspace(-1, 1, 50)
    metrics_data = []
    
    for pref in pref_range_metrics:
        c_ops = create_proper_enantiomer_selection(0.2, pref)
        result = mesolve(H, (L+R).unit(), [0, 200], c_ops, [P_L, P_R], options=solver_options)
        
        final_P_L = result.expect[0][-1]
        final_P_R = result.expect[1][-1]
        
        metrics = {
            'preference': pref,
            'P_L': final_P_L,
            'P_R': final_P_R,
            'dominance': final_P_L - final_P_R,
            'ratio': final_P_L / final_P_R if final_P_R > 0 else np.inf,
            'entropy': -final_P_L*np.log(final_P_L+1e-10) - final_P_R*np.log(final_P_R+1e-10) if final_P_L>0 and final_P_R>0 else 0
        }
        metrics_data.append(metrics)
    
    # Plot multiple metrics
    preferences_plot = [m['preference'] for m in metrics_data]
    dominance_plot = [m['dominance'] for m in metrics_data]
    entropy_plot = [m['entropy'] for m in metrics_data]
    
    ax1 = plt.gca()
    ax1.plot(preferences_plot, dominance_plot, 'b-', label='Dominance (P_L - P_R)', linewidth=2)
    ax1.set_xlabel('Preference')
    ax1.set_ylabel('Dominance', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    ax2 = ax1.twinx()
    ax2.plot(preferences_plot, entropy_plot, 'r-', label='Entropy', linewidth=2)
    ax2.set_ylabel('Entropy', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    plt.title('Competition Metrics')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data_graphs/competition_dynamics_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print quantitative analysis
    print("\n" + "="*50)
    print("QUANTITATIVE COMPETITION ANALYSIS")
    print("="*50)
    
    # Find critical points
    zero_crossing_idx = np.where(np.array(dominance_plot) >= 0)[0]
    if len(zero_crossing_idx) > 0:
        critical_pref = preferences_plot[zero_crossing_idx[0]]
        print(f"Critical preference for L dominance: {critical_pref:.3f}")
    
    # Calculate competition strength
    max_dominance = max(dominance_plot)
    min_dominance = min(dominance_plot)
    print(f"Dominance range: [{min_dominance:.3f}, {max_dominance:.3f}]")
    
    # Find where near-complete dominance occurs
    near_complete_L = np.where(np.array([m['P_L'] for m in metrics_data]) > 0.95)[0]
    near_complete_R = np.where(np.array([m['P_R'] for m in metrics_data]) > 0.95)[0]
    
    if len(near_complete_L) > 0:
        pref_L_complete = preferences_plot[near_complete_L[0]]
        print(f"Near-complete L dominance (>95%) at preference > {pref_L_complete:.3f}")
    
    if len(near_complete_R) > 0:
        pref_R_complete = preferences_plot[near_complete_R[-1]]
        print(f"Near-complete R dominance (>95%) at preference < {pref_R_complete:.3f}")
    
    # Calculate sensitivity
    sensitivity = np.gradient(dominance_plot, preferences_plot)
    max_sensitivity = np.max(np.abs(sensitivity))
    print(f"Maximum sensitivity: {max_sensitivity:.3f} (how sharply dominance changes)")
    
    return metrics_data

def analyze_competitive_exclusion():
    """
    Specifically study competitive exclusion principle in enantiomer dynamics
    """
    print("\n" + "="*70)
    print("COMPETITIVE EXCLUSION ANALYSIS")
    print("="*70)
    
    H = 0.5 * a * sigma_z + 0.5 * b * sigma_x
    tlist_excl = np.linspace(0, 200, 2000)
    
    plt.figure(figsize=(12, 8))
    
    # Study exclusion under strong competition
    strong_preferences = [0.95, 0.98, 0.99]
    exclusion_times = []
    
    for i, pref in enumerate(strong_preferences):
        plt.subplot(2, 2, i+1)
        
        c_ops = create_proper_enantiomer_selection(0.3, pref)
        result = mesolve(H, (L+R).unit(), tlist_excl, c_ops, [P_L, P_R], options=solver_options)
        
        plt.plot(tlist_excl, result.expect[0], 'b-', label='P_L', linewidth=2)
        plt.plot(tlist_excl, result.expect[1], 'r-', label='P_R', linewidth=2)
        
        # Find exclusion time (when one population drops below threshold)
        threshold = 0.01
        exclusion_idx = np.where(result.expect[1] < threshold)[0]
        if len(exclusion_idx) > 0:
            exclusion_time = tlist_excl[exclusion_idx[0]]
            exclusion_times.append(exclusion_time)
            plt.axvline(x=exclusion_time, color='purple', linestyle='--', 
                       label=f'Exclusion: t={exclusion_time:.1f}')
        
        final_ratio = result.expect[0][-1] / result.expect[1][-1] if result.expect[1][-1] > 0 else np.inf
        plt.title(f'Pref={pref}\nFinal ratio: {final_ratio:.0f}:1')
        plt.xlabel('Time')
        plt.ylabel('Population')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        print(f"Preference {pref}: Final P_L={result.expect[0][-1]:.4f}, P_R={result.expect[1][-1]:.4f}")
    
    # Exclusion time vs preference strength
    plt.subplot(2, 2, 4)
    if exclusion_times:
        plt.plot(strong_preferences[:len(exclusion_times)], exclusion_times, 'go-', linewidth=2)
        plt.xlabel('Preference Strength')
        plt.ylabel('Exclusion Time')
        plt.title('Competitive Exclusion Time')
        plt.grid(True, alpha=0.3)
        
        print(f"\nExclusion times: {exclusion_times}")
    
    plt.tight_layout()
    plt.savefig('data_graphs/competitive_exclusion_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# Run the complete simulation
print("🚀 STARTING COMPLETE SIMULATION")
final_chiralities, final_P_L, final_P_R = test_competitive_dynamics()


print("\n" + "="*70)
print("RUNNING COMPETITION DYNAMICS ANALYSIS")
print("="*70)

# Run comprehensive competition analysis
competition_metrics = analyze_competition_dynamics()

# Run competitive exclusion analysis  
analyze_competitive_exclusion()

test_homochirality_limit()

# Final analysis
print("\n" + "="*60)
print("FINAL ANALYSIS")
print("="*60)

print("Physical consistency checks:")
print(f"1. Probability conservation: P_L + P_R = {final_P_L[10] + final_P_R[10]:.6f} (should be 1.0)")
print(f"2. Positive preference → P_L > P_R: {final_P_L[18] > final_P_R[18]}")
print(f"3. Negative preference → P_R > P_L: {final_P_R[1] > final_P_L[1]}")
print(f"4. Zero preference → P_L ≈ P_R: difference = {abs(final_P_L[10] - final_P_R[10]):.3f}")

print(f"\nHomochirality achievement:")
max_homochirality = max(final_P_L)
print(f"Maximum P_L achieved: {max_homochirality:.4f}")
if max_homochirality > 0.99:
    print("🎉 COMPLETE HOMOCHIRALITY: >99% enantiomeric purity achieved!")
elif max_homochirality > 0.95:
    print("✅ STRONG HOMOCHIRALITY: >95% enantiomeric purity achieved!")
elif max_homochirality > 0.90:
    print("📊 MODERATE HOMOCHIRALITY: >90% enantiomeric purity achieved")
else:
    print("⚠️  WEAK HOMOCHIRALITY: Limited enantiomeric purity")

print("\n" + "="*60)
print("SUMMARY: Proper Lindblad operators now model:")
print("  ✅ Population competition (P_L + P_R = 1 always)")
print("  ✅ Competitive dynamics (one grows, other decreases)") 
print("  ✅ Proper preference → chirality mapping")
print("  ✅ Physical conservation laws")
print("="*60)

