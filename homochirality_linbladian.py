import numpy as np
import matplotlib.pyplot as plt
from qutip import *
import pandas as pd

print("🔬 QUANTUM HOMOCHIRALITY: LINDBLADIAN DYNAMICS WITH DEPHASING")
print("=" * 70)

# Basis states and operators
L = basis(2, 0)  # |L⟩ enantiomer
R = basis(2, 1)  # |R⟩ enantiomer
sigma_x = sigmax()
sigma_y = sigmay() 
sigma_z = sigmaz()
P_L = L * L.dag()  # L population projector
P_R = R * R.dag()  # R population projector

def create_competitive_lindbladian(gamma_select, preference, gamma_dephase=0):
    """
    Create competitive Lindblad operators for enantiomer selection
    """
    # Competitive conversion rates
    rate_RL = gamma_select * (1 + preference)  # R → L
    rate_LR = gamma_select * (1 - preference)  # L → R
    
    # Lindblad operators for competitive selection
    L_RL = np.sqrt(rate_RL) * L * R.dag()  # R decreases, L increases
    L_LR = np.sqrt(rate_LR) * R * L.dag()  # L decreases, R increases
    
    c_ops = [L_RL, L_LR]
    
    # Add dephasing channel if specified
    if gamma_dephase > 0:
        L_dephase = np.sqrt(gamma_dephase) * sigma_z
        c_ops.append(L_dephase)
    
    return c_ops

def analyze_steady_state(H, c_ops, initial_state=None):
    """
    Find steady state of Lindbladian dynamics
    """
    if initial_state is None:
        initial_state = (L + R).unit()  # Racemic mixture
    
    # Use long-time evolution to find steady state
    tlist = np.linspace(0, 100, 1000)
    e_ops = [P_L, P_R, sigma_z, sigma_x, sigma_y]
    
    result = mesolve(H, initial_state, tlist, c_ops, e_ops, 
                   options=Options(nsteps=10000, atol=1e-12, rtol=1e-10))
    
    # Extract steady state values (last time point)
    steady_state = {
        'P_L': result.expect[0][-1],
        'P_R': result.expect[1][-1],
        'sigma_z': result.expect[2][-1],
        'sigma_x': result.expect[3][-1],
        'sigma_y': result.expect[4][-1],
        'coherence': np.sqrt(result.expect[3][-1]**2 + result.expect[4][-1]**2),
        'enantiomeric_excess': abs(result.expect[0][-1] - result.expect[1][-1])
    }
    
    return steady_state, result

def parameter_sweep_analysis():
    """
    Comprehensive parameter sweep for homochirality dynamics
    """
    print("📊 RUNNING PARAMETER SWEEP ANALYSIS")
    print("=" * 50)
    
    # Parameter ranges
    preferences = np.linspace(-1, 1, 21)  # Environmental preference
    gamma_selects = [0.1, 0.5, 1.0, 2.0]  # Selection strengths
    gamma_dephases = [0.0, 0.1, 0.5, 1.0]  # Dephasing rates
    b_values = [0.01, 0.1, 0.5, 1.0]      # Tunneling rates
    
    results = []
    
    # Test different Hamiltonians
    Hamiltonians = {
        'Energy splitting only': 0.5 * 1.0 * sigma_z,
        'Small tunneling': 0.5 * 1.0 * sigma_z + 0.5 * 0.1 * sigma_x,
        'Large tunneling': 0.5 * 1.0 * sigma_z + 0.5 * 1.0 * sigma_x,
        'No energy splitting': 0.5 * 1.0 * sigma_x
    }
    
    for H_name, H in Hamiltonians.items():
        for gamma_select in gamma_selects:
            for gamma_dephase in gamma_dephases:
                for b in b_values:
                    # Update Hamiltonian with current b value
                    if 'tunneling' in H_name:
                        H = 0.5 * 1.0 * sigma_z + 0.5 * b * sigma_x
                    
                    for preference in preferences:
                        c_ops = create_competitive_lindbladian(gamma_select, preference, gamma_dephase)
                        steady_state, _ = analyze_steady_state(H, c_ops)
                        
                        results.append({
                            'Hamiltonian': H_name,
                            'gamma_select': gamma_select,
                            'gamma_dephase': gamma_dephase,
                            'preference': preference,
                            'b': b,
                            'P_L': steady_state['P_L'],
                            'P_R': steady_state['P_R'],
                            'sigma_z': steady_state['sigma_z'],
                            'coherence': steady_state['coherence'],
                            'enantiomeric_excess': steady_state['enantiomeric_excess'],
                            'homochiral': steady_state['P_L'] > 0.99 or steady_state['P_R'] > 0.99
                        })
    
    return pd.DataFrame(results)

def plot_parameter_sweep_results(df):
    """
    Create comprehensive visualization of parameter sweep results
    """
    print("\n📈 GENERATING ANALYSIS PLOTS")
    print("=" * 50)
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Preference vs Enantiomeric Excess for different selection strengths
    plt.subplot(3, 4, 1)
    for gamma_select in df['gamma_select'].unique():
        mask = (df['gamma_select'] == gamma_select) & (df['gamma_dephase'] == 0) & (df['b'] == 0.1)
        subset = df[mask].groupby('preference')['enantiomeric_excess'].mean()
        plt.plot(subset.index, subset.values, 'o-', label=f'γ_select={gamma_select}', markersize=3)
    
    plt.axhline(y=0.98, color='r', linestyle='--', alpha=0.7, label='Homochiral threshold')
    plt.xlabel('Environmental Preference')
    plt.ylabel('Enantiomeric Excess')
    plt.title('Selection Strength Effect\n(No dephasing, b=0.1)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Dephasing effect on homochirality
    plt.subplot(3, 4, 2)
    for gamma_dephase in df['gamma_dephase'].unique():
        mask = (df['gamma_dephase'] == gamma_dephase) & (df['gamma_select'] == 1.0) & (df['b'] == 0.1)
        subset = df[mask].groupby('preference')['P_L'].mean()
        plt.plot(subset.index, subset.values, 'o-', label=f'γ_dephase={gamma_dephase}', markersize=3)
    
    plt.axhline(y=0.99, color='r', linestyle='--', alpha=0.7, label='Homochiral')
    plt.xlabel('Environmental Preference')
    plt.ylabel('P_L (p=0.5)')
    plt.title('Dephasing Effect on Homochirality\n(γ_select=1.0, b=0.1)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Tunneling rate effect
    plt.subplot(3, 4, 3)
    for b in df['b'].unique():
        mask = (df['b'] == b) & (df['gamma_select'] == 1.0) & (df['gamma_dephase'] == 0)
        subset = df[mask].groupby('preference')['enantiomeric_excess'].mean()
        plt.plot(subset.index, subset.values, 'o-', label=f'b={b}', markersize=3)
    
    plt.axhline(y=0.98, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Environmental Preference')
    plt.ylabel('Enantiomeric Excess')
    plt.title('Tunneling Rate Effect\n(γ_select=1.0, No dephasing)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Coherence vs Enantiomeric Excess
    plt.subplot(3, 4, 4)
    scatter = plt.scatter(df['coherence'], df['enantiomeric_excess'], 
                         c=df['gamma_dephase'], alpha=0.6, s=20, cmap='viridis')
    plt.colorbar(scatter, label='Dephasing Rate')
    plt.xlabel('Quantum Coherence')
    plt.ylabel('Enantiomeric Excess')
    plt.title('Coherence vs Homochirality')
    plt.grid(True, alpha=0.3)
    
    # 5. Phase diagram: Selection vs Preference
    plt.subplot(3, 4, 5)
    gamma_range = np.linspace(0.1, 2.0, 20)
    pref_range = np.linspace(0, 1, 20)
    
    homochiral_region = np.zeros((len(gamma_range), len(pref_range)))
    
    for i, gamma in enumerate(gamma_range):
        for j, pref in enumerate(pref_range):
            mask = (np.abs(df['gamma_select'] - gamma) < 0.1) & (np.abs(df['preference'] - pref) < 0.05)
            if np.any(mask):
                homochiral_region[i,j] = df[mask]['homochiral'].mean()
    
    plt.contourf(pref_range, gamma_range, homochiral_region, levels=20, cmap='RdYlGn')
    plt.colorbar(label='Homochirality Probability')
    plt.xlabel('Environmental Preference')
    plt.ylabel('Selection Strength')
    plt.title('Phase Diagram: Homochirality Region')
    
    # 6. Hamiltonian comparison
    plt.subplot(3, 4, 6)
    for H_name in df['Hamiltonian'].unique():
        mask = (df['Hamiltonian'] == H_name) & (df['gamma_select'] == 1.0) & (df['gamma_dephase'] == 0)
        subset = df[mask].groupby('preference')['enantiomeric_excess'].mean()
        plt.plot(subset.index, subset.values, 'o-', label=H_name, markersize=3)
    
    plt.axhline(y=0.98, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Environmental Preference')
    plt.ylabel('Enantiomeric Excess')
    plt.title('Hamiltonian Comparison\n(γ_select=1.0, No dephasing)')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # 7. Dephasing vs Coherence
    plt.subplot(3, 4, 7)
    for gamma_select in [0.5, 1.0, 2.0]:
        mask = (df['gamma_select'] == gamma_select) & (df['preference'] == 0.8) & (df['b'] == 0.1)
        subset = df[mask].groupby('gamma_dephase')['coherence'].mean()
        plt.semilogy(subset.index, subset.values, 'o-', label=f'γ_select={gamma_select}', markersize=4)
    
    plt.xlabel('Dephasing Rate')
    plt.ylabel('Quantum Coherence (log)')
    plt.title('Dephasing Destroys Coherence\n(p=0.8, b=0.1)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 8. Success rate by parameter
    plt.subplot(3, 4, 8)
    param_groups = {
        'Weak Selection\n(γ<0.5)': df[df['gamma_select'] < 0.5],
        'Medium Selection\n(0.5≤γ<1.5)': df[(df['gamma_select'] >= 0.5) & (df['gamma_select'] < 1.5)],
        'Strong Selection\n(γ≥1.5)': df[df['gamma_select'] >= 1.5]
    }
    
    success_rates = [group['homochiral'].mean() * 100 for group in param_groups.values()]
    plt.bar(param_groups.keys(), success_rates, color=['lightcoral', 'lightblue', 'lightgreen'])
    plt.ylabel('Homochirality Success Rate (%)')
    plt.title('Selection Strength Effectiveness')
    plt.grid(True, alpha=0.3)
    
    # 9. Time dynamics example
    plt.subplot(3, 4, 9)
    H = 0.5 * 1.0 * sigma_z + 0.5 * 0.1 * sigma_x
    c_ops = create_competitive_lindbladian(1.0, 0.8, 0.1)
    _, result = analyze_steady_state(H, c_ops)
    
    tlist = np.linspace(0, 50, 500)
    result_dyn = mesolve(H, (L+R).unit(), tlist, c_ops, [P_L, P_R, sigma_x, sigma_y])
    
    plt.plot(tlist, result_dyn.expect[0], 'b-', label='P_L', linewidth=2)
    plt.plot(tlist, result_dyn.expect[1], 'r-', label='P_R', linewidth=2)
    plt.plot(tlist, np.sqrt(np.array(result_dyn.expect[2])**2 + np.array(result_dyn.expect[3])**2), 
             'g--', label='Coherence', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Population / Coherence')
    plt.title('Example Dynamics\n(γ_select=1.0, p=0.8, γ_dephase=0.1)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 10. Optimal parameter combinations
    plt.subplot(3, 4, 10)
    optimal_cases = df[df['homochiral'] == True]
    if len(optimal_cases) > 0:
        param_combinations = optimal_cases.groupby(['gamma_select', 'gamma_dephase']).size().unstack(fill_value=0)
        plt.imshow(param_combinations.values, cmap='YlOrRd', aspect='auto')
        plt.colorbar(label='Number of Successful Cases')
        plt.xlabel('Dephasing Rate')
        plt.ylabel('Selection Strength')
        plt.xticks(range(len(param_combinations.columns)), [f'{x:.1f}' for x in param_combinations.columns])
        plt.yticks(range(len(param_combinations.index)), [f'{x:.1f}' for x in param_combinations.index])
        plt.title('Optimal Parameter Combinations')
    
    # 11. Critical preference threshold
    plt.subplot(3, 4, 11)
    critical_thresholds = []
    gamma_selects_sorted = sorted(df['gamma_select'].unique())
    
    for gamma_select in gamma_selects_sorted:
        mask = (df['gamma_select'] == gamma_select) & (df['gamma_dephase'] == 0) & (df['b'] == 0.1)
        subset = df[mask]
        homochiral_cases = subset[subset['homochiral'] == True]
        if len(homochiral_cases) > 0:
            critical_pref = homochiral_cases['preference'].min()
            critical_thresholds.append(critical_pref)
        else:
            critical_thresholds.append(np.nan)
    
    plt.plot(gamma_selects_sorted, critical_thresholds, 'bo-', linewidth=2, markersize=6)
    plt.xlabel('Selection Strength')
    plt.ylabel('Critical Preference')
    plt.title('Minimum Preference for Homochirality\n(No dephasing, b=0.1)')
    plt.grid(True, alpha=0.3)
    
    # 12. Summary statistics
    plt.subplot(3, 4, 12)
    stats = {
        'Total Simulations': len(df),
        'Homochiral Cases': df['homochiral'].sum(),
        'Success Rate': df['homochiral'].mean() * 100,
        'Avg Enantiomeric Excess': df['enantiomeric_excess'].mean(),
        'Avg Coherence': df['coherence'].mean()
    }
    
    plt.bar(range(len(stats)), list(stats.values()), color='lightblue')
    plt.xticks(range(len(stats)), list(stats.keys()), rotation=45, ha='right')
    plt.ylabel('Value')
    plt.title('Summary Statistics')
    
    for i, v in enumerate(stats.values()):
        plt.text(i, v + 0.1, f'{v:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lindbladian_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def print_key_insights(df):
    """
    Print key insights from the parameter sweep
    """
    print("\n" + "="*70)
    print("🔑 KEY INSIGHTS FROM PARAMETER SWEEP")
    print("="*70)
    
    # Best performing parameters
    best_cases = df[df['homochiral'] == True]
    if len(best_cases) > 0:
        best_case = best_cases.loc[best_cases['enantiomeric_excess'].idxmax()]
        
        print(f"🏆 BEST PERFORMING PARAMETERS:")
        print(f"  • Hamiltonian: {best_case['Hamiltonian']}")
        print(f"  • Selection strength: γ_select = {best_case['gamma_select']:.2f}")
        print(f"  • Dephasing rate: γ_dephase = {best_case['gamma_dephase']:.2f}")
        print(f"  • Environmental preference: p = {best_case['preference']:.2f}")
        print(f"  • Tunneling rate: b = {best_case['b']:.2f}")
        print(f"  • Achieved enantiomeric excess: {best_case['enantiomeric_excess']:.4f}")
        print(f"  • Final P_L: {best_case['P_L']:.4f}")
    
    # Success rates by parameter
    print(f"\n📊 SUCCESS RATES:")
    print(f"  Overall homochirality rate: {df['homochiral'].mean()*100:.1f}%")
    print(f"  With strong selection (γ≥1.0): {df[df['gamma_select']>=1.0]['homochiral'].mean()*100:.1f}%")
    print(f"  With dephasing (γ_dephase>0): {df[df['gamma_dephase']>0]['homochiral'].mean()*100:.1f}%")
    print(f"  With high preference (p≥0.8): {df[df['preference']>=0.8]['homochiral'].mean()*100:.1f}%")
    
    # Parameter correlations
    print(f"\n📈 PARAMETER CORRELATIONS:")
    corr_matrix = df[['gamma_select', 'gamma_dephase', 'preference', 'b', 'enantiomeric_excess', 'coherence']].corr()
    print(f"  Selection vs Enantiomeric excess: {corr_matrix.loc['gamma_select', 'enantiomeric_excess']:.3f}")
    print(f"  Dephasing vs Coherence: {corr_matrix.loc['gamma_dephase', 'coherence']:.3f}")
    print(f"  Preference vs Enantiomeric excess: {corr_matrix.loc['preference', 'enantiomeric_excess']:.3f}")
    
    # Critical thresholds
    print(f"\n🎯 CRITICAL THRESHOLDS FOR HOMOCHIRALITY:")
    strong_selection_cases = df[df['gamma_select'] >= 1.0]
    if len(strong_selection_cases) > 0:
        critical_pref = strong_selection_cases[strong_selection_cases['homochiral'] == True]['preference'].min()
        print(f"  Minimum preference needed (γ_select≥1.0): {critical_pref:.2f}")
    
    print(f"\n💡 PRACTICAL RECOMMENDATIONS:")
    print(f"  1. Use selection strength γ_select ≥ 1.0")
    print(f"  2. Maintain environmental preference p ≥ 0.8") 
    print(f"  3. Apply moderate dephasing (γ_dephase ≈ 0.1-0.5) to suppress coherence")
    print(f"  4. Keep tunneling rate b ≤ 0.5 to avoid racemization")
    print(f"  5. Target enantiomeric excess > 0.98 for practical homochirality")

# Run the complete analysis
if __name__ == "__main__":
    # Perform parameter sweep
    df = parameter_sweep_analysis()
    
    # Generate comprehensive plots
    df = plot_parameter_sweep_results(df)
    
    # Print key insights
    print_key_insights(df)
    
    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"📁 Results saved to: lindbladian_analysis_comprehensive.png")
    print(f"📊 Total simulations: {len(df)}")
    print(f"🎯 Homochiral cases found: {df['homochiral'].sum()}")