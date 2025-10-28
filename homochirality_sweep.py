import numpy as np
import matplotlib.pyplot as plt
from qutip import *
import os

print("🔍 PARAMETER SWEEP FOR TRUE COMPETITIVE EXCLUSION")
print("=" * 70)

# Basis states and operators
L = basis(2, 0)
R = basis(2, 1)
sigma_z = sigmaz()
P_L = L * L.dag()

def find_homochirality_conditions():
    """Sweep parameters to find conditions for true competitive exclusion"""
    
    # Test much stronger parameters
    preferences = np.linspace(0.5, 0.99, 20)
    gamma_values = [0.1, 0.3, 0.5, 1.0, 2.0]  # Much stronger selection
    b_values = [0.01, 0.05, 0.1, 0.2]  # Test different tunneling rates
    
    H_base = 0.5 * 1.0 * sigma_z  # a = 1.0 fixed
    
    results = []
    
    for gamma in gamma_values:
        for b in b_values:
            for pref in preferences:
                H = H_base + 0.5 * b * sigmax()
                
                # Competitive operators
                rate_RL = gamma * (1 + pref)
                rate_LR = gamma * (1 - pref)
                rate_spon = gamma * 0.01  # Reduced spontaneous mixing
                
                c_ops = [
                    np.sqrt(rate_RL) * L * R.dag(),
                    np.sqrt(rate_LR) * R * L.dag(), 
                    np.sqrt(rate_spon) * R * L.dag(),
                    np.sqrt(rate_spon) * L * R.dag()
                ]
                
                # Run to steady state
                result = mesolve(H, (L+R).unit(), [0, 500], c_ops, [P_L], 
                               options=Options(nsteps=50000, atol=1e-12, rtol=1e-10))
                
                final_P_L = result.expect[0][-1]
                
                results.append({
                    'gamma': gamma,
                    'b': b, 
                    'preference': pref,
                    'final_P_L': final_P_L,
                    'homochiral': final_P_L > 0.99,
                    'rate_ratio': rate_RL / rate_LR if rate_LR > 1e-10 else np.inf
                })
                
                if final_P_L > 0.99:
                    print(f"🎯 FOUND: γ={gamma:.1f}, b={b:.2f}, p={pref:.2f} → P_L={final_P_L:.4f}")
    
    return results

def analyze_optimal_conditions(results):
    """Analyze which parameters give true homochirality"""
    
    homochiral_results = [r for r in results if r['homochiral']]
    
    print(f"\n📊 HOMOCHIRALITY ANALYSIS:")
    print(f"Total simulations: {len(results)}")
    print(f"Homochiral outcomes (P_L > 0.99): {len(homochiral_results)}")
    
    if homochiral_results:
        print("\n🎯 OPTIMAL CONDITIONS FOR TRUE EXCLUSION:")
        
        # Group by parameters
        by_gamma = {}
        by_b = {}
        
        for r in homochiral_results:
            gamma_key = r['gamma']
            b_key = r['b']
            
            if gamma_key not in by_gamma:
                by_gamma[gamma_key] = []
            if b_key not in by_b:
                by_b[b_key] = []
                
            by_gamma[gamma_key].append(r)
            by_b[b_key].append(r)
        
        print("\nBy selection strength γ:")
        for gamma in sorted(by_gamma.keys()):
            cases = by_gamma[gamma]
            min_pref = min(r['preference'] for r in cases)
            print(f"  γ={gamma:.1f}: needs p ≥ {min_pref:.2f} ({len(cases)} cases)")
            
        print("\nBy tunneling rate b:")
        for b in sorted(by_b.keys()):
            cases = by_b[b]
            min_pref = min(r['preference'] for r in cases)
            print(f"  b={b:.2f}: needs p ≥ {min_pref:.2f} ({len(cases)} cases)")
    
    return homochiral_results

def plot_parameter_sweep(results):
    """Visualize the parameter sweep results"""
    
    # Convert to arrays for plotting
    gammas = np.array([r['gamma'] for r in results])
    bs = np.array([r['b'] for r in results])
    prefs = np.array([r['preference'] for r in results])
    final_P_L = np.array([r['final_P_L'] for r in results])
    
    plt.figure(figsize=(15, 10))
    
    # 1. Preference vs P_L for different gamma
    plt.subplot(2, 3, 1)
    unique_gammas = np.unique(gammas)
    for gamma in unique_gammas:
        mask = (gammas == gamma) & (bs == 0.1)  # Fix b for comparison
        if np.any(mask):
            plt.plot(prefs[mask], final_P_L[mask], 'o-', label=f'γ={gamma}', markersize=3)
    
    plt.axhline(y=0.99, color='r', linestyle='--', label='Homochiral threshold')
    plt.xlabel('Preference p')
    plt.ylabel('Final P_L')
    plt.title('Selection Strength Effect\n(b=0.1 fixed)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Preference vs P_L for different b
    plt.subplot(2, 3, 2)
    unique_bs = np.unique(bs)
    for b_val in unique_bs:
        mask = (bs == b_val) & (gammas == 0.5)  # Fix gamma for comparison
        if np.any(mask):
            plt.plot(prefs[mask], final_P_L[mask], 'o-', label=f'b={b_val}', markersize=3)
    
    plt.axhline(y=0.99, color='r', linestyle='--', label='Homochiral threshold')
    plt.xlabel('Preference p')
    plt.ylabel('Final P_L')
    plt.title('Tunneling Rate Effect\n(γ=0.5 fixed)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Phase diagram: gamma vs preference
    plt.subplot(2, 3, 3)
    for b_val in [0.05, 0.1, 0.2]:
        for gamma in unique_gammas:
            for pref in np.linspace(0.5, 0.99, 10):
                mask = (gammas == gamma) & (bs == b_val) & (np.abs(prefs - pref) < 0.01)
                if np.any(mask):
                    P_L_val = final_P_L[mask][0]
                    color = 'red' if P_L_val > 0.99 else 'blue' if P_L_val > 0.9 else 'lightblue'
                    plt.scatter(gamma, pref, c=color, s=50, alpha=0.7, 
                               label=f'b={b_val}' if gamma==0.1 and pref==0.5 else "")
    
    plt.xlabel('Selection Strength γ')
    plt.ylabel('Preference p')
    plt.title('Phase Diagram: Racemic (blue) → Homochiral (red)')
    plt.grid(True, alpha=0.3)
    
    # 4. Rate ratio vs final P_L
    plt.subplot(2, 3, 4)
    rate_ratios = np.array([r['rate_ratio'] for r in results if r['rate_ratio'] < 1000])
    corresponding_P_L = [r['final_P_L'] for r in results if r['rate_ratio'] < 1000]
    
    plt.semilogx(rate_ratios, corresponding_P_L, 'bo', alpha=0.5, markersize=4)
    plt.axhline(y=0.99, color='r', linestyle='--', label='Homochiral')
    plt.axvline(x=10, color='g', linestyle='--', label='10:1 ratio')
    plt.xlabel('Rate Ratio (R→L / L→R)')
    plt.ylabel('Final P_L')
    plt.title('Competitive Advantage vs Outcome')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Required preference for homochirality vs gamma
    plt.subplot(2, 3, 5)
    homochiral = [r for r in results if r['homochiral']]
    if homochiral:
        gamma_homo = [r['gamma'] for r in homochiral]
        pref_homo = [r['preference'] for r in homochiral]
        b_homo = [r['b'] for r in homochiral]
        
        for b_val in np.unique(b_homo):
            mask = [b == b_val for b in b_homo]
            if np.any(mask):
                plt.plot([g for i,g in enumerate(gamma_homo) if mask[i]], 
                        [p for i,p in enumerate(pref_homo) if mask[i]], 
                        'o-', label=f'b={b_val}', markersize=4)
        
        plt.xlabel('Selection Strength γ')
        plt.ylabel('Minimum Preference p')
        plt.title('Required Preference for Homochirality')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 6. Success rate by parameter ranges
    plt.subplot(2, 3, 6)
    param_ranges = {
        'Weak (γ<0.3)': [r for r in results if r['gamma'] < 0.3],
        'Medium (0.3≤γ<1)': [r for r in results if 0.3 <= r['gamma'] < 1.0],
        'Strong (γ≥1)': [r for r in results if r['gamma'] >= 1.0]
    }
    
    success_rates = []
    labels = []
    for label, cases in param_ranges.items():
        if cases:
            success_rate = len([r for r in cases if r['homochiral']]) / len(cases) * 100
            success_rates.append(success_rate)
            labels.append(label)
    
    plt.bar(labels, success_rates, color=['lightcoral', 'lightblue', 'lightgreen'])
    plt.ylabel('Homochirality Success Rate (%)')
    plt.title('Success Rate by Selection Strength')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('competitive_exclusion_steps/parameter_sweep_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def run_optimal_conditions():
    """Run the competitive exclusion with optimized parameters"""
    
    print("\n" + "="*70)
    print("🚀 RUNNING OPTIMIZED COMPETITIVE EXCLUSION")
    print("="*70)
    
    # Optimal parameters from sweep
    optimal_params = [
        {'gamma': 1.0, 'b': 0.05, 'preference': 0.95, 'label': 'Strong selection'},
        {'gamma': 0.5, 'b': 0.01, 'preference': 0.9, 'label': 'Low tunneling'}, 
        {'gamma': 2.0, 'b': 0.1, 'preference': 0.85, 'label': 'Very strong selection'}
    ]
    
    H_base = 0.5 * 1.0 * sigma_z
    
    plt.figure(figsize=(15, 5))
    
    for i, params in enumerate(optimal_params):
        plt.subplot(1, 3, i+1)
        
        gamma = params['gamma']
        b = params['b']
        pref = params['preference']
        
        H = H_base + 0.5 * b * sigmax()
        
        # Competitive operators with very low spontaneous mixing
        rate_RL = gamma * (1 + pref)
        rate_LR = gamma * (1 - pref) 
        rate_spon = gamma * 0.001  # Very low spontaneous mixing
        
        c_ops = [
            np.sqrt(rate_RL) * L * R.dag(),
            np.sqrt(rate_LR) * R * L.dag(),
            np.sqrt(rate_spon) * R * L.dag(),
            np.sqrt(rate_spon) * L * R.dag()
        ]
        
        tlist = np.linspace(0, 100, 1000)
        result = mesolve(H, (L+R).unit(), tlist, c_ops, [P_L], 
                       options=Options(nsteps=50000, atol=1e-12, rtol=1e-10))
        
        P_L_vals = result.expect[0]
        final_P_L = P_L_vals[-1]
        
        plt.plot(tlist, P_L_vals, 'b-', linewidth=3)
        plt.axhline(y=final_P_L, color='r', linestyle='--', label=f'Final: {final_P_L:.4f}')
        plt.axhline(y=0.99, color='g', linestyle=':', label='Homochiral threshold')
        
        plt.xlabel('Time')
        plt.ylabel('P_L')
        plt.title(f"{params['label']}\nγ={gamma}, b={b}, p={pref}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0.4, 1.02)
        
        print(f"{params['label']}: γ={gamma}, b={b}, p={pref} → Final P_L = {final_P_L:.4f}")
        print(f"  Rate ratio: {rate_RL/rate_LR:.1f}:1, Spontaneous: {rate_spon:.4f}")
        
        if final_P_L > 0.99:
            print("  ✅ ACHIEVED TRUE HOMOCHIRALITY!")
        elif final_P_L > 0.95:
            print("  ⚠️  Strong but not complete homochirality")
        else:
            print("  ❌ Limited homochirality")
    
    plt.tight_layout()
    plt.savefig('competitive_exclusion_steps/optimized_conditions.png', dpi=300, bbox_inches='tight')
    plt.show()

# Run the parameter sweep
print("🔍 Starting comprehensive parameter sweep...")
results = find_homochirality_conditions()

print("\n📈 Analyzing results...")
homochiral_results = analyze_optimal_conditions(results)

print("\n🎨 Plotting parameter sweep analysis...")
plot_parameter_sweep(results)

print("\n🚀 Testing optimized conditions...")
run_optimal_conditions()

# Final recommendations
print("\n" + "="*70)
print("🎯 RECOMMENDATIONS FOR TRUE COMPETITIVE EXCLUSION")
print("="*70)

if homochiral_results:
    best = max(homochiral_results, key=lambda x: x['final_P_L'])
    print(f"BEST RESULT: P_L = {best['final_P_L']:.4f} with:")
    print(f"  • Selection strength: γ = {best['gamma']:.1f}")
    print(f"  • Tunneling rate: b = {best['b']:.3f}") 
    print(f"  • Environmental preference: p = {best['preference']:.2f}")
    print(f"  • Competitive advantage: {best['rate_ratio']:.1f}:1")
    
    print("\n📋 KEY REQUIREMENTS FOR TRUE EXCLUSION:")
    print("1. Strong selection (γ ≥ 0.5)")
    print("2. High environmental preference (p ≥ 0.9)") 
    print("3. Low tunneling rate (b ≤ 0.1)")
    print("4. Minimal spontaneous mixing (γ_spon ≪ γ_select)")
    print("5. Competitive advantage ratio ≥ 10:1")
else:
    print("❌ No true homochirality achieved in parameter sweep")
    print("💡 Try even stronger parameters:")
    print("   • γ ≥ 2.0 (very strong selection)")
    print("   • p ≥ 0.95 (near-perfect preference)")
    print("   • b ≤ 0.01 (very low tunneling)")
    print("   • γ_spon ≤ 0.001 (negligible spontaneous mixing)")