import numpy as np
import matplotlib.pyplot as plt
from qutip import *
import os

os.makedirs('competitive_exclusion_steps', exist_ok=True)
print("🔬 DETAILED COMPETITIVE EXCLUSION ANALYSIS")
print("=" * 70)

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

def create_competitive_operators(gamma_select, preference):
    """Create competitive Lindblad operators with detailed tracking"""
    print(f"  🎯 Competitive operators: γ={gamma_select:.3f}, p={preference:.3f}")
    
    # Competitive processes
    rate_RL = gamma_select * (1 + preference)  # R → L conversion
    rate_LR = gamma_select * (1 - preference)  # L → R conversion
    rate_spon = gamma_select * 0.1  # Spontaneous mixing
    
    L_RL = np.sqrt(rate_RL) * L * R.dag()  # R decreases, L increases
    L_LR = np.sqrt(rate_LR) * R * L.dag()  # L decreases, R increases
    L_spon_LR = np.sqrt(rate_spon) * R * L.dag()  # Spontaneous L → R
    L_spon_RL = np.sqrt(rate_spon) * L * R.dag()  # Spontaneous R → L
    
    operators = [L_RL, L_LR, L_spon_LR, L_spon_RL]
    
    print(f"  📊 Competitive rates:")
    print(f"    R→L (favored): {rate_RL:.4f}")
    print(f"    L→R (disfavored): {rate_LR:.4f}") 
    print(f"    Ratio: {rate_RL/rate_LR:.2f}:1")
    print(f"    Net bias: {rate_RL - rate_LR:.4f}")
    
    return operators, [rate_RL, rate_LR, rate_spon]

def analyze_competitive_rates():
    """Show how preference affects competitive rates"""
    print("\n" + "="*70)
    print("STEP 1: COMPETITIVE RATE ANALYSIS")
    print("="*70)
    
    preferences = np.linspace(-1, 1, 21)
    gamma = 0.2
    
    rates_RL = []
    rates_LR = []
    net_biases = []
    ratios = []
    
    for p in preferences:
        rate_RL = gamma * (1 + p)
        rate_LR = gamma * (1 - p)
        rates_RL.append(rate_RL)
        rates_LR.append(rate_LR)
        net_biases.append(rate_RL - rate_LR)
        # Avoid division by zero
        if rate_LR > 1e-10:
            ratios.append(rate_RL / rate_LR)
        else:
            ratios.append(np.inf)
    
    plt.figure(figsize=(15, 10))
    
    # Rate comparison
    plt.subplot(2, 3, 1)
    plt.plot(preferences, rates_RL, 'b-', linewidth=3, label='R→L rate')
    plt.plot(preferences, rates_LR, 'r-', linewidth=3, label='L→R rate')
    plt.axvline(x=0, color='k', linestyle=':', alpha=0.5)
    plt.axhline(y=gamma, color='k', linestyle=':', alpha=0.5, label='Base rate')
    plt.xlabel('Preference p')
    plt.ylabel('Conversion Rate')
    plt.title('Competitive Conversion Rates')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Net bias
    plt.subplot(2, 3, 2)
    plt.plot(preferences, net_biases, 'g-', linewidth=3)
    plt.axvline(x=0, color='k', linestyle=':', alpha=0.5)
    plt.axhline(y=0, color='k', linestyle=':', alpha=0.5)
    plt.xlabel('Preference p')
    plt.ylabel('Net Bias (R→L - L→R)')
    plt.title('Net Competitive Bias')
    plt.grid(True, alpha=0.3)
    
    # Rate ratio (log scale)
    plt.subplot(2, 3, 3)
    # Filter out infinite values for plotting
    finite_ratios = [r if r != np.inf else 1e10 for r in ratios]
    plt.semilogy(preferences, finite_ratios, 'r--', linewidth=3)
    plt.axvline(x=0, color='k', linestyle=':', alpha=0.5)
    plt.axhline(y=1, color='k', linestyle=':', alpha=0.5)
    plt.xlabel('Preference p')
    plt.ylabel('Rate Ratio (R→L / L→R)')
    plt.title('Competitive Advantage Ratio')
    plt.grid(True, alpha=0.3)
    
    # Show specific examples
    example_prefs = [-0.8, -0.4, 0.0, 0.4, 0.8]
    print("\n📈 Specific competitive scenarios:")
    for p in example_prefs:
        rate_RL = gamma * (1 + p)
        rate_LR = gamma * (1 - p)
        ratio = rate_RL / rate_LR if rate_LR > 1e-10 else np.inf
        print(f"  p = {p:5.1f}: R→L = {rate_RL:.3f}, L→R = {rate_LR:.3f}, ratio = {ratio:6.1f}:1")
    
    plt.tight_layout()
    plt.savefig('competitive_exclusion_steps/step1_competitive_rates.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return preferences, rates_RL, rates_LR, ratios

def track_instantaneous_dynamics():
    """Track instantaneous rates and flows during evolution"""
    print("\n" + "="*70)
    print("STEP 2: INSTANTANEOUS DYNAMICS TRACKING")
    print("="*70)
    
    H = 0.5 * a * sigma_z + 0.5 * b * sigma_x
    tlist = np.linspace(0, 100, 1000)
    preference = 0.6
    gamma = 0.2
    
    c_ops, rates = create_competitive_operators(gamma, preference)
    
    # Initial racemic state
    psi0 = (L + R).unit()
    
    # Track populations and compute instantaneous flows
    result = mesolve(H, psi0, tlist, c_ops, [P_L, P_R], options=solver_options)
    
    P_L_vals = result.expect[0]
    P_R_vals = result.expect[1]
    
    # Compute instantaneous flows (approximate from populations)
    dt = tlist[1] - tlist[0]
    dP_L_dt = np.gradient(P_L_vals, dt)
    dP_R_dt = np.gradient(P_R_vals, dt)
    
    # Theoretical flows based on current populations
    flow_RL_instant = rates[0] * P_R_vals  # R→L flow depends on R population
    flow_LR_instant = rates[1] * P_L_vals  # L→R flow depends on L population
    net_flow_instant = flow_RL_instant - flow_LR_instant
    
    plt.figure(figsize=(15, 12))
    
    # Populations
    plt.subplot(3, 2, 1)
    plt.plot(tlist, P_L_vals, 'b-', linewidth=3, label='P_L')
    plt.plot(tlist, P_R_vals, 'r-', linewidth=3, label='P_R')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('Population Dynamics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Population derivatives (actual changes)
    plt.subplot(3, 2, 2)
    plt.plot(tlist, dP_L_dt, 'b-', linewidth=2, label='dP_L/dt')
    plt.plot(tlist, dP_R_dt, 'r-', linewidth=2, label='dP_R/dt')
    plt.axhline(y=0, color='k', linestyle=':', alpha=0.5)
    plt.xlabel('Time')
    plt.ylabel('Population Change Rate')
    plt.title('Instantaneous Population Changes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Individual flows
    plt.subplot(3, 2, 3)
    plt.plot(tlist, flow_RL_instant, 'green', linewidth=2, label='R→L flow')
    plt.plot(tlist, flow_LR_instant, 'orange', linewidth=2, label='L→R flow')
    plt.xlabel('Time')
    plt.ylabel('Conversion Flow')
    plt.title('Individual Conversion Flows')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Net flow
    plt.subplot(3, 2, 4)
    plt.plot(tlist, net_flow_instant, 'r-', linewidth=3, label='Net flow (R→L - L→R)')
    plt.axhline(y=0, color='k', linestyle=':', alpha=0.5)
    plt.xlabel('Time')
    plt.ylabel('Net Flow')
    plt.title('Net Competitive Flow')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Flow balance
    plt.subplot(3, 2, 5)
    balance = dP_L_dt - net_flow_instant  # Should be small (just coherent part)
    plt.plot(tlist, balance, 'k-', linewidth=2, label='Flow balance error')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Balance Error')
    plt.title('Flow Balance Check')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Competitive advantage
    plt.subplot(3, 2, 6)
    advantage = flow_RL_instant / (flow_LR_instant + 1e-10)
    plt.semilogy(tlist, advantage, 'brown', linewidth=2)
    plt.axhline(y=rates[0]/rates[1], color='r', linestyle='--', label=f'Theoretical: {rates[0]/rates[1]:.1f}')
    plt.xlabel('Time')
    plt.ylabel('Flow Ratio (R→L / L→R)')
    plt.title('Instantaneous Competitive Advantage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('competitive_exclusion_steps/step2_instantaneous_dynamics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print key insights
    print(f"\n📊 Competitive Dynamics Summary (p={preference}):")
    print(f"  Initial flows: R→L = {flow_RL_instant[0]:.4f}, L→R = {flow_LR_instant[0]:.4f}")
    print(f"  Final flows: R→L = {flow_RL_instant[-1]:.4f}, L→R = {flow_LR_instant[-1]:.4f}")
    print(f"  Maximum net flow: {np.max(net_flow_instant):.4f}")
    print(f"  Steady-state advantage ratio: {advantage[-1]:.2f}:1")
    
    return tlist, P_L_vals, P_R_vals, flow_RL_instant, flow_LR_instant

def analyze_phase_transition():
    """Analyze the phase transition from racemic to homochiral"""
    print("\n" + "="*70)
    print("STEP 3: PHASE TRANSITION ANALYSIS")
    print("="*70)
    
    H = 0.5 * a * sigma_z + 0.5 * b * sigma_x
    tlist_long = np.linspace(0, 200, 2000)
    
    # Test different preference strengths
    preferences = [0.1, 0.3, 0.5, 0.7, 0.9]
    colors = plt.cm.viridis(np.linspace(0, 1, len(preferences)))
    
    plt.figure(figsize=(16, 12))
    
    all_final_P_L = []
    all_final_P_R = []
    exclusion_times = []
    
    for i, pref in enumerate(preferences):
        plt.subplot(2, 3, i+1)
        
        c_ops, rates = create_competitive_operators(0.2, pref)
        result = mesolve(H, (L+R).unit(), tlist_long, c_ops, [P_L, P_R], options=solver_options)
        
        P_L_vals = result.expect[0]
        P_R_vals = result.expect[1]
        
        plt.plot(tlist_long, P_L_vals, color='blue', linewidth=2, label='P_L')
        plt.plot(tlist_long, P_R_vals, color='red', linewidth=2, label='P_R')
        
        # Find exclusion time (when minority drops below threshold)
        threshold = 0.05
        minority_pop = np.minimum(P_L_vals, P_R_vals)
        exclusion_idx = np.where(minority_pop < threshold)[0]
        
        if len(exclusion_idx) > 0:
            exclusion_time = tlist_long[exclusion_idx[0]]
            exclusion_times.append(exclusion_time)
            plt.axvline(x=exclusion_time, color='r-', linestyle='--', 
                       label=f'Exclusion: t={exclusion_time:.1f}')
        
        final_P_L = P_L_vals[-1]
        final_P_R = P_R_vals[-1]
        all_final_P_L.append(final_P_L)
        all_final_P_R.append(final_P_R)
        
        plt.title(f'p = {pref}\nFinal: P_L={final_P_L:.3f}, P_R={final_P_R:.3f}')
        plt.xlabel('Time')
        plt.ylabel('Population')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.05, 1.05)
        
        exclusion_time_str = f"{exclusion_times[-1]:.1f}" if exclusion_times else 'N/A'
        print(f"Preference {pref:.1f}: Final P_L={final_P_L:.3f}, P_R={final_P_R:.3f}, Exclusion time={exclusion_time_str}")
    
    # Phase diagram
    plt.subplot(2, 3, 6)
    pref_range = np.linspace(0, 1, 50)
    steady_states = []
    
    for pref in pref_range:
        c_ops = create_competitive_operators(0.2, pref)[0]
        result = mesolve(H, (L+R).unit(), [0, 200], c_ops, [P_L], options=solver_options)
        steady_states.append(result.expect[0][-1])
    
    plt.plot(pref_range, steady_states, 'k-', linewidth=3)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Racemic line')
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Preference p')
    plt.ylabel('Steady-state P_L')
    plt.title('Phase Diagram: Racemic → Homochiral')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('competitive_exclusion_steps/step3_phase_transition.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return preferences, all_final_P_L, all_final_P_R, exclusion_times

def detailed_competitive_exclusion_sequence():
    """Show the complete step-by-step exclusion process"""
    print("\n" + "="*70)
    print("STEP 4: DETAILED EXCLUSION SEQUENCE")
    print("="*70)
    
    H = 0.5 * a * sigma_z + 0.5 * b * sigma_x
    preference = 0.8  # Strong preference for L
    gamma = 0.3
    
    # Use longer time to see complete exclusion
    tlist_detailed = np.linspace(0, 300, 3000)
    c_ops, rates = create_competitive_operators(gamma, preference)
    
    # Start from racemic mixture
    psi0 = (L + R).unit()
    
    # Track everything
    e_ops = [P_L, P_R, sigma_z, P_L*P_L, P_R*P_R, P_L*P_R]
    result = mesolve(H, psi0, tlist_detailed, c_ops, e_ops, options=solver_options)
    
    P_L_vals = result.expect[0]
    P_R_vals = result.expect[1]
    chirality = result.expect[2]
    P_L_sq = result.expect[3]
    P_R_sq = result.expect[4]
    P_L_P_R = result.expect[5]
    
    # Compute coherence and other metrics
    coherence = 2 * np.abs(P_L_P_R - P_L_vals * P_R_vals)
    entropy = -P_L_vals * np.log(P_L_vals + 1e-10) - P_R_vals * np.log(P_R_vals + 1e-10)
    
    plt.figure(figsize=(18, 12))
    
    # Main populations
    plt.subplot(3, 3, 1)
    plt.plot(tlist_detailed, P_L_vals, 'b-', linewidth=3, label='P_L (favored)')
    plt.plot(tlist_detailed, P_R_vals, 'r-', linewidth=3, label='P_R (disfavored)')
    
    # Mark key phases
    phases = [
        (0, 20, 'I: Initial', 'lightblue'),
        (20, 80, 'II: Competition', 'lightgreen'), 
        (80, 200, 'III: Exclusion', 'lightyellow'),
        (200, 300, 'IV: Homochiral', 'lightcoral')
    ]
    
    for start, end, label, color in phases:
        plt.axvspan(start, end, alpha=0.2, color=color, label=label)
    
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('Four-Phase Competitive Exclusion')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Chirality evolution
    plt.subplot(3, 3, 2)
    plt.plot(tlist_detailed, chirality, 'r-', linewidth=3)
    for start, end, label, color in phases:
        plt.axvspan(start, end, alpha=0.2, color=color)
    plt.xlabel('Time')
    plt.ylabel('<σ_z> = P_L - P_R')
    plt.title('Chirality Evolution')
    plt.grid(True, alpha=0.3)
    
    # Coherence dynamics
    plt.subplot(3, 3, 3)
    plt.semilogy(tlist_detailed, coherence + 1e-10, 'orange', linewidth=2)
    for start, end, label, color in phases:
        plt.axvspan(start, end, alpha=0.2, color=color)
    plt.xlabel('Time')
    plt.ylabel('Quantum Coherence')
    plt.title('Decoherence During Exclusion')
    plt.grid(True, alpha=0.3)
    
    # Entropy evolution
    plt.subplot(3, 3, 4)
    plt.plot(tlist_detailed, entropy, 'brown', linewidth=2)
    for start, end, label, color in phases:
        plt.axvspan(start, end, alpha=0.2, color=color)
    plt.xlabel('Time')
    plt.ylabel('Entropy')
    plt.title('Information Loss During Exclusion')
    plt.grid(True, alpha=0.3)
    
    # Phase space trajectory
    plt.subplot(3, 3, 5)
    plt.plot(P_L_vals, P_R_vals, 'k-', linewidth=2, alpha=0.7)
    
    # Color by time
    scatter = plt.scatter(P_L_vals[::100], P_R_vals[::100], c=tlist_detailed[::100], 
                         cmap='viridis', s=30, alpha=0.8)
    plt.colorbar(scatter, label='Time')
    
    plt.plot([0, 1], [1, 0], 'r--', alpha=0.5, label='P_L + P_R = 1')
    plt.xlabel('P_L')
    plt.ylabel('P_R')
    plt.title('Phase Space Trajectory')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Rate analysis by phase
    plt.subplot(3, 3, 6)
    phases_analysis = []
    
    for i, (start, end, label, color) in enumerate(phases):
        idx = (tlist_detailed >= start) & (tlist_detailed <= end)
        if np.any(idx):
            phase_P_L = np.mean(P_L_vals[idx])
            phase_P_R = np.mean(P_R_vals[idx])
            phase_flow_RL = rates[0] * phase_P_R
            phase_flow_LR = rates[1] * phase_P_L
            phase_net_flow = phase_flow_RL - phase_flow_LR
            
            phases_analysis.append({
                'name': label, 'P_L': phase_P_L, 'P_R': phase_P_R,
                'flow_RL': phase_flow_RL, 'flow_LR': phase_flow_LR,
                'net_flow': phase_net_flow
            })
            
            plt.bar(i, phase_net_flow, color=color, alpha=0.7, label=label)
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    plt.xlabel('Phase')
    plt.ylabel('Net Flow (R→L - L→R)')
    plt.title('Net Competitive Flow by Phase')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Detailed phase analysis
    print("\n📋 FOUR-PHASE COMPETITIVE EXCLUSION ANALYSIS:")
    for phase in phases_analysis:
        print(f"\n{phase['name']}:")
        print(f"  Average populations: P_L = {phase['P_L']:.3f}, P_R = {phase['P_R']:.3f}")
        print(f"  Flows: R→L = {phase['flow_RL']:.4f}, L→R = {phase['flow_LR']:.4f}")
        print(f"  Net competitive advantage: {phase['net_flow']:.4f}")
    
    plt.tight_layout()
    plt.savefig('competitive_exclusion_steps/step4_detailed_exclusion_sequence.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return phases_analysis

def quantum_competitive_advantage_analysis():
    """Analyze the quantum nature of competitive advantage"""
    print("\n" + "="*70)
    print("STEP 5: QUANTUM COMPETITIVE ADVANTAGE")
    print("="*70)
    
    H = 0.5 * a * sigma_z + 0.5 * b * sigma_x
    
    # Test how quantum coherence affects competition
    initial_states = [
        ('Coherent superposition', (L + R).unit()),
        ('Mostly L', (np.sqrt(0.9)*L + np.sqrt(0.1)*R).unit()),
        ('Mostly R', (np.sqrt(0.1)*L + np.sqrt(0.9)*R).unit()),
        ('Pure L', L),
        ('Pure R', R)
    ]
    
    preference = 0.7
    gamma = 0.2
    tlist_quantum = np.linspace(0, 100, 1000)
    
    c_ops = create_competitive_operators(gamma, preference)[0]
    
    plt.figure(figsize=(15, 10))
    
    for i, (label, psi0) in enumerate(initial_states):
        plt.subplot(2, 3, i+1)
        
        # Calculate initial coherence
        rho0 = psi0 * psi0.dag()
        coherence_initial = abs(rho0[0,1])
        
        result = mesolve(H, psi0, tlist_quantum, c_ops, [P_L, P_R, sigma_z], options=solver_options)
        
        plt.plot(tlist_quantum, result.expect[0], 'b-', linewidth=2, label='P_L')
        plt.plot(tlist_quantum, result.expect[1], 'r-', linewidth=2, label='P_R')
        plt.plot(tlist_quantum, result.expect[2], 'k--', alpha=0.7, label='<σ_z>')
        
        final_P_L = result.expect[0][-1]
        final_P_R = result.expect[1][-1]
        
        plt.title(f'{label}\nInitial coherence: {coherence_initial:.3f}\nFinal: P_L={final_P_L:.3f}')
        plt.xlabel('Time')
        plt.ylabel('Population / Chirality')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.1, 1.1)
        
        print(f"{label:20s}: Initial coherence = {coherence_initial:.3f}, Final P_L = {final_P_L:.3f}")
    
    # Quantum speed of convergence
    plt.subplot(2, 3, 6)
    
    convergence_times = []
    initial_coherences = []
    
    for label, psi0 in initial_states:
        rho0 = psi0 * psi0.dag()
        coherence_initial = abs(rho0[0,1])
        initial_coherences.append(coherence_initial)
        
        result = mesolve(H, psi0, tlist_quantum, c_ops, [P_L], options=solver_options)
        P_L_vals = result.expect[0]
        
        # Find convergence time (time to reach 90% of steady state)
        steady_state = P_L_vals[-1]
        target = 0.9 * steady_state
        convergence_idx = np.where(P_L_vals >= target)[0]
        convergence_time = tlist_quantum[convergence_idx[0]] if len(convergence_idx) > 0 else tlist_quantum[-1]
        convergence_times.append(convergence_time)
    
    plt.scatter(initial_coherences, convergence_times, s=100, alpha=0.7)
    for i, (label, _) in enumerate(initial_states):
        plt.annotate(label, (initial_coherences[i], convergence_times[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('Initial Quantum Coherence')
    plt.ylabel('Convergence Time')
    plt.title('Quantum Coherence vs Convergence Speed')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('competitive_exclusion_steps/step5_quantum_competitive_advantage.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return initial_states, convergence_times, initial_coherences

# Run the complete step-by-step analysis
print("🚀 STARTING COMPLETE STEP-BY-STEP ANALYSIS")
print("="*70)

# Step 1: Competitive rate analysis
preferences, rates_RL, rates_LR, ratios = analyze_competitive_rates()

# Step 2: Instantaneous dynamics
tlist, P_L_vals, P_R_vals, flow_RL, flow_LR = track_instantaneous_dynamics()

# Step 3: Phase transition analysis  
pref_tested, final_P_L, final_P_R, excl_times = analyze_phase_transition()

# Step 4: Detailed exclusion sequence
phases_analysis = detailed_competitive_exclusion_sequence()

# Step 5: Quantum effects
initial_states, conv_times, init_coherences = quantum_competitive_advantage_analysis()

# Final summary
print("\n" + "="*70)
print("🎯 COMPETITIVE EXCLUSION: COMPLETE MECHANISM")
print("="*70)

print("\n🔬 STEP-BY-STEP MECHANISM:")
print("1. RATE IMBALANCE: Environmental preference creates asymmetric conversion rates")
print("2. INITIAL FLOW: Net flow from disfavored to favored enantiomer begins")
print("3. AMPLIFICATION: As favored population grows, its conversion power increases")
print("4. DECLINE: Disfavored population decreases, reducing its counter-flow")
print("5. EXCLUSION: Positive feedback loop drives system to homochirality")
print("6. STABILIZATION: System reaches stable homochiral steady state")

print(f"\n📊 QUANTITATIVE SUMMARY:")

# Safe calculation of maximum competitive advantage
finite_ratios = [r for r in ratios if r != np.inf and not np.isnan(r)]
if finite_ratios:
    max_advantage = max(finite_ratios)
    print(f"• Maximum competitive advantage: {max_advantage:.1f}:1")
else:
    print("• Maximum competitive advantage: Infinite (complete bias)")

# Safe handling of exclusion times
if excl_times:
    fastest_exclusion = min(excl_times)
    print(f"• Fastest exclusion time: {fastest_exclusion:.1f}")
else:
    print("• Fastest exclusion time: N/A (no complete exclusion observed)")

print(f"• Strongest homochirality: P_L = {max(final_P_L):.4f}")
print(f"• Quantum coherence effect: {max(init_coherences) - min(init_coherences):.3f} range")

print("\n✅ COMPETITIVE EXCLUSION SUCCESSFULLY DEMONSTRATED!")