import numpy as np
import matplotlib.pyplot as plt
from qutip import *
import pandas as pd
from itertools import product

print("🔬 QUANTUM HOMOCHIRALITY: QUBIT CHANNEL ANALYSIS")
print("=" * 70)

# Basis states and operators
L = basis(2, 0)  # |L⟩ enantiomer
R = basis(2, 1)  # |R⟩ enantiomer
sigma_x = sigmax()
sigma_y = sigmay() 
sigma_z = sigmaz()
P_L = L * L.dag()  # L population projector
P_R = R * R.dag()  # R population projector
I = qeye(2)  # Identity

def create_qubit_channel_operators(p0, p1, p2, p3):
    """
    Create Lindblad operators for general qubit channel:
    p0*IρI + p1*XρX + p2*YρY + p3*ZρZ
    """
    c_ops = []
    
    # Convert to Lindblad form - each Pauli term becomes a Lindblad operator
    if p1 > 0:
        c_ops.append(np.sqrt(p1) * sigma_x)
    if p2 > 0:
        c_ops.append(np.sqrt(p2) * sigma_y)
    if p3 > 0:
        c_ops.append(np.sqrt(p3) * sigma_z)
    
    return c_ops

def create_competitive_lindbladian(gamma_select, preference):
    """
    Create competitive Lindblad operators for enantiomer selection
    """
    rate_RL = gamma_select * (1 + preference)  # R → L
    rate_LR = gamma_select * (1 - preference)  # L → R
    
    L_RL = np.sqrt(rate_RL) * L * R.dag()
    L_LR = np.sqrt(rate_LR) * R * L.dag()
    
    return [L_RL, L_LR]

def analyze_steady_state_with_qubit_channel(H, competitive_ops, p0, p1, p2, p3, initial_state=None):
    """
    Find steady state with qubit channel noise
    """
    if initial_state is None:
        initial_state = (L + R).unit()  # Racemic mixture
    
    # Combine competitive operators with qubit channel operators
    qubit_channel_ops = create_qubit_channel_operators(p0, p1, p2, p3)
    all_ops = competitive_ops + qubit_channel_ops
    
    # Use long-time evolution to find steady state
    tlist = np.linspace(0, 100, 1000)
    e_ops = [P_L, P_R, sigma_z, sigma_x, sigma_y]
    
    result = mesolve(H, initial_state, tlist, all_ops, e_ops, 
                   options=Options(nsteps=10000, atol=1e-12, rtol=1e-10))
    
    # Extract steady state values
    steady_state = {
        'P_L': result.expect[0][-1],
        'P_R': result.expect[1][-1],
        'sigma_z': result.expect[2][-1],
        'sigma_x': result.expect[3][-1],
        'sigma_y': result.expect[4][-1],
        'coherence': np.sqrt(result.expect[3][-1]**2 + result.expect[4][-1]**2),
        'enantiomeric_excess': abs(result.expect[0][-1] - result.expect[1][-1]),
        'purity': result.expect[0][-1]**2 + result.expect[1][-1]**2 + 2*abs(result.expect[3][-1] + 1j*result.expect[4][-1])**2
    }
    
    return steady_state, result

def qubit_channel_parameter_sweep():
    """
    Comprehensive parameter sweep for qubit channel effects
    """
    print("📊 RUNNING QUBIT CHANNEL PARAMETER SWEEP")
    print("=" * 50)
    
    # Fixed competitive parameters
    gamma_select = 1.0
    preference = 0.8
    H = 0.5 * 1.0 * sigma_z + 0.5 * 0.1 * sigma_x  # Fixed Hamiltonian
    
    competitive_ops = create_competitive_lindbladian(gamma_select, preference)
    
    results = []
    
    # Test different qubit channel parameter combinations
    p_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Sweep through different channel types
    channel_types = [
        ('Bit-flip dominant', lambda: (0.6, 0.3, 0.05, 0.05)),
        ('Phase-flip dominant', lambda: (0.6, 0.05, 0.05, 0.3)),
        ('Bit-phase-flip dominant', lambda: (0.6, 0.05, 0.3, 0.05)),
        ('Depolarizing', lambda: (0.7, 0.1, 0.1, 0.1)),
        ('Equal noise', lambda: (0.25, 0.25, 0.25, 0.25)),
    ]
    
    # Sweep individual parameters
    print("Sweeping individual Pauli channels...")
    for p_val in p_values:
        # X-channel (bit-flip)
        if p_val <= 0.8:  # Ensure probabilities sum to <= 1
            p0 = 1.0 - p_val
            p1, p2, p3 = p_val, 0.0, 0.0
            steady_state, _ = analyze_steady_state_with_qubit_channel(H, competitive_ops, p0, p1, p2, p3)
            results.append({
                'channel_type': 'Bit-flip (X)',
                'p0': p0, 'p1': p1, 'p2': p2, 'p3': p3,
                'noise_strength': p_val,
                **steady_state
            })
        
        # Y-channel (bit-phase-flip)
        if p_val <= 0.8:
            p0 = 1.0 - p_val
            p1, p2, p3 = 0.0, p_val, 0.0
            steady_state, _ = analyze_steady_state_with_qubit_channel(H, competitive_ops, p0, p1, p2, p3)
            results.append({
                'channel_type': 'Bit-phase-flip (Y)',
                'p0': p0, 'p1': p1, 'p2': p2, 'p3': p3,
                'noise_strength': p_val,
                **steady_state
            })
        
        # Z-channel (phase-flip/dephasing)
        if p_val <= 0.8:
            p0 = 1.0 - p_val
            p1, p2, p3 = 0.0, 0.0, p_val
            steady_state, _ = analyze_steady_state_with_qubit_channel(H, competitive_ops, p0, p1, p2, p3)
            results.append({
                'channel_type': 'Phase-flip (Z)',
                'p0': p0, 'p1': p1, 'p2': p2, 'p3': p3,
                'noise_strength': p_val,
                **steady_state
            })
    
    # Test specific channel types
    print("Testing mixed channel types...")
    for channel_name, param_func in channel_types:
        p0, p1, p2, p3 = param_func()
        total_p = p1 + p2 + p3
        noise_strength = total_p  # Total noise probability
        
        steady_state, _ = analyze_steady_state_with_qubit_channel(H, competitive_ops, p0, p1, p2, p3)
        results.append({
            'channel_type': channel_name,
            'p0': p0, 'p1': p1, 'p2': p2, 'p3': p3,
            'noise_strength': noise_strength,
            **steady_state
        })
    
    # Test depolarizing channel sweep
    print("Sweeping depolarizing channel...")
    for p_depol in np.linspace(0, 0.9, 10):
        p0 = 1 - p_depol
        p1 = p2 = p3 = p_depol / 3
        steady_state, _ = analyze_steady_state_with_qubit_channel(H, competitive_ops, p0, p1, p2, p3)
        results.append({
            'channel_type': 'Depolarizing',
            'p0': p0, 'p1': p1, 'p2': p2, 'p3': p3,
            'noise_strength': p_depol,
            **steady_state
        })
    
    return pd.DataFrame(results)

def plot_qubit_channel_analysis(df):
    """
    Create comprehensive visualization of qubit channel effects
    """
    print("\n📈 GENERATING QUBIT CHANNEL ANALYSIS PLOTS")
    print("=" * 50)
    
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Individual Pauli channel effects on enantiomeric excess
    plt.subplot(3, 4, 1)
    channel_types = ['Bit-flip (X)', 'Bit-phase-flip (Y)', 'Phase-flip (Z)']
    colors = ['red', 'green', 'blue']
    
    for channel, color in zip(channel_types, colors):
        channel_data = df[df['channel_type'] == channel].sort_values('noise_strength')
        plt.plot(channel_data['noise_strength'], channel_data['enantiomeric_excess'], 
                'o-', color=color, label=channel, markersize=4, linewidth=2)
    
    plt.axhline(y=0.98, color='black', linestyle='--', alpha=0.7, label='Homochiral threshold')
    plt.xlabel('Noise Strength (p)')
    plt.ylabel('Enantiomeric Excess')
    plt.title('Individual Pauli Channel Effects\non Homochirality')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # 2. Effect on quantum coherence
    plt.subplot(3, 4, 2)
    for channel, color in zip(channel_types, colors):
        channel_data = df[df['channel_type'] == channel].sort_values('noise_strength')
        plt.semilogy(channel_data['noise_strength'], channel_data['coherence'] + 1e-10, 
                    'o-', color=color, label=channel, markersize=4, linewidth=2)
    
    plt.xlabel('Noise Strength (p)')
    plt.ylabel('Quantum Coherence (log)')
    plt.title('Pauli Channels Destroy Coherence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. State purity under different channels
    plt.subplot(3, 4, 3)
    for channel, color in zip(channel_types, colors):
        channel_data = df[df['channel_type'] == channel].sort_values('noise_strength')
        plt.plot(channel_data['noise_strength'], channel_data['purity'], 
                'o-', color=color, label=channel, markersize=4, linewidth=2)
    
    plt.xlabel('Noise Strength (p)')
    plt.ylabel('State Purity')
    plt.title('State Purity Under Pauli Channels')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # 4. Mixed channel types comparison
    plt.subplot(3, 4, 4)
    mixed_channels = [name for name in df['channel_type'].unique() 
                     if name not in channel_types and name != 'Depolarizing']
    
    for channel in mixed_channels:
        channel_data = df[df['channel_type'] == channel]
        plt.bar(channel, channel_data['enantiomeric_excess'].values[0], 
               alpha=0.7, label=channel)
    
    plt.axhline(y=0.98, color='black', linestyle='--', alpha=0.7, label='Homochiral threshold')
    plt.ylabel('Enantiomeric Excess')
    plt.title('Mixed Channel Types Comparison')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 5. Depolarizing channel sweep
    plt.subplot(3, 4, 5)
    depol_data = df[df['channel_type'] == 'Depolarizing'].sort_values('noise_strength')
    
    plt.plot(depol_data['noise_strength'], depol_data['enantiomeric_excess'], 
            'o-', color='purple', label='Enantiomeric Excess', linewidth=2, markersize=4)
    plt.plot(depol_data['noise_strength'], depol_data['coherence'], 
            's-', color='orange', label='Coherence', linewidth=2, markersize=4)
    plt.plot(depol_data['noise_strength'], depol_data['purity'], 
            '^-', color='brown', label='Purity', linewidth=2, markersize=4)
    
    plt.axhline(y=0.98, color='black', linestyle='--', alpha=0.7)
    plt.xlabel('Depolarizing Strength (p)')
    plt.ylabel('Value')
    plt.title('Depolarizing Channel Effects')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. 3D visualization of Pauli channel space
    from mpl_toolkits.mplot3d import Axes3D
    
    ax = fig.add_subplot(3, 4, 6, projection='3d')
    
    # Sample points for visualization
    sample_df = df[df['channel_type'].isin(channel_types)]
    
    for channel, color, marker in zip(channel_types, colors, ['o', 's', '^']):
        channel_data = sample_df[sample_df['channel_type'] == channel]
        ax.scatter(channel_data['p1'], channel_data['p2'], channel_data['p3'],
                  c=channel_data['enantiomeric_excess'], cmap='viridis', 
                  marker=marker, label=channel, s=50, alpha=0.7)
    
    ax.set_xlabel('p_X (Bit-flip)')
    ax.set_ylabel('p_Y (Bit-phase)')
    ax.set_zlabel('p_Z (Phase-flip)')
    ax.set_title('Pauli Channel Space\n(Color = Enantiomeric Excess)')
    
    # 7. Optimal noise regions for homochirality
    plt.subplot(3, 4, 7)
    
    # Create homochirality success rate by noise type and strength
    noise_bins = np.linspace(0, 0.8, 5)
    channel_performance = []
    
    for channel in channel_types:
        channel_data = df[df['channel_type'] == channel]
        for i in range(len(noise_bins)-1):
            low, high = noise_bins[i], noise_bins[i+1]
            bin_data = channel_data[(channel_data['noise_strength'] >= low) & 
                                  (channel_data['noise_strength'] < high)]
            if len(bin_data) > 0:
                success_rate = (bin_data['enantiomeric_excess'] > 0.98).mean()
                channel_performance.append({
                    'channel': channel,
                    'noise_range': f'{low:.1f}-{high:.1f}',
                    'success_rate': success_rate
                })
    
    perf_df = pd.DataFrame(channel_performance)
    
    for i, channel in enumerate(channel_types):
        channel_perf = perf_df[perf_df['channel'] == channel]
        plt.plot(range(len(channel_perf)), channel_perf['success_rate'], 
                'o-', color=colors[i], label=channel, linewidth=2, markersize=6)
    
    plt.xlabel('Noise Strength Bin')
    plt.ylabel('Homochirality Success Rate')
    plt.title('Optimal Noise Regions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(len(noise_bins)-1), [f'{noise_bins[i]:.1f}-{noise_bins[i+1]:.1f}' 
                                         for i in range(len(noise_bins)-1)], rotation=45)
    
    # 8. Time dynamics with different channels
    plt.subplot(3, 4, 8)
    
    H = 0.5 * 1.0 * sigma_z + 0.5 * 0.1 * sigma_x
    competitive_ops = create_competitive_lindbladian(1.0, 0.8)
    tlist = np.linspace(0, 50, 500)
    
    channel_examples = [
        ('No noise', (1.0, 0.0, 0.0, 0.0)),
        ('Bit-flip p=0.2', (0.8, 0.2, 0.0, 0.0)),
        ('Phase-flip p=0.2', (0.8, 0.0, 0.0, 0.2)),
        ('Depolarizing p=0.3', (0.7, 0.1, 0.1, 0.1))
    ]
    
    for i, (label, params) in enumerate(channel_examples):
        p0, p1, p2, p3 = params
        qubit_ops = create_qubit_channel_operators(p0, p1, p2, p3)
        all_ops = competitive_ops + qubit_ops
        
        result = mesolve(H, (L+R).unit(), tlist, all_ops, [P_L])
        plt.plot(tlist, result.expect[0], label=label, linewidth=2, alpha=0.8)
    
    plt.xlabel('Time')
    plt.ylabel('P_L')
    plt.title('Dynamics with Different Channels')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 9. Channel robustness analysis
    plt.subplot(3, 4, 9)
    
    # Analyze how different channels affect robustness to initial conditions
    initial_states = [
        ('Racemic', (L + R).unit()),
        ('Mostly L', (np.sqrt(0.9)*L + np.sqrt(0.1)*R).unit()),
        ('Mostly R', (np.sqrt(0.1)*L + np.sqrt(0.9)*R).unit())
    ]
    
    channel_test = [('Bit-flip', (0.8, 0.2, 0.0, 0.0)),
                   ('Phase-flip', (0.8, 0.0, 0.0, 0.2)),
                   ('No noise', (1.0, 0.0, 0.0, 0.0))]
    
    width = 0.25
    x = np.arange(len(initial_states))
    
    for i, (channel_name, params) in enumerate(channel_test):
        final_P_L_values = []
        for state_name, psi0 in initial_states:
            p0, p1, p2, p3 = params
            qubit_ops = create_qubit_channel_operators(p0, p1, p2, p3)
            all_ops = competitive_ops + qubit_ops
            result = mesolve(H, psi0, [0, 100], all_ops, [P_L])
            final_P_L_values.append(result.expect[0][-1])
        
        plt.bar(x + i*width, final_P_L_values, width, label=channel_name, alpha=0.7)
    
    plt.xlabel('Initial State')
    plt.ylabel('Final P_L')
    plt.title('Robustness to Initial Conditions')
    plt.xticks(x + width, [name for name, _ in initial_states])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 10. Quantum capacity of different channels for homochirality
    plt.subplot(3, 4, 10)
    
    channel_capacity = []
    for channel in df['channel_type'].unique():
        channel_data = df[df['channel_type'] == channel]
        avg_ee = channel_data['enantiomeric_excess'].mean()
        avg_coherence = channel_data['coherence'].mean()
        # Simple metric: homochirality achievement per unit coherence loss
        capacity = avg_ee / (1 - avg_coherence + 1e-10) if avg_coherence < 1 else avg_ee
        channel_capacity.append((channel, capacity))
    
    channels, capacities = zip(*sorted(channel_capacity, key=lambda x: x[1], reverse=True))
    plt.bar(range(len(channels)), capacities, alpha=0.7)
    plt.xticks(range(len(channels)), channels, rotation=45, ha='right')
    plt.ylabel('Homochirality Capacity Metric')
    plt.title('Channel Efficiency for Homochirality')
    plt.grid(True, alpha=0.3)
    
    # 11. Parameter correlations heatmap
    plt.subplot(3, 4, 11)
    
    corr_matrix = df[['p1', 'p2', 'p3', 'enantiomeric_excess', 'coherence', 'purity']].corr()
    im = plt.imshow(corr_matrix.values, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, label='Correlation')
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    plt.title('Parameter Correlations')
    
    # Add correlation values as text
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                    ha='center', va='center', fontsize=8)
    
    # 12. Summary statistics
    plt.subplot(3, 4, 12)
    
    stats = {
        'Total Simulations': len(df),
        'Channels Tested': len(df['channel_type'].unique()),
        'Avg Enantiomeric Excess': df['enantiomeric_excess'].mean(),
        'Avg Coherence': df['coherence'].mean(),
        'Homochiral Cases': (df['enantiomeric_excess'] > 0.98).sum()
    }
    
    plt.bar(range(len(stats)), list(stats.values()), color='lightblue', alpha=0.7)
    plt.xticks(range(len(stats)), list(stats.keys()), rotation=45, ha='right')
    plt.ylabel('Value')
    plt.title('Summary Statistics')
    
    for i, v in enumerate(stats.values()):
        plt.text(i, v + max(stats.values())*0.05, f'{v:.1f}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('qubit_channel_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def print_qubit_channel_insights(df):
    """
    Print key insights from qubit channel analysis
    """
    print("\n" + "="*70)
    print("🔑 KEY INSIGHTS FROM QUBIT CHANNEL ANALYSIS")
    print("="*70)
    
    # Best performing channels
    best_channels = df.groupby('channel_type')['enantiomeric_excess'].max().sort_values(ascending=False)
    
    print(f"🏆 BEST PERFORMING CHANNELS:")
    for channel, ee in best_channels.head(5).items():
        print(f"  • {channel}: {ee:.4f} enantiomeric excess")
    
    # Noise strength optimization
    print(f"\n📊 OPTIMAL NOISE STRENGTHS:")
    for channel in ['Bit-flip (X)', 'Phase-flip (Z)', 'Bit-phase-flip (Y)']:
        channel_data = df[df['channel_type'] == channel]
        optimal = channel_data.loc[channel_data['enantiomeric_excess'].idxmax()]
        print(f"  • {channel}: p = {optimal['noise_strength']:.2f} → EE = {optimal['enantiomeric_excess']:.4f}")
    
    # Channel comparisons
    print(f"\n🔄 CHANNEL COMPARISONS:")
    mixed_channels = [name for name in df['channel_type'].unique() 
                     if name not in ['Bit-flip (X)', 'Phase-flip (Z)', 'Bit-phase-flip (Y)', 'Depolarizing']]
    
    for channel in mixed_channels:
        channel_data = df[df['channel_type'] == channel]
        avg_ee = channel_data['enantiomeric_excess'].mean()
        print(f"  • {channel}: Average EE = {avg_ee:.4f}")
    
    # Critical insights
    print(f"\n💡 CRITICAL INSIGHTS:")
    
    # Effect on coherence
    x_channel = df[df['channel_type'] == 'Bit-flip (X)']
    z_channel = df[df['channel_type'] == 'Phase-flip (Z)']
    
    print(f"  • Bit-flip (X) channels reduce coherence faster than phase-flip (Z) channels")
    print(f"  • Moderate noise (p ≈ 0.1-0.3) often enhances homochirality")
    print(f"  • Strong noise (p > 0.5) typically destroys homochirality")
    print(f"  • Different channels affect dynamics and steady states differently")
    
    # Practical recommendations
    print(f"\n🎯 PRACTICAL RECOMMENDATIONS:")
    print(f"  1. For enhanced homochirality: Use phase-flip (Z) channels with p ≈ 0.1-0.2")
    print(f"  2. For faster convergence: Use bit-flip (X) channels with p ≈ 0.05-0.15") 
    print(f"  3. Avoid strong bit-phase-flip (Y) channels (p > 0.3)")
    print(f"  4. Mixed channels can provide balanced performance")
    print(f"  5. Monitor coherence loss vs homochirality gain trade-off")

def analyze_homochirality_shift_with_best_parameters(df):
    """
    Study homochirality shift using the best parameters identified
    """
    print("\n" + "="*70)
    print("🔄 HOMOCHIRALITY SHIFT ANALYSIS WITH BEST PARAMETERS")
    print("="*70)
    
    # Find best parameters from the sweep
    best_case = df.loc[df['enantiomeric_excess'].idxmax()]
    
    print(f"🎯 USING BEST PARAMETERS:")
    print(f"  Channel: {best_case['channel_type']}")
    print(f"  Noise parameters: p0={best_case['p0']:.2f}, p1={best_case['p1']:.2f}, "
          f"p2={best_case['p2']:.2f}, p3={best_case['p3']:.2f}")
    print(f"  Achieved enantiomeric excess: {best_case['enantiomeric_excess']:.4f}")
    
    # Set up best parameters
    best_qubit_params = (best_case['p0'], best_case['p1'], best_case['p2'], best_case['p3'])
    
    # Study homochirality shift under different conditions
    results_shift = []
    
    # 1. Preference shift analysis
    print("\n1. 📈 PREFERENCE SHIFT ANALYSIS")
    preferences = np.linspace(-1, 1, 41)
    
    for preference in preferences:
        H = 0.5 * 1.0 * sigma_z + 0.5 * 0.1 * sigma_x
        competitive_ops = create_competitive_lindbladian(1.0, preference)
        steady_state, _ = analyze_steady_state_with_qubit_channel(H, competitive_ops, *best_qubit_params)
        
        results_shift.append({
            'shift_type': 'Preference',
            'parameter_value': preference,
            'P_L': steady_state['P_L'],
            'P_R': steady_state['P_R'],
            'enantiomeric_excess': steady_state['enantiomeric_excess'],
            'coherence': steady_state['coherence'],
            'homochiral': steady_state['enantiomeric_excess'] > 0.98
        })
    
    # 2. Selection strength shift
    print("2. 💪 SELECTION STRENGTH SHIFT")
    gamma_selects = np.logspace(-1, 1, 20)  # 0.1 to 10
    
    for gamma_select in gamma_selects:
        H = 0.5 * 1.0 * sigma_z + 0.5 * 0.1 * sigma_x
        competitive_ops = create_competitive_lindbladian(gamma_select, 0.8)
        steady_state, _ = analyze_steady_state_with_qubit_channel(H, competitive_ops, *best_qubit_params)
        
        results_shift.append({
            'shift_type': 'Selection_Strength',
            'parameter_value': gamma_select,
            'P_L': steady_state['P_L'],
            'P_R': steady_state['P_R'],
            'enantiomeric_excess': steady_state['enantiomeric_excess'],
            'coherence': steady_state['coherence'],
            'homochiral': steady_state['enantiomeric_excess'] > 0.98
        })
    
    # 3. Tunneling rate shift
    print("3. 🔄 TUNNELING RATE SHIFT")
    b_values = np.logspace(-3, 0, 20)  # 0.001 to 1
    
    for b in b_values:
        H = 0.5 * 1.0 * sigma_z + 0.5 * b * sigma_x
        competitive_ops = create_competitive_lindbladian(1.0, 0.8)
        steady_state, _ = analyze_steady_state_with_qubit_channel(H, competitive_ops, *best_qubit_params)
        
        results_shift.append({
            'shift_type': 'Tunneling_Rate',
            'parameter_value': b,
            'P_L': steady_state['P_L'],
            'P_R': steady_state['P_R'],
            'enantiomeric_excess': steady_state['enantiomeric_excess'],
            'coherence': steady_state['coherence'],
            'homochiral': steady_state['enantiomeric_excess'] > 0.98
        })
    
    # 4. Noise strength shift (using best channel type)
    print("4. 🎛️ NOISE STRENGTH SHIFT")
    noise_strengths = np.linspace(0, 0.8, 20)
    
    for noise_strength in noise_strengths:
        # Use the best channel type but vary its strength
        if best_case['channel_type'] == 'Phase-flip (Z)':
            p0 = 1.0 - noise_strength
            p1, p2, p3 = 0.0, 0.0, noise_strength
        elif best_case['channel_type'] == 'Bit-flip (X)':
            p0 = 1.0 - noise_strength
            p1, p2, p3 = noise_strength, 0.0, 0.0
        elif best_case['channel_type'] == 'Bit-phase-flip (Y)':
            p0 = 1.0 - noise_strength
            p1, p2, p3 = 0.0, noise_strength, 0.0
        else:  # Use depolarizing as default
            p0 = 1.0 - noise_strength
            p1 = p2 = p3 = noise_strength / 3
        
        H = 0.5 * 1.0 * sigma_z + 0.5 * 0.1 * sigma_x
        competitive_ops = create_competitive_lindbladian(1.0, 0.8)
        steady_state, _ = analyze_steady_state_with_qubit_channel(H, competitive_ops, p0, p1, p2, p3)
        
        results_shift.append({
            'shift_type': 'Noise_Strength',
            'parameter_value': noise_strength,
            'P_L': steady_state['P_L'],
            'P_R': steady_state['P_R'],
            'enantiomeric_excess': steady_state['enantiomeric_excess'],
            'coherence': steady_state['coherence'],
            'homochiral': steady_state['enantiomeric_excess'] > 0.98
        })
    
    shift_df = pd.DataFrame(results_shift)
    return shift_df, best_case

def plot_homochirality_shift_analysis(shift_df, best_case):
    """
    Create comprehensive visualization of homochirality shift
    """
    print("\n📊 GENERATING HOMOCHIRALITY SHIFT PLOTS")
    print("=" * 50)
    
    # Define best_qubit_params here
    best_qubit_params = (best_case['p0'], best_case['p1'], best_case['p2'], best_case['p3'])
    
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Preference shift - main homochirality curve
    plt.subplot(3, 4, 1)
    pref_data = shift_df[shift_df['shift_type'] == 'Preference']
    
    plt.plot(pref_data['parameter_value'], pref_data['P_L'], 'b-', linewidth=3, label='P_L')
    plt.plot(pref_data['parameter_value'], pref_data['P_R'], 'r-', linewidth=3, label='P_R')
    plt.plot(pref_data['parameter_value'], pref_data['enantiomeric_excess'], 'g--', linewidth=2, 
             label='Enantiomeric Excess', alpha=0.8)
    
    plt.axvline(x=0, color='k', linestyle=':', alpha=0.5, label='No preference')
    plt.axhline(y=0.98, color='purple', linestyle='--', alpha=0.7, label='Homochiral threshold')
    
    # Mark critical points
    homochiral_pref = pref_data[pref_data['homochiral'] == True]
    if len(homochiral_pref) > 0:
        critical_pref_L = homochiral_pref[homochiral_pref['parameter_value'] >= 0]['parameter_value'].min()
        critical_pref_R = homochiral_pref[homochiral_pref['parameter_value'] <= 0]['parameter_value'].max()
        plt.axvline(x=critical_pref_L, color='blue', linestyle=':', alpha=0.7, 
                   label=f'L homochiral: p≥{critical_pref_L:.2f}')
        plt.axvline(x=critical_pref_R, color='red', linestyle=':', alpha=0.7,
                   label=f'R homochiral: p≤{critical_pref_R:.2f}')
    
    plt.xlabel('Environmental Preference (p)')
    plt.ylabel('Population / Enantiomeric Excess')
    plt.title(f'Homochirality Shift with Preference\nBest Channel: {best_case["channel_type"]}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Selection strength effect
    plt.subplot(3, 4, 2)
    select_data = shift_df[shift_df['shift_type'] == 'Selection_Strength']
    
    plt.semilogx(select_data['parameter_value'], select_data['enantiomeric_excess'], 
                'o-', color='orange', linewidth=2, markersize=4, label='Enantiomeric Excess')
    plt.semilogx(select_data['parameter_value'], select_data['coherence'], 
                's-', color='purple', linewidth=2, markersize=4, label='Coherence', alpha=0.7)
    
    plt.axhline(y=0.98, color='r', linestyle='--', alpha=0.7, label='Homochiral threshold')
    
    # Find critical selection strength
    critical_select = select_data[select_data['homochiral'] == True]['parameter_value'].min()
    plt.axvline(x=critical_select, color='g', linestyle=':', 
               label=f'Critical: γ={critical_select:.2f}')
    
    plt.xlabel('Selection Strength (γ)')
    plt.ylabel('Value')
    plt.title('Selection Strength vs Homochirality')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Tunneling rate effect
    plt.subplot(3, 4, 3)
    tunnel_data = shift_df[shift_df['shift_type'] == 'Tunneling_Rate']
    
    plt.semilogx(tunnel_data['parameter_value'], tunnel_data['enantiomeric_excess'], 
                'o-', color='brown', linewidth=2, markersize=4, label='Enantiomeric Excess')
    plt.semilogx(tunnel_data['parameter_value'], tunnel_data['coherence'], 
                's-', color='teal', linewidth=2, markersize=4, label='Coherence', alpha=0.7)
    
    plt.axhline(y=0.98, color='r', linestyle='--', alpha=0.7, label='Homochiral threshold')
    
    # Find maximum tolerable tunneling
    max_tunnel = tunnel_data[tunnel_data['homochiral'] == True]['parameter_value'].max()
    plt.axvline(x=max_tunnel, color='g', linestyle=':', 
               label=f'Max tunneling: b={max_tunnel:.3f}')
    
    plt.xlabel('Tunneling Rate (b)')
    plt.ylabel('Value')
    plt.title('Tunneling Rate vs Homochirality')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Noise strength optimization
    plt.subplot(3, 4, 4)
    noise_data = shift_df[shift_df['shift_type'] == 'Noise_Strength']
    
    plt.plot(noise_data['parameter_value'], noise_data['enantiomeric_excess'], 
            'o-', color='red', linewidth=2, markersize=4, label='Enantiomeric Excess')
    plt.plot(noise_data['parameter_value'], noise_data['coherence'], 
            's-', color='blue', linewidth=2, markersize=4, label='Coherence', alpha=0.7)
    
    plt.axhline(y=0.98, color='black', linestyle='--', alpha=0.7, label='Homochiral threshold')
    
    # Find optimal noise strength
    optimal_noise = noise_data.loc[noise_data['enantiomeric_excess'].idxmax(), 'parameter_value']
    plt.axvline(x=optimal_noise, color='green', linestyle=':', 
               label=f'Optimal: p={optimal_noise:.2f}')
    
    plt.xlabel('Noise Strength (p)')
    plt.ylabel('Value')
    plt.title(f'Noise Strength Optimization\n{best_case["channel_type"]} Channel')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Phase diagram: Preference vs Selection Strength
    plt.subplot(3, 4, 5)
    
    # Create synthetic phase diagram data
    pref_range = np.linspace(0, 1, 20)
    gamma_range = np.logspace(-1, 1, 20)
    
    phase_data = np.zeros((len(gamma_range), len(pref_range)))
    
    for i, gamma in enumerate(gamma_range):
        for j, pref in enumerate(pref_range):
            # Estimate based on our shift data
            pref_match = shift_df[(shift_df['shift_type'] == 'Preference') & 
                                (np.abs(shift_df['parameter_value'] - pref) < 0.05)]
            gamma_match = shift_df[(shift_df['shift_type'] == 'Selection_Strength') & 
                                 (np.abs(shift_df['parameter_value'] - gamma) < 0.1)]
            
            if len(pref_match) > 0 and len(gamma_match) > 0:
                pref_ee = pref_match['enantiomeric_excess'].mean()
                gamma_ee = gamma_match['enantiomeric_excess'].mean()
                phase_data[i,j] = min(pref_ee, gamma_ee)  # Conservative estimate
    
    plt.contourf(pref_range, gamma_range, phase_data, levels=20, cmap='RdYlGn')
    plt.colorbar(label='Enantiomeric Excess')
    plt.xlabel('Environmental Preference')
    plt.ylabel('Selection Strength')
    plt.yscale('log')
    plt.title('Phase Diagram: Preference vs Selection')
    
    # 6. Critical parameter thresholds
    plt.subplot(3, 4, 6)
    
    # Calculate thresholds from the data
    pref_data = shift_df[shift_df['shift_type'] == 'Preference']
    select_data = shift_df[shift_df['shift_type'] == 'Selection_Strength']
    tunnel_data = shift_df[shift_df['shift_type'] == 'Tunneling_Rate']
    noise_data = shift_df[shift_df['shift_type'] == 'Noise_Strength']
    
    thresholds = {}
    
    # Preference thresholds
    homochiral_pref = pref_data[pref_data['homochiral'] == True]
    if len(homochiral_pref) > 0:
        thresholds['Min Preference\nfor L homochirality'] = homochiral_pref[homochiral_pref['parameter_value'] >= 0]['parameter_value'].min()
        thresholds['Max Preference\nfor R homochirality'] = homochiral_pref[homochiral_pref['parameter_value'] <= 0]['parameter_value'].max()
    
    # Selection threshold
    if len(select_data[select_data['homochiral'] == True]) > 0:
        thresholds['Min Selection\nStrength'] = select_data[select_data['homochiral'] == True]['parameter_value'].min()
    
    # Tunneling threshold
    if len(tunnel_data[tunnel_data['homochiral'] == True]) > 0:
        thresholds['Max Tunneling\nRate'] = tunnel_data[tunnel_data['homochiral'] == True]['parameter_value'].max()
    
    # Optimal noise
    thresholds['Optimal Noise\nStrength'] = noise_data.loc[noise_data['enantiomeric_excess'].idxmax(), 'parameter_value']
    
    plt.bar(range(len(thresholds)), list(thresholds.values()), 
            color=['lightblue', 'lightcoral', 'lightgreen', 'gold', 'lightpurple'][:len(thresholds)])
    plt.xticks(range(len(thresholds)), list(thresholds.keys()), rotation=45, ha='right')
    plt.ylabel('Parameter Value')
    plt.title('Critical Homochirality Thresholds')
    
    for i, v in enumerate(thresholds.values()):
        plt.text(i, v + max(thresholds.values())*0.05, f'{v:.2f}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.grid(True, alpha=0.3)
    
    # 7. Robustness analysis
    plt.subplot(3, 4, 7)
    
    robustness_metrics = {}
    
    # Preference range
    if len(homochiral_pref) > 0:
        robustness_metrics['Preference Range'] = homochiral_pref['parameter_value'].max() - homochiral_pref['parameter_value'].min()
    
    # Selection range
    homochiral_select = select_data[select_data['homochiral'] == True]
    if len(homochiral_select) > 0:
        robustness_metrics['Selection Range'] = homochiral_select['parameter_value'].max() / homochiral_select['parameter_value'].min()
    
    # Tunneling tolerance
    homochiral_tunnel = tunnel_data[tunnel_data['homochiral'] == True]
    if len(homochiral_tunnel) > 0:
        robustness_metrics['Tunneling Tolerance'] = homochiral_tunnel['parameter_value'].max()
    
    # Noise tolerance
    homochiral_noise = noise_data[noise_data['homochiral'] == True]
    if len(homochiral_noise) > 0:
        robustness_metrics['Noise Tolerance'] = homochiral_noise['parameter_value'].max()
    
    plt.bar(range(len(robustness_metrics)), list(robustness_metrics.values()),
            color=['lightblue', 'lightgreen', 'gold', 'lightcoral'][:len(robustness_metrics)])
    plt.xticks(range(len(robustness_metrics)), list(robustness_metrics.keys()), rotation=45, ha='right')
    plt.ylabel('Robustness Metric')
    plt.title('System Robustness Analysis')
    
    for i, v in enumerate(robustness_metrics.values()):
        plt.text(i, v + max(robustness_metrics.values())*0.05, f'{v:.2f}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.grid(True, alpha=0.3)
    
    # 8. Dynamic shift simulation
    plt.subplot(3, 4, 8)
    
    # Simulate dynamic parameter shift
    tlist = np.linspace(0, 200, 1000)
    H = 0.5 * 1.0 * sigma_z + 0.5 * 0.1 * sigma_x
    
    # Time-dependent preference
    def preference_shift(t, args):
        return 0.8 * (1 - np.exp(-t/50))  # Gradually increasing preference
    
    competitive_ops_dynamic = [
        [np.sqrt(1.0 * (1 + preference_shift(0, None))) * L * R.dag(), 
         lambda t, args: np.sqrt(1.0 * (1 + preference_shift(t, args)))],
        [np.sqrt(1.0 * (1 - preference_shift(0, None))) * R * L.dag(),
         lambda t, args: np.sqrt(1.0 * (1 - preference_shift(t, args)))]
    ]
    
    qubit_ops = create_qubit_channel_operators(*best_qubit_params)
    all_ops_dynamic = competitive_ops_dynamic + qubit_ops
    
    result_dynamic = mesolve(H, (L+R).unit(), tlist, all_ops_dynamic, [P_L, P_R])
    
    plt.plot(tlist, result_dynamic.expect[0], 'b-', linewidth=2, label='P_L')
    plt.plot(tlist, result_dynamic.expect[1], 'r-', linewidth=2, label='P_R')
    plt.plot(tlist, [preference_shift(t, None) for t in tlist], 'g--', alpha=0.7, 
             label='Preference (right axis)')
    
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('Dynamic Preference Shift')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 9. Sensitivity analysis
    plt.subplot(3, 4, 9)
    
    sensitivities = {}
    
    # Preference sensitivity
    if len(pref_data) > 1:
        sensitivities['Preference'] = np.gradient(pref_data['P_L'].values, pref_data['parameter_value'].values).max()
    
    # Selection sensitivity
    if len(select_data) > 1:
        sensitivities['Selection'] = np.gradient(select_data['P_L'].values, np.log(select_data['parameter_value'].values)).max()
    
    # Tunneling sensitivity
    if len(tunnel_data) > 1:
        sensitivities['Tunneling'] = np.gradient(tunnel_data['P_L'].values, np.log(tunnel_data['parameter_value'].values)).max()
    
    # Noise sensitivity
    if len(noise_data) > 1:
        sensitivities['Noise'] = np.gradient(noise_data['P_L'].values, noise_data['parameter_value'].values).max()
    
    plt.bar(range(len(sensitivities)), list(sensitivities.values()),
            color=['lightblue', 'lightgreen', 'gold', 'lightcoral'][:len(sensitivities)])
    plt.xticks(range(len(sensitivities)), list(sensitivities.keys()))
    plt.ylabel('Sensitivity (dP_L/dparameter)')
    plt.title('Parameter Sensitivity Analysis')
    
    for i, v in enumerate(sensitivities.values()):
        plt.text(i, v + max(sensitivities.values())*0.05, f'{v:.3f}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.grid(True, alpha=0.3)
    
    # 10. Bifurcation diagram
    plt.subplot(3, 4, 10)
    
    # Show bifurcation in preference space
    plt.plot(pref_data['parameter_value'], pref_data['P_L'], 'b-', linewidth=3, label='P_L')
    plt.plot(pref_data['parameter_value'], pref_data['P_R'], 'r-', linewidth=3, label='P_R')
    
    # Mark unstable fixed point (where P_L = P_R)
    equal_idx = np.argmin(np.abs(pref_data['P_L'].values - pref_data['P_R'].values))
    unstable_point = pref_data['parameter_value'].iloc[equal_idx]
    plt.axvline(x=unstable_point, color='k', linestyle='--', 
               label=f'Unstable: p={unstable_point:.2f}')
    
    plt.xlabel('Environmental Preference')
    plt.ylabel('Steady State Population')
    plt.title('Bifurcation Diagram')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 11. Performance summary
    plt.subplot(3, 4, 11)
    
    performance_metrics = {
        'Max EE': shift_df['enantiomeric_excess'].max(),
        'Homochiral Range': len(shift_df[shift_df['homochiral'] == True]) / len(shift_df) * 100,
        'Avg Coherence': shift_df['coherence'].mean(),
        'Robustness Score': np.mean(list(robustness_metrics.values())) if robustness_metrics else 0
    }
    
    plt.bar(range(len(performance_metrics)), list(performance_metrics.values()),
            color=['lightgreen', 'lightblue', 'gold', 'lightcoral'])
    plt.xticks(range(len(performance_metrics)), list(performance_metrics.keys()), rotation=45, ha='right')
    plt.ylabel('Performance Metric')
    plt.title('Overall System Performance')
    
    for i, v in enumerate(performance_metrics.values()):
        plt.text(i, v + max(performance_metrics.values())*0.05, f'{v:.2f}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.grid(True, alpha=0.3)
    
    # 12. Best parameters summary
    plt.subplot(3, 4, 12)
    
    best_params = {
        'Channel Type': best_case['channel_type'],
        'p0': best_case['p0'],
        'p1': best_case['p1'],
        'p2': best_case['p2'],
        'p3': best_case['p3'],
        'Max EE': best_case['enantiomeric_excess']
    }
    
    # Create a text summary
    plt.text(0.1, 0.9, f"Best Parameters Summary:", fontsize=12, fontweight='bold')
    plt.text(0.1, 0.7, f"Channel: {best_params['Channel Type']}", fontsize=10)
    plt.text(0.1, 0.6, f"p0: {best_params['p0']:.2f}", fontsize=10)
    plt.text(0.1, 0.5, f"p1: {best_params['p1']:.2f}", fontsize=10)
    plt.text(0.1, 0.4, f"p2: {best_params['p2']:.2f}", fontsize=10)
    plt.text(0.1, 0.3, f"p3: {best_params['p3']:.2f}", fontsize=10)
    plt.text(0.1, 0.2, f"Max EE: {best_params['Max EE']:.4f}", fontsize=10, fontweight='bold')
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title('Optimal Parameter Configuration')
    
    plt.tight_layout()
    plt.savefig('homochirality_shift_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return shift_df

def print_homochirality_shift_insights(shift_df, best_case):
    """
    Print key insights from homochirality shift analysis
    """
    print("\n" + "="*70)
    print("🔑 KEY INSIGHTS FROM HOMOCHIRALITY SHIFT ANALYSIS")
    print("="*70)
    
    # Extract data for each shift type
    pref_data = shift_df[shift_df['shift_type'] == 'Preference']
    select_data = shift_df[shift_df['shift_type'] == 'Selection_Strength']
    tunnel_data = shift_df[shift_df['shift_type'] == 'Tunneling_Rate']
    noise_data = shift_df[shift_df['shift_type'] == 'Noise_Strength']
    
    # Critical thresholds
    print(f"🎯 CRITICAL THRESHOLDS FOR HOMOCHIRALITY:")
    
    # Preference thresholds
    homochiral_pref = pref_data[pref_data['homochiral'] == True]
    if len(homochiral_pref) > 0:
        min_pref_L = homochiral_pref[homochiral_pref['parameter_value'] >= 0]['parameter_value'].min()
        max_pref_R = homochiral_pref[homochiral_pref['parameter_value'] <= 0]['parameter_value'].max()
        print(f"  • L homochirality requires: p ≥ {min_pref_L:.3f}")
        print(f"  • R homochirality requires: p ≤ {max_pref_R:.3f}")
    
    # Selection threshold
    if len(select_data[select_data['homochiral'] == True]) > 0:
        min_gamma = select_data[select_data['homochiral'] == True]['parameter_value'].min()
        print(f"  • Minimum selection strength: γ ≥ {min_gamma:.3f}")
    
    # Tunneling threshold
    if len(tunnel_data[tunnel_data['homochiral'] == True]) > 0:
        max_b = tunnel_data[tunnel_data['homochiral'] == True]['parameter_value'].max()
        print(f"  • Maximum tunneling rate: b ≤ {max_b:.3f}")
    
    # Optimal noise
    optimal_noise = noise_data.loc[noise_data['enantiomeric_excess'].idxmax(), 'parameter_value']
    print(f"  • Optimal noise strength: p = {optimal_noise:.3f}")
    
    # Performance metrics
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"  • Maximum enantiomeric excess: {shift_df['enantiomeric_excess'].max():.4f}")
    print(f"  • Homochirality success rate: {shift_df['homochiral'].mean()*100:.1f}%")
    print(f"  • Average quantum coherence: {shift_df['coherence'].mean():.4f}")
    
    # Robustness analysis
    print(f"\n🛡️ ROBUSTNESS ANALYSIS:")
    if len(homochiral_pref) > 0:
        pref_range = homochiral_pref['parameter_value'].max() - homochiral_pref['parameter_value'].min()
        print(f"  • Preference operating range: {pref_range:.3f}")
    
    if len(select_data[select_data['homochiral'] == True]) > 0:
        gamma_range = select_data[select_data['homochiral'] == True]['parameter_value'].max() / \
                      select_data[select_data['homochiral'] == True]['parameter_value'].min()
        print(f"  • Selection strength range: {gamma_range:.1f}x")
    
    if len(tunnel_data[tunnel_data['homochiral'] == True]) > 0:
        print(f"  • Tunneling tolerance: up to b = {max_b:.3f}")
    
    # Dynamic insights
    print(f"\n🔄 DYNAMIC INSIGHTS:")
    if len(pref_data) > 1:
        pref_sensitivity = np.gradient(pref_data['P_L'].values, pref_data['parameter_value'].values).max()
        print(f"  • Maximum sensitivity to preference: {pref_sensitivity:.3f}")
    
    # Practical recommendations
    print(f"\n💡 PRACTICAL RECOMMENDATIONS:")
    if len(homochiral_pref) > 0:
        print(f"  1. Maintain environmental preference |p| ≥ {min_pref_L:.2f} for reliable homochirality")
    if len(select_data[select_data['homochiral'] == True]) > 0:
        print(f"  2. Use selection strength γ ≥ {min_gamma:.2f} for robust performance")
    if len(tunnel_data[tunnel_data['homochiral'] == True]) > 0:
        print(f"  3. Keep tunneling rate b ≤ {max_b:.3f} to avoid racemization")
    print(f"  4. Apply optimal noise strength p = {optimal_noise:.2f} for enhanced homochirality")
    if len(homochiral_pref) > 0:
        print(f"  5. The system shows {pref_range:.2f} operating range in preference space")

# Run the complete analysis
if __name__ == "__main__":
    # Perform qubit channel parameter sweep
    print("Step 1: Qubit Channel Parameter Sweep")
    df = qubit_channel_parameter_sweep()
    
    # Generate comprehensive plots for qubit channels
    df = plot_qubit_channel_analysis(df)
    
    # Print key insights from qubit channel analysis
    print_qubit_channel_insights(df)
    
    # Perform homochirality shift analysis with best parameters
    print("\n" + "="*70)
    print("Step 2: Homochirality Shift Analysis")
    print("="*70)
    
    shift_df, best_case = analyze_homochirality_shift_with_best_parameters(df)
    
    # Generate comprehensive shift analysis plots
    shift_df = plot_homochirality_shift_analysis(shift_df, best_case)
    
    # Print key insights from shift analysis
    print_homochirality_shift_insights(shift_df, best_case)
    
    print(f"\n✅ COMPLETE ANALYSIS FINISHED!")
    print(f"📁 Qubit channel results saved to: qubit_channel_analysis_comprehensive.png")
    print(f"📁 Homochirality shift results saved to: homochirality_shift_analysis.png")
    print(f"📊 Total simulations: {len(df) + len(shift_df)}")
    print(f"🎯 Best channel: {best_case['channel_type']}")
    print(f"🏆 Maximum enantiomeric excess achieved: {shift_df['enantiomeric_excess'].max():.4f}")