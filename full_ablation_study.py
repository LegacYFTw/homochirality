# ===============================================================
# ENHANCED ABLATION STUDY WITH TQDM AND PARAMETER-BASED SAVING
# ===============================================================

import json
import pickle
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field
import numpy as np
import qutip as qt
import cvxpy as cp
from tqdm.auto import tqdm
import time

# %%
# ===============================================================
# ENHANCED DATA COLLECTOR WITH PARAMETER-BASED DIRECTORIES
# ===============================================================

class EnhancedQuantumSteeringDataCollector:
    """Enhanced data collector with parameter-based directory structure"""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.current_study_id = self._generate_study_id()
        self.experiments_completed = 0
        self.experiments_failed = 0
        
    def _generate_study_id(self) -> str:
        """Generate unique study ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"steering_ablation_{timestamp}"
    
    def _create_parameter_path(self, config: Dict[str, Any]) -> Tuple[Path, str]:
        """Create parameter-based directory path and experiment ID"""
        
        # Extract key parameters for directory structure
        h_params = config['hamiltonian']
        state_name = config['initial_state']['name']
        constraint_config = config['constraint_config']
        
        # Create directory path based on parameters
        param_path = (
            f"H_a{h_params['a_value']:.1f}_b{h_params['b_value']:.1f}/"
            f"state_{state_name}/"
            f"gamma_{constraint_config['gamma']:.3f}/"
            f"cov_{constraint_config['impose_covariance']}_"
            f"pass_{constraint_config['impose_passivity']}/"
            f"tol_cov{constraint_config['relaxation_params']['covariance_tolerance']:.3f}_"
            f"pass{constraint_config['relaxation_params']['passivity_tolerance']:.3f}"
        )
        
        # Create experiment ID
        exp_id = hashlib.md5(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:12]
        
        return Path(param_path), exp_id
    
    def save_experiment_data(self, trajectory_data: TrajectoryData, 
                           raw_cvxpy_logs: Optional[str] = None) -> str:
        """Save individual experiment data with parameter-based directories"""
        
        try:
            # Create parameter-based directory
            config_dict = trajectory_data.experiment_config.dict()
            param_path, exp_id = self._create_parameter_path(config_dict)
            exp_dir = self.data_dir / self.current_study_id / param_path / f"exp_{exp_id}"
            exp_dir.mkdir(parents=True, exist_ok=True)
            
            # Save structured data
            with open(exp_dir / "trajectory_data.json", "w") as f:
                f.write(trajectory_data.json(indent=2))
            
            # Save raw CVXPY logs if available
            if raw_cvxpy_logs:
                with open(exp_dir / "cvxpy_logs.txt", "w") as f:
                    f.write(raw_cvxpy_logs)
            
            # Save numpy arrays separately
            np_data = {
                'fidelity_curve': [step.fidelity for step in trajectory_data.steps],
                'time_points': [step.time for step in trajectory_data.steps],
                'energy_curve': [step.energy for step in trajectory_data.steps],
                'purity_curve': [step.purity for step in trajectory_data.steps],
                'coherence_curve': [step.coherence for step in trajectory_data.steps],
                'bloch_x': [step.bloch_x for step in trajectory_data.steps],
                'bloch_y': [step.bloch_y for step in trajectory_data.steps],
                'bloch_z': [step.bloch_z for step in trajectory_data.steps],
            }
            
            np.savez(exp_dir / "numerical_data.npz", **np_data)
            
            # Save quick summary for easy browsing
            summary = {
                'experiment_id': exp_id,
                'final_fidelity': trajectory_data.final_fidelity,
                'total_improvement': trajectory_data.total_improvement,
                'convergence_speed': trajectory_data.convergence_speed,
                'constraint_satisfaction': trajectory_data.constraint_satisfaction,
                'num_steps': len(trajectory_data.steps),
                'success': any(step.success for step in trajectory_data.steps),
                'hamiltonian': f"a{config_dict['hamiltonian']['a_value']}_b{config_dict['hamiltonian']['b_value']}",
                'initial_state': config_dict['initial_state']['name'],
                'gamma': config_dict['constraint_config']['gamma'],
                'constraints': {
                    'covariance': config_dict['constraint_config']['impose_covariance'],
                    'passivity': config_dict['constraint_config']['impose_passivity']
                }
            }
            
            with open(exp_dir / "quick_summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            
            self.experiments_completed += 1
            return str(exp_dir)
            
        except Exception as e:
            self.experiments_failed += 1
            print(f"❌ Failed to save experiment data: {e}")
            return ""
    
    def get_progress_report(self) -> Dict[str, Any]:
        """Get current progress report"""
        return {
            'study_id': self.current_study_id,
            'completed': self.experiments_completed,
            'failed': self.experiments_failed,
            'total': self.experiments_completed + self.experiments_failed,
            'success_rate': self.experiments_completed / max(1, self.experiments_completed + self.experiments_failed)
        }

# %%
# ===============================================================
# PROGRESS TRACKING AND REPORTING
# ===============================================================

class AblationProgressTracker:
    """Track and report ablation study progress"""
    
    def __init__(self, total_experiments: int):
        self.total_experiments = total_experiments
        self.start_time = time.time()
        self.completed = 0
        self.failed = 0
        self.last_report_time = self.start_time
        self.report_interval = 60  # Report every 60 seconds
        
    def update(self, success: bool = True):
        """Update progress counters"""
        if success:
            self.completed += 1
        else:
            self.failed += 1
        
        self._maybe_report_progress()
    
    def _maybe_report_progress(self):
        """Report progress if enough time has passed"""
        current_time = time.time()
        if current_time - self.last_report_time >= self.report_interval:
            self._report_progress()
            self.last_report_time = current_time
    
    def _report_progress(self):
        """Report current progress"""
        elapsed = time.time() - self.start_time
        total_done = self.completed + self.failed
        progress_pct = (total_done / self.total_experiments) * 100
        
        # Calculate ETA
        if total_done > 0:
            time_per_experiment = elapsed / total_done
            remaining_time = time_per_experiment * (self.total_experiments - total_done)
            eta_str = f"{remaining_time/3600:.1f}h"
        else:
            eta_str = "Unknown"
        
        print(f"\n📊 PROGRESS REPORT [{datetime.now().strftime('%H:%M:%S')}]")
        print(f"   Completed: {self.completed}/{self.total_experiments} ({progress_pct:.1f}%)")
        print(f"   Failed: {self.failed}")
        print(f"   Success Rate: {self.completed/max(1, total_done)*100:.1f}%")
        print(f"   Elapsed: {elapsed/3600:.1f}h, ETA: {eta_str}")
        print(f"   Current Rate: {total_done/elapsed*60:.1f} experiments/min")
        print("-" * 50)
    
    def final_report(self):
        """Print final report"""
        total_time = time.time() - self.start_time
        print(f"\n🎯 ABLATION STUDY COMPLETE!")
        print(f"📈 Final Results:")
        print(f"   ✅ Successful: {self.completed}")
        print(f"   ❌ Failed: {self.failed}")
        print(f"   📊 Success Rate: {self.completed/self.total_experiments*100:.1f}%")
        print(f"   ⏱️  Total Time: {total_time/3600:.2f} hours")
        print(f"   🚀 Average Rate: {self.total_experiments/total_time*60:.1f} experiments/min")

# %%
# ===============================================================
# ENHANCED ABLATION STUDY WITH TQDM
# ===============================================================

def run_comprehensive_ablation_study_with_tqdm():
    """Run comprehensive ablation study with enhanced progress tracking"""
    
    print("🚀 STARTING COMPREHENSIVE ABLATION STUDY WITH TQDM")
    print("="*80)
    
    # Initialize data collector and progress tracker
    data_collector = EnhancedQuantumSteeringDataCollector()
    
    # Create configurations
    hamiltonians, initial_states, constraint_configs = create_ablation_study_configurations()
    
    # Calculate total experiments
    total_experiments = len(constraint_configs) * len(hamiltonians) * len(initial_states)
    
    # Initialize progress tracker
    progress_tracker = AblationProgressTracker(total_experiments)
    
    print(f"📊 Study Configuration:")
    print(f"   • Hamiltonians: {len(hamiltonians)}")
    print(f"   • Initial States: {len(initial_states)}")
    print(f"   • Constraint Configs: {len(constraint_configs)}")
    print(f"   • Total Experiments: {total_experiments:,}")
    print(f"   • Study ID: {data_collector.current_study_id}")
    print("="*80)
    
    # Time parameters
    tlist = np.linspace(0, 8, 40)  # Reduced for comprehensive study
    dt = tlist[1] - tlist[0] if len(tlist) > 1 else 0.2
    
    # Track experiments
    experiment_configs = []
    trajectory_data_list = []
    
    # Main experiment loop with nested TQDM
    print("🔬 Starting experiment loop...")
    
    # Outer loop: constraint configurations
    constraint_pbar = tqdm(constraint_configs, desc="Constraint Configs", position=0, leave=True)
    
    for constraint_config in constraint_pbar:
        constraint_pbar.set_postfix({
            'gamma': f"{constraint_config['gamma']:.3f}",
            'cov': constraint_config['impose_covariance'],
            'pass': constraint_config['impose_passivity']
        })
        
        # Middle loop: Hamiltonians
        hamiltonian_pbar = tqdm(hamiltonians.items(), desc="Hamiltonians", position=1, leave=False)
        
        for (a, b), H in hamiltonian_pbar:
            hamiltonian_pbar.set_postfix({'a': a, 'b': b})
            
            # Inner loop: initial states
            state_pbar = tqdm(initial_states.items(), desc="Initial States", position=2, leave=False)
            
            for state_name, initial_state in state_pbar:
                state_pbar.set_postfix({'state': state_name})
                
                try:
                    # Create experiment configuration
                    exp_config = create_experiment_config(
                        H, (a, b), initial_state, state_name, constraint_config, tlist
                    )
                    
                    # Run experiment
                    trajectory_data, cvxpy_logs = run_single_ablation_experiment(
                        H, initial_state, tlist, dt, constraint_config, exp_config
                    )
                    
                    if trajectory_data and len(trajectory_data.steps) > 0:
                        # Save experiment data
                        save_path = data_collector.save_experiment_data(trajectory_data, cvxpy_logs)
                        
                        # Store for summary
                        experiment_configs.append(exp_config)
                        trajectory_data_list.append(trajectory_data)
                        
                        # Update progress
                        progress_tracker.update(success=True)
                        
                        # Update TQDM postfix with recent results
                        recent_fidelity = trajectory_data.final_fidelity
                        state_pbar.set_postfix({
                            'state': state_name, 
                            'fid': f'{recent_fidelity:.3f}'
                        })
                        
                    else:
                        progress_tracker.update(success=False)
                        
                except Exception as e:
                    progress_tracker.update(success=False)
                    tqdm.write(f"❌ Experiment failed: H(a={a}, b={b}), state={state_name}, error: {str(e)[:100]}...")
                    continue
                
                # Brief pause to prevent overwhelming the system
                time.sleep(0.01)
            
            # Close state pbar
            state_pbar.close()
        
        # Close Hamiltonian pbar
        hamiltonian_pbar.close()
        
        # Report progress after each constraint config
        progress_report = data_collector.get_progress_report()
        tqdm.write(f"📈 Completed constraint config: γ={constraint_config['gamma']:.3f}, "
                  f"cov={constraint_config['impose_covariance']}, "
                  f"pass={constraint_config['impose_passivity']}")
        tqdm.write(f"   Overall progress: {progress_report['completed']}/{total_experiments} "
                  f"({progress_report['completed']/total_experiments*100:.1f}%)")
    
    # Close constraint pbar
    constraint_pbar.close()
    
    # Final progress report
    progress_tracker.final_report()
    
    # Create study summary
    ablation_results = AblationStudyResults(
        study_id=data_collector.current_study_id,
        timestamp=datetime.now().isoformat(),
        total_experiments=total_experiments,
        successful_experiments=progress_tracker.completed,
        experiment_configs=experiment_configs,
        trajectory_data=trajectory_data_list,
        summary_statistics=compute_study_statistics(trajectory_data_list)
    )
    
    # Save study summary
    study_path = data_collector.save_study_summary(ablation_results)
    
    print(f"\n✅ ABLATION STUDY COMPLETE!")
    print(f"📁 Data saved to: {study_path}")
    
    # Print final statistics
    print_final_statistics(ablation_results)
    
    return ablation_results

def print_final_statistics(ablation_results: AblationStudyResults):
    """Print comprehensive final statistics"""
    
    stats = ablation_results.summary_statistics
    trajectory_data = ablation_results.trajectory_data
    
    print(f"\n📋 FINAL STUDY STATISTICS")
    print("="*60)
    print(f"🎯 Fidelity Performance:")
    print(f"   Mean Final Fidelity: {stats['mean_fidelity']:.3f} ± {stats['std_fidelity']:.3f}")
    print(f"   Max Fidelity: {stats['max_fidelity']:.3f}")
    print(f"   Min Fidelity: {stats['min_fidelity']:.3f}")
    print(f"   Average Improvement: {stats['mean_improvement']:.3f}")
    
    print(f"\n⚡ Convergence Metrics:")
    print(f"   Mean Convergence Speed: {stats['mean_convergence_speed']:.3f}")
    
    print(f"\n🔧 Constraint Performance:")
    print(f"   Mean Constraint Satisfaction: {stats['mean_constraint_satisfaction']:.3f}")
    
    print(f"\n📊 Overall Success:")
    print(f"   Successful Experiments: {ablation_results.successful_experiments}/{ablation_results.total_experiments}")
    print(f"   Success Rate: {stats['success_rate']:.3f}")
    
    # Additional insights
    if trajectory_data:
        # Best performing configurations
        best_experiments = sorted(trajectory_data, key=lambda x: x.final_fidelity, reverse=True)[:5]
        print(f"\n🏆 TOP 5 PERFORMING CONFIGURATIONS:")
        for i, exp in enumerate(best_experiments):
            config = exp.experiment_config
            print(f"   {i+1}. Fidelity: {exp.final_fidelity:.3f} | "
                  f"H(a={config.hamiltonian.a_value}, b={config.hamiltonian.b_value}) | "
                  f"State: {config.initial_state.name} | "
                  f"γ={config.constraint_config.gamma:.3f}")
    
    # Performance by constraint type
    if trajectory_data:
        constraint_groups = {}
        for exp in trajectory_data:
            key = (exp.experiment_config.constraint_config.impose_covariance,
                   exp.experiment_config.constraint_config.impose_passivity)
            if key not in constraint_groups:
                constraint_groups[key] = []
            constraint_groups[key].append(exp.final_fidelity)
        
        print(f"\n🔍 PERFORMANCE BY CONSTRAINT TYPE:")
        for (cov, passv), fidelities in constraint_groups.items():
            avg_fid = np.mean(fidelities)
            print(f"   Covariance: {cov}, Passivity: {passv} -> "
                  f"Avg Fidelity: {avg_fid:.3f} (n={len(fidelities)})")

# %%
# ===============================================================
# OPTIMIZED CONFIGURATION GENERATION
# ===============================================================

def create_optimized_ablation_configurations():
    """Create optimized ablation study configurations for comprehensive coverage"""
    
    print("🔧 Generating optimized ablation configurations...")
    
    # Extensive Hamiltonian configurations
    a_values = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    b_values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    hamiltonians = generate_hamiltonians(a_values, b_values)
    
    # Representative initial states covering Bloch sphere
    initial_states = {
        'ground': qt.basis(2, 0),
        'excited': qt.basis(2, 1),
        'superpos_plus': (qt.basis(2, 0) + qt.basis(2, 1)).unit(),
        'superpos_i': (qt.basis(2, 0) + 1j * qt.basis(2, 1)).unit(),
        'mixed_x': 0.7 * qt.basis(2, 0) + 0.3 * qt.basis(2, 1),
    }
    
    # Comprehensive constraint configurations
    constraint_configs = []
    
    # Gamma values covering different regimes
    gammas = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
    
    # All constraint mode combinations
    constraint_modes = [
        (True, True),   # Both constraints
        (True, False),  # Only covariance
        (False, True),  # Only passivity
        (False, False), # No constraints
    ]
    
    # Relaxation parameters covering tight to loose tolerances
    relaxation_params_list = [
        {'covariance_tolerance': 0.001, 'passivity_tolerance': 0.001, 'state_distance_tolerance': 0.01},
        {'covariance_tolerance': 0.005, 'passivity_tolerance': 0.005, 'state_distance_tolerance': 0.03},
        {'covariance_tolerance': 0.01, 'passivity_tolerance': 0.01, 'state_distance_tolerance': 0.05},
        {'covariance_tolerance': 0.05, 'passivity_tolerance': 0.05, 'state_distance_tolerance': 0.1},
    ]
    
    # Generate all combinations
    for gamma in gammas:
        for cov, passv in constraint_modes:
            for relax_params in relaxation_params_list:
                constraint_configs.append({
                    'impose_covariance': cov,
                    'impose_passivity': passv,
                    'gamma': gamma,
                    'relaxation_params': relax_params
                })
    
    print(f"✅ Generated {len(hamiltonians)} Hamiltonians, {len(initial_states)} initial states, "
          f"{len(constraint_configs)} constraint configs")
    print(f"📊 Total experiments: {len(hamiltonians) * len(initial_states) * len(constraint_configs):,}")
    
    return hamiltonians, initial_states, constraint_configs

# %%
# ===============================================================
# QUICK VALIDATION RUN
# ===============================================================

def run_quick_validation_study():
    """Run a quick validation study to test the framework"""
    
    print("🧪 RUNNING QUICK VALIDATION STUDY")
    print("="*60)
    
    # Smaller configuration set for validation
    a_values = np.array([-2.0, 0.0, 2.0])
    b_values = np.array([-1.0, 0.0, 1.0])
    hamiltonians = generate_hamiltonians(a_values, b_values)
    
    initial_states = {
        'ground': qt.basis(2, 0),
        'superpos_plus': (qt.basis(2, 0) + qt.basis(2, 1)).unit(),
    }
    
    constraint_configs = [
        {'impose_covariance': True, 'impose_passivity': True, 'gamma': 0.1, 
         'relaxation_params': {'covariance_tolerance': 0.01, 'passivity_tolerance': 0.01, 'state_distance_tolerance': 0.05}},
        {'impose_covariance': False, 'impose_passivity': True, 'gamma': 0.1, 
         'relaxation_params': {'covariance_tolerance': 0.01, 'passivity_tolerance': 0.01, 'state_distance_tolerance': 0.05}},
    ]
    
    total_experiments = len(hamiltonians) * len(initial_states) * len(constraint_configs)
    print(f"Validation study: {total_experiments} experiments")
    
    # Time parameters (very short for validation)
    tlist = np.linspace(0, 2, 10)
    dt = tlist[1] - tlist[0] if len(tlist) > 1 else 0.2
    
    # Initialize data collector
    data_collector = EnhancedQuantumSteeringDataCollector()
    data_collector.current_study_id = "validation_" + data_collector.current_study_id
    
    progress_tracker = AblationProgressTracker(total_experiments)
    
    # Run validation loop
    for constraint_config in tqdm(constraint_configs, desc="Constraint Configs"):
        for (a, b), H in hamiltonians.items():
            for state_name, initial_state in initial_states.items():
                try:
                    exp_config = create_experiment_config(
                        H, (a, b), initial_state, state_name, constraint_config, tlist
                    )
                    
                    trajectory_data, cvxpy_logs = run_single_ablation_experiment(
                        H, initial_state, tlist, dt, constraint_config, exp_config
                    )
                    
                    if trajectory_data:
                        data_collector.save_experiment_data(trajectory_data, cvxpy_logs)
                        progress_tracker.update(success=True)
                    else:
                        progress_tracker.update(success=False)
                        
                except Exception as e:
                    progress_tracker.update(success=False)
                    print(f"❌ Validation experiment failed: {e}")
    
    progress_tracker.final_report()
    print("✅ Validation study complete!")

# %%
# ===============================================================
# MAIN EXECUTION
# ===============================================================

if __name__ == "__main__":
    
    print("🎯 QUANTUM STEERING ABLATION STUDY LAUNCHER")
    print("="*60)
    print("Choose study type:")
    print("1. Quick Validation (~100 experiments)")
    print("2. Full Ablation Study (~20,000+ experiments)")
    print("3. Custom Configuration")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        run_quick_validation_study()
    elif choice == "2":
        # Use optimized configurations
        global create_ablation_study_configurations
        create_ablation_study_configurations = create_optimized_ablation_configurations
        ablation_results = run_comprehensive_ablation_study_with_tqdm()
    elif choice == "3":
        print("Custom configuration not yet implemented. Running full study...")
        create_ablation_study_configurations = create_optimized_ablation_configurations
        ablation_results = run_comprehensive_ablation_study_with_tqdm()
    else:
        print("Invalid choice. Running quick validation...")
        run_quick_validation_study()