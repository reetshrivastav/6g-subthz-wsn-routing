"""
Centralized Configuration for Review 2: A vs B vs C Comparative Benchmark
"""
import os

CFG = {
    # ── Paths ──
    'data_path': 'data/module1_synthetic_dataset_balanced.csv',
    'artifacts_dir': './artifacts',
    'figures_dir': './figures',

    # ── Global Seed & Device ──
    'global_seed': 42,

    # ── Dataset Split ──
    'test_size': 0.20,
    'val_size': 0.10,

    # ── Model A: Base Paper (Alanazi et al., 2025) ──
    'ann_hidden': 10,               # 1 hidden layer with 10 neurons
    'tabular_lr': 0.1,
    'tabular_gamma': 0.95,
    'tabular_temp': 0.5,            # SoftMax temperature tau

    # ── Model B: Senior Work (Rahul Anand, 2026) ──
    'link_hidden': 32,
    'link_dropout': 0.30,
    'link_lr': 1e-3,
    'link_epochs': 100,
    'link_patience': 10,
    'mc_samples': 30,

    # ── Model C: Extension MVP (Temporal Transformer + QR-DQN/CVaR) ──
    'seq_len': 10,                  # Temporal sliding window T
    'trans_d_model': 32,
    'trans_nhead': 2,
    'trans_layers': 2,
    'trans_dropout': 0.1,
    'trans_lr': 1e-3,
    'trans_epochs': 60,
    'ar1_rho': 0.82,                # Autoregressive correlation coefficient
    'qr_quantiles': 21,             # N=21 quantile atoms
    'cvar_alpha': 0.25,             # CVaR-alpha worst-case tail fraction
    'qr_lr': 3e-4,

    # ── Standard Canonical WSN Simulation Environment ──
    'n_nodes': 40,
    'area': 100.0,
    'comm_range': 40.0,
    'init_energy': 1.0,
    'E_elec': 50e-9,
    'E_amp': 100e-12,
    'packet_bits': 4000,
    'mobile_fraction': 0.20,
    'mobility_speed': 2.0,
    'MAX_A': 6,
    'MAX_HOPS': 15,

    # ── Reward Function Weights (Canonical) ──
    'R_deliver': 10.0,
    'R_fail': -10.0,
    'w_energy': 5.0,
    'w_delay': 0.1,
    'w_uncert': 0.25,

    # ── RL Training Parameters ──
    'dqn_hidden': 256,
    'dqn_episodes': 10000,          # 10k episodes for rapid convergence in Colab
    'batch_size': 256,
    'replay_cap': 80000,
    'gamma': 0.99,
    'lr_dqn': 3e-4,
    'target_update': 1000,
    'eps_start': 1.0,
    'eps_end': 0.05,
    'eps_decay': 0.9995,

    # ── Evaluation Parameters ──
    'eval_rounds': 150,
    'packets_per_round': 40,
    'eval_seeds': [42, 43, 44],

    # ── Sparse Stress Scenario (Key Differentiator for C) ──
    'sparse_n_nodes': 15,
    'sparse_comm_range': 30.0,
    'sparse_mobile_frac': 0.25,

    # ── Phase 4: Adaptive Multi-Objective Reward & Context-Aware State Expansion ──
    'enable_phase4': False,         # Backwards compatible: False by default
    'E_crit': 0.30,                 # Critical battery threshold for exponential barrier
    'lambda_energy_exp': 3.0,       # Scaling factor for low-battery penalty
    'gamma_queue': 2.0,             # Queue backlog penalty coefficient
    'max_queue_cap': 20,            # Max packet buffer capacity per node
    'queue_service_rate': 4,        # Packets cleared per node per round
    'per_action_dim_phase3': 4,     # Phase 3 per-action features: [mean, std, energy, progress]
    'per_action_dim_phase4': 7,     # Phase 4 per-action: [mean, std, energy, progress, queue, stability, adv_ratio]
}

# Derived State Dimensions
STATE_DIM_PHASE3 = 2 + 4 * int(CFG['MAX_A'])   # 26
STATE_DIM_PHASE4 = 3 + 7 * int(CFG['MAX_A'])   # 45 (3 base: cur_energy, cur_dist, cur_queue)

# Default to Phase 3 for Review 2 compatibility
STATE_DIM = STATE_DIM_PHASE4 if CFG.get('enable_phase4', False) else STATE_DIM_PHASE3
CFG['STATE_DIM'] = STATE_DIM
CFG['STATE_DIM_PHASE3'] = STATE_DIM_PHASE3
CFG['STATE_DIM_PHASE4'] = STATE_DIM_PHASE4

# ── Phase 5: Four 6G Physical Stress Environments ──
PHASE5_SCENARIOS = {
    'sparse': {
        'name': 'Sparse Bottleneck (15 Nodes)',
        'n_nodes': 15,
        'comm_range': 30.0,
        'init_energy': 1.0,
        'mobile_fraction': 0.20,
        'mobility_speed': 2.0,
        'obstacles': [],
    },
    'obstacle': {
        'name': 'Obstacle LoS Blockage (Sub-THz)',
        'n_nodes': 40,
        'comm_range': 40.0,
        'init_energy': 1.0,
        'mobile_fraction': 0.20,
        'mobility_speed': 2.0,
        'obstacles': [(35.0, 35.0, 65.0, 65.0)],  # Center corridor blocker (xmin, ymin, xmax, ymax)
        'blockage_attenuation_rssi': 28.0,
        'blockage_attenuation_snr': 20.0,
        'blockage_plr_penalty': 0.65,
    },
    'high_mobility': {
        'name': 'Extreme Mobility (40% @ 3.5m/s)',
        'n_nodes': 40,
        'comm_range': 40.0,
        'init_energy': 1.0,
        'mobile_fraction': 0.40,
        'mobility_speed': 3.5,
        'obstacles': [],
    },
    'energy_scarce': {
        'name': 'Energy-Scarce Depletion (E0=0.5J)',
        'n_nodes': 40,
        'comm_range': 40.0,
        'init_energy': 0.5,
        'mobile_fraction': 0.20,
        'mobility_speed': 2.0,
        'obstacles': [],
    },
}
CFG['PHASE5_SCENARIOS'] = PHASE5_SCENARIOS

def ensure_dirs():
    os.makedirs(CFG['artifacts_dir'], exist_ok=True)
    os.makedirs(CFG['figures_dir'], exist_ok=True)
