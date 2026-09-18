"""
Phase 6 Master Ablation & Sensitivity Sweep Suite:
1. Component Ablation: Full Model C+ vs MLP Predictor vs Risk-Neutral (alpha=1.0) vs Fixed Linear Reward.
2. Continuous CVaR-alpha Risk Sensitivity Sweep: alpha in [0.10, 0.25, 0.50, 0.75, 1.00].
"""
import os
import sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
for _p in [
    os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib"),
    "C:/Users/Reet/AppData/Local/Programs/Python/Python310/lib/site-packages/torch/lib",
]:
    if os.path.exists(_p):
        try:
            os.add_dll_directory(_p)
        except Exception:
            pass

import math
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from src.config import CFG, ensure_dirs
from src.dataset import load_or_create_dataset, split_dataset, generate_temporal_sequences
from sklearn.preprocessing import StandardScaler
from src.models_link import (
    LinkUQNet, UncertaintyModel,
    TemporalLinkTransformer, TemporalUncertaintyModel,
    train_point_model, train_temporal_model
)
from src.env import build_env
from src.models_rl import (
    QRDQN, QRDQNRouter, ReplayBuffer,
    compute_qrdqn_loss, masked_argmax
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def gini_coefficient(array):
    """Computes Gini index on node energy consumption."""
    array = np.array(array, dtype=np.float64).flatten()
    if np.amin(array) < 0:
        array -= np.amin(array)
    array += 1e-9
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return float((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))


def train_c_agent(cfg: dict, link_model, state_dim: int, alpha: float = 0.25,
                  episodes: int = 250, seed: int = 42, is_temporal: bool = True):
    """Trains a QR-DQN policy network with the specified link model and alpha."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    pol = QRDQN(state_dim, cfg['MAX_A'], num_quantiles=cfg['qr_quantiles'],
                hidden=cfg['dqn_hidden'], use_layernorm=True).to(DEVICE)
    tgt = QRDQN(state_dim, cfg['MAX_A'], num_quantiles=cfg['qr_quantiles'],
                hidden=cfg['dqn_hidden'], use_layernorm=True).to(DEVICE)
    tgt.load_state_dict(pol.state_dict())
    opt = torch.optim.Adam(pol.parameters(), lr=cfg['qr_lr'])
    rb = ReplayBuffer(cfg['replay_cap'])

    temp_kw = {'temporal_model': link_model} if is_temporal else {'uq': link_model}
    env, _ = build_env(cfg, seed=seed, **temp_kw)

    eps = cfg['eps_start']
    for ep in range(episodes):
        s, m, acts = env.reset_packet()
        if s is None:
            env, _ = build_env(cfg, seed=seed + ep, **temp_kw)
            s, m, acts = env.reset_packet()
        if s is None: continue
        done = False
        while not done:
            valid = [i for i in range(cfg['MAX_A']) if m[i] == 1.0]
            if np.random.uniform(0, 1) < eps:
                a = np.random.choice(valid) if valid else 0
            else:
                with torch.no_grad():
                    q = pol(torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(DEVICE))[0]
                    k = max(1, int(math.ceil(alpha * cfg['qr_quantiles'])))
                    q_sorted, _ = torch.sort(q, dim=-1)
                    cvar_vals = q_sorted[:, :k].mean(dim=-1)
                    a = masked_argmax(cvar_vals, torch.tensor(m).to(DEVICE))

            step_res, r, done, _ = env.step(a)
            sn = step_res[0] if not done else np.zeros(state_dim, dtype=np.float32)
            mn = step_res[1] if not done else np.zeros(cfg['MAX_A'], dtype=np.float32)
            rb.push(s, m, a, r, sn, mn, float(done))
            if not done: s, m = sn, mn

            if len(rb) >= cfg['batch_size']:
                batch = rb.sample(cfg['batch_size'])
                loss = compute_qrdqn_loss(pol, tgt, batch, gamma=cfg['gamma'],
                                          alpha=alpha, device=str(DEVICE))
                opt.zero_grad()
                loss.backward()
                opt.step()
        eps = max(cfg['eps_end'], eps * cfg['eps_decay'])
        if ep % 200 == 0: tgt.load_state_dict(pol.state_dict())
        if ep % 50 == 0:
            env.service_queues()
            env.apply_mobility()

    return QRDQNRouter(pol, alpha=alpha, device=str(DEVICE))


def evaluate_agent(router, cfg: dict, link_model, is_temporal: bool = True,
                   rounds: int = 50, pkts_per_round: int = 25, seed: int = 42,
                   scenario: str | None = None):
    """Evaluates router policy over rounds."""
    temp_kw = {'temporal_model': link_model} if is_temporal else {'uq': link_model}
    env, nodes = build_env(cfg, seed=seed, scenario=scenario, **temp_kw)
    total_pkts = 0
    deliv_pkts = 0
    total_hops = 0
    first_death = rounds

    for r in range(rounds):
        for _ in range(pkts_per_round):
            s, m, acts = env.reset_packet()
            if s is None: break
            total_pkts += 1
            done = False
            while not done:
                a = router.select_action(s, m)
                step_res, r_val, done, info = env.step(a)
                if done:
                    if info.get('delivered', False):
                        deliv_pkts += 1
                        total_hops += info.get('hops', 0)
                    break
                s, m = step_res

        env.service_queues()
        env.apply_mobility()

        alive = sum(1 for n in nodes if n.alive and not n.is_sink)
        if alive < (len(nodes) - 1) and first_death == rounds:
            first_death = r + 1

    pdr = (deliv_pkts / total_pkts * 100.0) if total_pkts > 0 else 0.0
    avg_hops = (total_hops / deliv_pkts) if deliv_pkts > 0 else 0.0
    throughput = (deliv_pkts / max(1, rounds))
    init_e = float(env.cfg.get('init_energy', 1.0))
    consumed = [max(0.0, init_e - n.energy) for n in nodes if not n.is_sink]
    gini = gini_coefficient(consumed)

    return {
        'pdr': pdr,
        'hops': avg_hops,
        'fnd': first_death,
        'throughput': throughput,
        'gini': gini,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=250, help='RL training episodes')
    parser.add_argument('--rounds', type=int, default=50, help='Evaluation rounds')
    parser.add_argument('--pkts_per_round', type=int, default=25, help='Packets per round')
    parser.add_argument('--seeds', type=str, default='42,101,2024', help='Evaluation seeds')
    parser.add_argument('--trans_epochs', type=int, default=25, help='Transformer training epochs')
    parser.add_argument('--mc_samples', type=int, default=15, help='MC dropout sample count')
    args = parser.parse_args()

    ensure_dirs()
    seeds = [int(s) for s in args.seeds.split(',')]

    print("=" * 85)
    print("  PHASE 6: COMPREHENSIVE ABLATION STUDY & CVaR-ALPHA SENSITIVITY SWEEP")
    print(f"  Device: {DEVICE} | Episodes: {args.episodes} | Rounds: {args.rounds} | Seeds: {seeds}")
    print("=" * 85)
    sys.stdout.flush()

    # 1. Dataset & Predictors
    print("\n[Step 1] Initializing Datasets & Training Link Models...")
    sys.stdout.flush()
    df = load_or_create_dataset(CFG['data_path'], seed=CFG['global_seed'])
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df, seed=CFG['global_seed'])
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_val_s = scaler.transform(X_train), scaler.transform(X_val)

    X_seq_train, y_seq_train = generate_temporal_sequences(X_train_s, y_train, seq_len=CFG['seq_len'], rho=CFG['ar1_rho'], seed=CFG['global_seed'])
    X_seq_val, y_seq_val = generate_temporal_sequences(X_val_s, y_val, seq_len=CFG['seq_len'], rho=CFG['ar1_rho'], seed=CFG['global_seed'])

    print("  -> Training Static Point MLP (LinkUQNet with MC dropout)...")
    net_mlp = train_point_model(LinkUQNet(hidden=CFG['link_hidden'], dropout=CFG['link_dropout']),
                                X_train_s, y_train, X_val_s, y_val, epochs=40, lr=CFG['link_lr'], device=str(DEVICE))
    uq_mlp = UncertaintyModel(net_mlp, scaler, device=str(DEVICE), mc_samples=args.mc_samples)

    print(f"  -> Training Temporal Link Transformer ({args.trans_epochs} epochs)...")
    net_trans = train_temporal_model(TemporalLinkTransformer(d_model=CFG['trans_d_model'], nhead=CFG['trans_nhead'], num_layers=CFG['trans_layers']),
                                     X_seq_train, y_seq_train, X_seq_val, y_seq_val, epochs=args.trans_epochs, lr=CFG['trans_lr'], device=str(DEVICE))
    temporal_model = TemporalUncertaintyModel(net_trans, scaler, device=str(DEVICE), mc_samples=args.mc_samples, seq_len=CFG['seq_len'])

    # Configurations
    cfg_p4 = dict(CFG)
    cfg_p4['enable_phase4'] = True
    dim_p4 = CFG['STATE_DIM_PHASE4']

    cfg_p3 = dict(CFG)
    cfg_p3['enable_phase4'] = False
    dim_p3 = CFG['STATE_DIM_PHASE3']

    # ─────────────────────────────────────────────────────────────────────────
    # PART 1: COMPONENT ABLATION STUDY
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 85)
    print("  PART 1: COMPONENT ABLATION MATRIX (Deconstructing Model C+)")
    print("=" * 85)
    sys.stdout.flush()

    ablation_configs = [
        ('Full Model C+ (Complete)', temporal_model, True, cfg_p4, dim_p4, 0.25),
        ('Ablation 1: Static MLP Predictor', uq_mlp, False, cfg_p4, dim_p4, 0.25),
        ('Ablation 2: Risk-Neutral Expected Q (alpha=1.0)', temporal_model, True, cfg_p4, dim_p4, 1.00),
        ('Ablation 3: Fixed Linear Reward (No Barrier)', temporal_model, True, cfg_p3, dim_p3, 0.25),
    ]

    ablation_records = []
    # Test ablations under the Obstacle LoS Blockage stress scenario
    stress_scenario = 'obstacle'

    for name, link_mod, is_temp, cfg_mod, st_dim, alpha_val in ablation_configs:
        print(f"\n  -> Evaluating: {name} (alpha={alpha_val})...")
        sys.stdout.flush()
        router = train_c_agent(cfg_mod, link_mod, st_dim, alpha=alpha_val,
                               episodes=args.episodes, seed=seeds[0], is_temporal=is_temp)

        res_list = []
        for s in seeds:
            res = evaluate_agent(router, cfg_mod, link_mod, is_temporal=is_temp,
                                 rounds=args.rounds, pkts_per_round=args.pkts_per_round, seed=s,
                                 scenario=stress_scenario)
            res_list.append(res)

        pdr_m, pdr_s = float(np.mean([r['pdr'] for r in res_list])), float(np.std([r['pdr'] for r in res_list]))
        hop_m, hop_s = float(np.mean([r['hops'] for r in res_list])), float(np.std([r['hops'] for r in res_list]))
        fnd_m, fnd_s = float(np.mean([r['fnd'] for r in res_list])), float(np.std([r['fnd'] for r in res_list]))
        gini_m, gini_s = float(np.mean([r['gini'] for r in res_list])), float(np.std([r['gini'] for r in res_list]))

        print(f"     PDR: {pdr_m:5.1f}% ± {pdr_s:4.1f}% | Latency: {hop_m:4.2f} hops | FND: {fnd_m:4.1f} rnd | Gini: {gini_m:.3f}")
        sys.stdout.flush()

        ablation_records.append({
            'Configuration': name,
            'PDR_mean': pdr_m,
            'PDR_std': pdr_s,
            'Latency_hops_mean': hop_m,
            'Latency_hops_std': hop_s,
            'FND_rounds_mean': fnd_m,
            'FND_rounds_std': fnd_s,
            'Gini_mean': gini_m,
            'Gini_std': gini_s,
        })

    df_abl = pd.DataFrame(ablation_records)
    csv_abl_path = os.path.join(CFG['artifacts_dir'], 'phase6_ablation_table.csv')
    df_abl.to_csv(csv_abl_path, index=False)
    print(f"\n[Saved Artifact] {csv_abl_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # PART 2: CONTINUOUS CVaR-ALPHA RISK SENSITIVITY SWEEP
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 85)
    print("  PART 2: CONTINUOUS CVaR-ALPHA RISK SENSITIVITY SWEEP")
    print("=" * 85)
    sys.stdout.flush()

    alphas = [0.10, 0.25, 0.50, 0.75, 1.00]
    sensitivity_records = []

    for a_val in alphas:
        print(f"\n  -> Sweeping CVaR alpha = {a_val:.2f}...")
        sys.stdout.flush()
        router_alpha = train_c_agent(cfg_p4, temporal_model, dim_p4, alpha=a_val,
                                     episodes=args.episodes, seed=seeds[0], is_temporal=True)

        res_list = []
        for s in seeds:
            res = evaluate_agent(router_alpha, cfg_p4, temporal_model, is_temporal=True,
                                 rounds=args.rounds, pkts_per_round=args.pkts_per_round, seed=s,
                                 scenario=stress_scenario)
            res_list.append(res)

        pdr_m, pdr_s = float(np.mean([r['pdr'] for r in res_list])), float(np.std([r['pdr'] for r in res_list]))
        hop_m, hop_s = float(np.mean([r['hops'] for r in res_list])), float(np.std([r['hops'] for r in res_list]))
        gini_m, gini_s = float(np.mean([r['gini'] for r in res_list])), float(np.std([r['gini'] for r in res_list]))

        print(f"     alpha={a_val:.2f} | PDR: {pdr_m:5.1f}% ± {pdr_s:4.1f}% | Latency: {hop_m:4.2f} hops | Gini: {gini_m:.3f}")
        sys.stdout.flush()

        sensitivity_records.append({
            'alpha': a_val,
            'PDR_mean': pdr_m,
            'PDR_std': pdr_s,
            'Latency_mean': hop_m,
            'Latency_std': hop_s,
            'Gini_mean': gini_m,
            'Gini_std': gini_s,
        })

    df_sens = pd.DataFrame(sensitivity_records)
    csv_sens_path = os.path.join(CFG['artifacts_dir'], 'phase6_cvar_sensitivity.csv')
    df_sens.to_csv(csv_sens_path, index=False)
    print(f"\n[Saved Artifact] {csv_sens_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. GENERATE 4-PANEL PUBLICATION CHART
    # ─────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Component PDR Ablation Bar Chart
    configs_labels = ['Full Model C+', 'Static MLP', 'Risk-Neutral\n(E[Q])', 'Fixed Linear\nReward']
    pdr_means = df_abl['PDR_mean'].values
    pdr_stds = df_abl['PDR_std'].values
    colors = ['#10b981', '#64748b', '#3b82f6', '#f59e0b']

    bars1 = axes[0, 0].bar(configs_labels, pdr_means, yerr=pdr_stds, capsize=5, color=colors, alpha=0.85, edgecolor='black')
    axes[0, 0].set_title('Ablation: Packet Delivery Ratio (PDR %)', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('PDR (%)')
    axes[0, 0].set_ylim(0, 105)
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.5)
    for bar, val in zip(bars1, pdr_means):
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2.0, val + 2.0, f'{val:.1f}%', ha='center', fontweight='bold')

    # Panel 2: Energy Gini Inequality by Component
    gini_means = df_abl['Gini_mean'].values
    gini_stds = df_abl['Gini_std'].values
    bars2 = axes[0, 1].bar(configs_labels, gini_means, yerr=gini_stds, capsize=5, color=colors, alpha=0.85, edgecolor='black')
    axes[0, 1].set_title('Ablation: Energy Depletion Gini (0=Equal, Lower is Better)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Gini Index')
    axes[0, 1].set_ylim(0, 0.7)
    axes[0, 1].grid(axis='y', linestyle='--', alpha=0.5)
    for bar, val in zip(bars2, gini_means):
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2.0, val + 0.02, f'{val:.3f}', ha='center', fontweight='bold')

    # Panel 3: Continuous CVaR-alpha PDR Curve
    alphas_plot = df_sens['alpha'].values
    pdr_sens = df_sens['PDR_mean'].values
    pdr_sens_err = df_sens['PDR_std'].values
    axes[1, 0].plot(alphas_plot, pdr_sens, marker='o', linewidth=2.5, color='#10b981', label='PDR (%)')
    axes[1, 0].fill_between(alphas_plot, pdr_sens - pdr_sens_err, pdr_sens + pdr_sens_err, color='#10b981', alpha=0.2)
    axes[1, 0].axvline(0.25, color='red', linestyle='--', label='Proposed Default (alpha=0.25)')
    axes[1, 0].set_title('Sensitivity Sweep: PDR vs CVaR-alpha Risk Level', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('CVaR Quantile Fraction alpha (1.0 = Risk-Neutral)')
    axes[1, 0].set_ylabel('Packet Delivery Ratio (%)')
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)
    axes[1, 0].legend()

    # Panel 4: Pareto Trade-off: PDR vs Delivery Latency
    hops_sens = df_sens['Latency_mean'].values
    axes[1, 1].scatter(hops_sens, pdr_sens, color='#3b82f6', s=120, edgecolors='black', zorder=5)
    for a_val, h_val, p_val in zip(alphas_plot, hops_sens, pdr_sens):
        axes[1, 1].annotate(f'alpha={a_val:.2f}', (h_val, p_val), textcoords="offset points", xytext=(8, -4), fontweight='bold')
    axes[1, 1].set_title('Pareto Frontier: Reliability vs Delivery Latency', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Average Delivery Latency (Hops)')
    axes[1, 1].set_ylabel('Packet Delivery Ratio (PDR %)')
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)

    plt.suptitle('Phase 6 Multi-Factor Ablation & CVaR-alpha Risk Sensitivity Analysis', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_path = os.path.join(CFG['figures_dir'], 'phase6_ablation_breakdown.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[Saved Figure] {fig_path}")

    print("\n[Phase 6 Complete] All ablations and sensitivity sweeps executed successfully.")


if __name__ == '__main__':
    main()
