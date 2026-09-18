"""
Phase 4 Ablation Runner:
Comparing Model C Baseline (Phase 3) vs Model C+ (Phase 4 Adaptive Multi-Objective & Context-Aware)
Under Bursty Traffic Load in 6G WSN.
"""
import os
import sys
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
from src.models_link import TemporalLinkTransformer, TemporalUncertaintyModel, train_temporal_model
from src.env import build_env
from src.models_rl import QRDQN, QRDQNRouter, ReplayBuffer, compute_qrdqn_loss, masked_argmax

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def gini_coefficient(array):
    """Computes the Gini coefficient of an array (measure of energy consumption inequality)."""
    array = np.array(array, dtype=np.float64).flatten()
    if np.amin(array) < 0:
        array -= np.amin(array)
    array += 1e-9
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return float((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))


def train_c_router(cfg: dict, temporal_model, state_dim: int, episodes: int = 500, seed: int = 42):
    """Trains a QR-DQN router under the specified environment configuration."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    pol = QRDQN(state_dim, cfg['MAX_A'], num_quantiles=cfg['qr_quantiles'],
                hidden=cfg['dqn_hidden'], use_layernorm=True).to(DEVICE)
    tgt = QRDQN(state_dim, cfg['MAX_A'], num_quantiles=cfg['qr_quantiles'],
                hidden=cfg['dqn_hidden'], use_layernorm=True).to(DEVICE)
    tgt.load_state_dict(pol.state_dict())
    
    opt = torch.optim.Adam(pol.parameters(), lr=cfg['qr_lr'])
    rb = ReplayBuffer(cfg['replay_cap'])
    env, _ = build_env(cfg, temporal_model=temporal_model, seed=seed)
    
    eps = cfg['eps_start']
    for ep in range(episodes):
        s, m, acts = env.reset_packet()
        if s is None or ep % 200 == 0:
            env, _ = build_env(cfg, temporal_model=temporal_model, seed=seed + ep)
            s, m, acts = env.reset_packet()
        if s is None:
            continue
            
        done = False
        while not done:
            valid_acts = [i for i in range(cfg['MAX_A']) if m[i] == 1.0]
            if np.random.uniform(0, 1) < eps:
                a = np.random.choice(valid_acts) if valid_acts else 0
            else:
                with torch.no_grad():
                    q = pol(torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(DEVICE))[0]
                    k = max(1, int(math.ceil(cfg['cvar_alpha'] * cfg['qr_quantiles'])))
                    q_sorted, _ = torch.sort(q, dim=-1)
                    cvar_vals = q_sorted[:, :k].mean(dim=-1)
                    a = masked_argmax(cvar_vals, torch.tensor(m).to(DEVICE))
                    
            step_res, r, done, _ = env.step(a)
            sn = step_res[0] if not done else np.zeros(state_dim, dtype=np.float32)
            mn = step_res[1] if not done else np.zeros(cfg['MAX_A'], dtype=np.float32)
            rb.push(s, m, a, r, sn, mn, float(done))
            if not done:
                s, m = sn, mn
                
            if len(rb) >= cfg['batch_size']:
                batch = rb.sample(cfg['batch_size'])
                loss = compute_qrdqn_loss(pol, tgt, batch, gamma=cfg['gamma'],
                                          alpha=cfg['cvar_alpha'], device=str(DEVICE))
                opt.zero_grad()
                loss.backward()
                opt.step()
                
        eps = max(cfg['eps_end'], eps * cfg['eps_decay'])
        if ep % 250 == 0:
            tgt.load_state_dict(pol.state_dict())
        if ep % 50 == 0:
            env.service_queues()
            env.apply_mobility()
        if (ep + 1) % 50 == 0 or (ep + 1) == episodes:
            print(f"    Episode {ep + 1}/{episodes} (Replay: {len(rb)} | eps: {eps:.3f})")
            sys.stdout.flush()
            
    return QRDQNRouter(pol, alpha=cfg['cvar_alpha'], device=str(DEVICE))


def evaluate_ablation(router, cfg: dict, temporal_model, rounds: int = 80,
                      pkts_per_round: int = 35, seed: int = 42):
    """Evaluates router under bursty traffic conditions."""
    env, nodes = build_env(cfg, temporal_model=temporal_model, seed=seed)
    total_pkts = 0
    deliv_pkts = 0
    buffer_drops = 0
    total_hops = 0
    first_death = rounds
    
    for r in range(rounds):
        for _ in range(pkts_per_round):
            s, m, acts = env.reset_packet()
            if s is None:
                break
            total_pkts += 1
            done = False
            while not done:
                a = router.select_action(s, m)
                step_res, r_val, done, info = env.step(a)
                if done:
                    if info.get('delivered', False):
                        deliv_pkts += 1
                        total_hops += info.get('hops', 0)
                    elif info.get('reason') == 'buffer_overflow':
                        buffer_drops += 1
                    break
                s, m = step_res
                
        # Service queues and apply mobility each round
        env.service_queues()
        env.apply_mobility()
        
        # Check first node death
        alive = sum(1 for n in nodes if n.alive and not n.is_sink)
        if alive < (len(nodes) - 1) and first_death == rounds:
            first_death = r + 1
            
        if (r + 1) % 15 == 0 or (r + 1) == rounds:
            cur_pdr = (deliv_pkts / max(1, total_pkts)) * 100.0
            print(f"      Round {r + 1}/{rounds} | Packets: {total_pkts} | Delivered: {deliv_pkts} ({cur_pdr:.1f}%)")
            sys.stdout.flush()
            
    pdr = (deliv_pkts / total_pkts * 100.0) if total_pkts > 0 else 0.0
    buf_drop_rate = (buffer_drops / total_pkts * 100.0) if total_pkts > 0 else 0.0
    avg_hops = (total_hops / deliv_pkts) if deliv_pkts > 0 else 0.0
    surviving = sum(1 for n in nodes if n.alive and not n.is_sink)
    
    # Calculate Gini coefficient on energy consumed across nodes
    consumed_energy = [max(0.0, float(cfg['init_energy']) - n.energy) for n in nodes if not n.is_sink]
    e_gini = gini_coefficient(consumed_energy)
    
    return {
        'pdr': pdr,
        'buf_drop_rate': buf_drop_rate,
        'avg_hops': avg_hops,
        'first_death': first_death,
        'surviving': surviving,
        'gini': e_gini
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=300, help='RL training episodes')
    parser.add_argument('--rounds', type=int, default=60, help='Evaluation rounds')
    parser.add_argument('--pkts_per_round', type=int, default=30, help='Packets per round (burst traffic)')
    parser.add_argument('--seeds', type=str, default='42,101,2024', help='Evaluation seeds')
    parser.add_argument('--trans_epochs', type=int, default=25, help='Transformer training epochs')
    parser.add_argument('--mc_samples', type=int, default=15, help='MC dropout sample count')
    args = parser.parse_args()
    
    ensure_dirs()
    seeds = [int(s) for s in args.seeds.split(',')]
    
    print("=" * 80)
    print("  PHASE 4 ABLATION BENCHMARK: Model C (Phase 3) vs Model C+ (Phase 4)")
    print(f"  Device: {DEVICE} | Episodes: {args.episodes} | Rounds: {args.rounds} | Seeds: {seeds}")
    print("=" * 80)
    sys.stdout.flush()
    
    # 1. Dataset & Temporal Transformer
    print("\n[Step 1] Loading Dataset & Initializing Temporal Transformer...")
    sys.stdout.flush()
    df = load_or_create_dataset(CFG['data_path'], seed=CFG['global_seed'])
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df, seed=CFG['global_seed'])
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    
    X_seq_train, y_seq_train = generate_temporal_sequences(
        X_train_s, y_train, seq_len=CFG['seq_len'], rho=CFG['ar1_rho'], seed=CFG['global_seed']
    )
    X_seq_val, y_seq_val = generate_temporal_sequences(
        X_val_s, y_val, seq_len=CFG['seq_len'], rho=CFG['ar1_rho'], seed=CFG['global_seed']
    )
    
    print(f"  -> Training Temporal Transformer ({args.trans_epochs} epochs)...")
    sys.stdout.flush()
    net_c = train_temporal_model(
        TemporalLinkTransformer(d_model=CFG['trans_d_model'], nhead=CFG['trans_nhead'], num_layers=CFG['trans_layers']),
        X_seq_train, y_seq_train, X_seq_val, y_seq_val,
        epochs=args.trans_epochs, lr=CFG['trans_lr'], device=str(DEVICE)
    )
    temporal_model = TemporalUncertaintyModel(net_c, scaler, device=str(DEVICE),
                                              mc_samples=args.mc_samples, seq_len=CFG['seq_len'])
    
    # Config definitions
    cfg_base = dict(CFG)
    cfg_base['enable_phase4'] = False
    state_dim_base = CFG['STATE_DIM_PHASE3']
    
    cfg_p4 = dict(CFG)
    cfg_p4['enable_phase4'] = True
    cfg_p4['max_queue_cap'] = 15          # Buffer pressure
    cfg_p4['queue_service_rate'] = 3
    state_dim_p4 = CFG['STATE_DIM_PHASE4']
    
    # 2. Train & Evaluate Baseline Model C (Phase 3)
    print("\n[Step 2] Evaluating Model C (Phase 3 Baseline: 7D State, Linear Reward)...")
    router_c = train_c_router(cfg_base, temporal_model, state_dim_base, episodes=args.episodes, seed=seeds[0])
    
    results_base = []
    for s in seeds:
        res = evaluate_ablation(router_c, cfg_base, temporal_model, rounds=args.rounds,
                                pkts_per_round=args.pkts_per_round, seed=s)
        results_base.append(res)
        print(f"  Seed {s}: PDR={res['pdr']:.1f}% | DropRate={res['buf_drop_rate']:.1f}% | "
              f"Hops={res['avg_hops']:.2f} | FND={res['first_death']} | Gini={res['gini']:.3f}")
              
    # 3. Train & Evaluate Model C+ (Phase 4)
    print("\n[Step 3] Evaluating Model C+ (Phase 4: 10D State, Adaptive Barrier & Queue Congestion)...")
    router_c_plus = train_c_router(cfg_p4, temporal_model, state_dim_p4, episodes=args.episodes, seed=seeds[0])
    
    results_p4 = []
    for s in seeds:
        res = evaluate_ablation(router_c_plus, cfg_p4, temporal_model, rounds=args.rounds,
                                pkts_per_round=args.pkts_per_round, seed=s)
        results_p4.append(res)
        print(f"  Seed {s}: PDR={res['pdr']:.1f}% | DropRate={res['buf_drop_rate']:.1f}% | "
              f"Hops={res['avg_hops']:.2f} | FND={res['first_death']} | Gini={res['gini']:.3f}")
              
    # 4. Summary Leaderboard Table
    metrics = ['pdr', 'buf_drop_rate', 'avg_hops', 'first_death', 'surviving', 'gini']
    summary = []
    for name, res_list in [('Model C (Phase 3 Baseline)', results_base),
                           ('Model C+ (Phase 4 Adaptive)', results_p4)]:
        row = {'Model': name}
        for m in metrics:
            vals = [r[m] for r in res_list]
            row[f'{m}_mean'] = float(np.mean(vals))
            row[f'{m}_std'] = float(np.std(vals))
        summary.append(row)
        
    df_res = pd.DataFrame(summary)
    csv_path = os.path.join(CFG['artifacts_dir'], 'phase4_ablation_results.csv')
    df_res.to_csv(csv_path, index=False)
    print(f"\n[Saved Artifact] {csv_path}")
    
    # Print formatted comparison table
    print("\n" + "=" * 90)
    print(f"{'Metric':<32} | {'Model C (Phase 3 Baseline)':<25} | {'Model C+ (Phase 4 Adaptive)':<25}")
    print("-" * 90)
    labels = [
        ('pdr', 'Packet Delivery Ratio (%)'),
        ('buf_drop_rate', 'Buffer Overflow Drop Rate (%)'),
        ('avg_hops', 'Average Latency (Hops)'),
        ('first_death', 'First Node Death (Rounds)'),
        ('surviving', 'Surviving Nodes at End'),
        ('gini', 'Energy Depletion Gini (0=Equal)'),
    ]
    for key, name in labels:
        b_mean, b_std = summary[0][f'{key}_mean'], summary[0][f'{key}_std']
        p_mean, p_std = summary[1][f'{key}_mean'], summary[1][f'{key}_std']
        print(f"{name:<32} | {b_mean:6.2f} ± {b_std:4.2f}{'':<12} | {p_mean:6.2f} ± {p_std:4.2f}")
    print("=" * 90)
    
    # 5. Generate Publication Chart
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    models = ['Model C (Phase 3)', 'Model C+ (Phase 4)']
    colors = ['#3b82f6', '#10b981']
    
    # 1. PDR Comparison
    pdr_means = [summary[0]['pdr_mean'], summary[1]['pdr_mean']]
    pdr_stds = [summary[0]['pdr_std'], summary[1]['pdr_std']]
    axes[0].bar(models, pdr_means, yerr=pdr_stds, capsize=5, color=colors, alpha=0.85, edgecolor='black')
    axes[0].set_title('Packet Delivery Ratio (PDR %)', fontweight='bold')
    axes[0].set_ylabel('PDR (%)')
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    for i, v in enumerate(pdr_means):
        axes[0].text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')
        
    # 2. Buffer Drop Rate
    drop_means = [summary[0]['buf_drop_rate_mean'], summary[1]['buf_drop_rate_mean']]
    drop_stds = [summary[0]['buf_drop_rate_std'], summary[1]['buf_drop_rate_std']]
    axes[1].bar(models, drop_means, yerr=drop_stds, capsize=5, color=['#ef4444', '#10b981'], alpha=0.85, edgecolor='black')
    axes[1].set_title('Buffer Overflow Drop Rate (%)', fontweight='bold')
    axes[1].set_ylabel('Drop Rate (%)')
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    for i, v in enumerate(drop_means):
        axes[1].text(i, v + 0.5, f"{v:.1f}%", ha='center', fontweight='bold')
        
    # 3. First Node Death
    fnd_means = [summary[0]['first_death_mean'], summary[1]['first_death_mean']]
    fnd_stds = [summary[0]['first_death_std'], summary[1]['first_death_std']]
    axes[2].bar(models, fnd_means, yerr=fnd_stds, capsize=5, color=colors, alpha=0.85, edgecolor='black')
    axes[2].set_title('First Node Death (Rounds)', fontweight='bold')
    axes[2].set_ylabel('Rounds (Higher is Better)')
    axes[2].grid(axis='y', linestyle='--', alpha=0.5)
    for i, v in enumerate(fnd_means):
        axes[2].text(i, v + 1, f"{v:.1f} rnd", ha='center', fontweight='bold')
        
    # 4. Energy Gini Coefficient
    gini_means = [summary[0]['gini_mean'], summary[1]['gini_mean']]
    gini_stds = [summary[0]['gini_std'], summary[1]['gini_std']]
    axes[3].bar(models, gini_means, yerr=gini_stds, capsize=5, color=['#f59e0b', '#10b981'], alpha=0.85, edgecolor='black')
    axes[3].set_title('Energy Inequality Gini (Lower is Better)', fontweight='bold')
    axes[3].set_ylabel('Gini Index')
    axes[3].grid(axis='y', linestyle='--', alpha=0.5)
    for i, v in enumerate(gini_means):
        axes[3].text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')
        
    plt.tight_layout()
    fig_path = os.path.join(CFG['figures_dir'], 'phase4_ablation_comparison.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[Saved Figure] {fig_path}")
    print("\n[Phase 4 Ablation Complete] All benchmarks executed successfully.")


if __name__ == '__main__':
    main()
