"""
Phase 8: Dynamic State-Adaptive CVaR-alpha(t) & Byzantine Adversarial Resilience Suite

Part 1: Static CVaR (alpha=0.25) vs Dynamic State-Adaptive CVaR (alpha(t, s)).
Part 2: Byzantine Black-Hole Attack Resilience (Model A vs Model B vs Model C+).
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
import torch.nn as nn

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
    compute_qrdqn_loss, masked_argmax,
    TabularQRouter, DQN, DoubleDQNRouter, compute_dqn_loss
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


def train_model_a(cfg: dict, episodes: int = 200, seed: int = 42):
    """Trains Model A Tabular Q-Router."""
    router = TabularQRouter(max_a=cfg['MAX_A'], lr=cfg['tabular_lr'], tau=cfg['tabular_temp'])
    env, _ = build_env(cfg, seed=seed)
    for ep in range(episodes):
        s, m, acts = env.reset_packet()
        if s is None:
            env, _ = build_env(cfg, seed=seed + ep)
            s, m, acts = env.reset_packet()
        if s is None: continue
        done = False
        while not done:
            a = router.select_action(s, m)
            step_res, r, done, _ = env.step(a)
            sn, mn = step_res if not done else (None, None)
            router.update(s, a, r, sn, mn, done)
            if not done: s, m = sn, mn
        if ep % 50 == 0: env.apply_mobility()
    return router


def train_model_b(cfg: dict, uq_model, episodes: int = 200, seed: int = 42):
    """Trains Senior Model B (Double-DQN with static MLP)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    st_dim = cfg.get('STATE_DIM_PHASE3', 26)
    pol = DQN(st_dim, cfg['MAX_A'], hidden=cfg['dqn_hidden']).to(DEVICE)
    tgt = DQN(st_dim, cfg['MAX_A'], hidden=cfg['dqn_hidden']).to(DEVICE)
    tgt.load_state_dict(pol.state_dict())
    opt = torch.optim.Adam(pol.parameters(), lr=cfg.get('lr_dqn', 3e-4))
    rb = ReplayBuffer(cfg['replay_cap'])

    env, _ = build_env(cfg, uq=uq_model, seed=seed)
    eps = cfg['eps_start']
    eps_decay = (cfg['eps_start'] - cfg['eps_end']) / max(1, episodes)

    for ep in range(episodes):
        s, m, acts = env.reset_packet()
        if s is None: continue
        done = False
        while not done:
            if np.random.rand() < eps:
                valid = [i for i, a in enumerate(acts) if a != -1 and m[i] == 1.0]
                a = np.random.choice(valid) if valid else 0
            else:
                with torch.no_grad():
                    s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    m_t = torch.tensor(m, dtype=torch.float32).to(DEVICE)
                    q = pol(s_t)[0]
                    a = masked_argmax(q, m_t)

            step_res, r, done, info = env.step(a)
            if done or step_res is None:
                ns, nm = np.zeros(st_dim, dtype=np.float32), np.zeros(cfg['MAX_A'], dtype=np.float32)
            else:
                ns, nm = step_res

            rb.push(s, m, a, r, ns, nm, done)
            if len(rb) >= cfg['batch_size']:
                b = rb.sample(cfg['batch_size'])
                loss = compute_dqn_loss(pol, tgt, b, gamma=cfg['gamma'], device=str(DEVICE))
                opt.zero_grad()
                loss.backward()
                opt.step()
            s, m = ns, nm
        eps = max(cfg['eps_end'], eps - eps_decay)
        if ep % cfg.get('target_update', 200) == 0:
            tgt.load_state_dict(pol.state_dict())
    return DoubleDQNRouter(pol, device=str(DEVICE))


def train_model_c(cfg: dict, temporal_model, episodes: int = 200, seed: int = 42):
    """Trains Model C+ (Distributional QR-DQN)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    st_dim = cfg.get('STATE_DIM_PHASE4', 45)
    pol = QRDQN(st_dim, cfg['MAX_A'], num_quantiles=cfg['qr_quantiles'],
                hidden=cfg['dqn_hidden'], use_layernorm=True).to(DEVICE)
    tgt = QRDQN(st_dim, cfg['MAX_A'], num_quantiles=cfg['qr_quantiles'],
                hidden=cfg['dqn_hidden'], use_layernorm=True).to(DEVICE)
    tgt.load_state_dict(pol.state_dict())
    opt = torch.optim.Adam(pol.parameters(), lr=cfg.get('qr_lr', 3e-4))
    rb = ReplayBuffer(cfg['replay_cap'])

    env, _ = build_env(cfg, temporal_model=temporal_model, seed=seed)
    eps = cfg['eps_start']
    eps_decay = (cfg['eps_start'] - cfg['eps_end']) / max(1, episodes)

    for ep in range(episodes):
        s, m, acts = env.reset_packet()
        if s is None: continue
        done = False
        while not done:
            if np.random.rand() < eps:
                valid = [i for i, a in enumerate(acts) if a != -1 and m[i] == 1.0]
                a = np.random.choice(valid) if valid else 0
            else:
                with torch.no_grad():
                    s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    m_t = torch.tensor(m, dtype=torch.float32).to(DEVICE)
                    quantiles = pol(s_t)[0]
                    quantiles_sorted, _ = torch.sort(quantiles, dim=-1)
                    k = max(1, int(math.ceil(0.25 * quantiles.size(-1))))
                    cvar_vals = quantiles_sorted[:, :k].mean(dim=-1)
                    a = masked_argmax(cvar_vals, m_t)

            step_res, r, done, info = env.step(a)
            if done or step_res is None:
                ns, nm = np.zeros(st_dim, dtype=np.float32), np.zeros(cfg['MAX_A'], dtype=np.float32)
            else:
                ns, nm = step_res

            rb.push(s, m, a, r, ns, nm, done)
            if len(rb) >= cfg['batch_size']:
                b = rb.sample(cfg['batch_size'])
                loss = compute_qrdqn_loss(pol, tgt, b, gamma=cfg['gamma'], kappa=cfg.get('qr_kappa', 1.0),
                                         alpha=0.25, device=str(DEVICE))
                opt.zero_grad()
                loss.backward()
                opt.step()
            s, m = ns, nm
        eps = max(cfg['eps_end'], eps - eps_decay)
        if ep % cfg.get('target_update', 200) == 0:
            tgt.load_state_dict(pol.state_dict())
    return pol


def evaluate_policy(router, cfg: dict, link_kw: dict, rounds: int = 40,
                    pkts_per_round: int = 25, seed: int = 42,
                    adversary_ids: list[int] = None, adversary_drop_prob: float = 1.0):
    """Runs episodic evaluation with optional Byzantine adversary black-holes."""
    env, nodes = build_env(
        cfg, seed=seed,
        adversary_ids=adversary_ids,
        adversary_drop_prob=adversary_drop_prob,
        **link_kw
    )

    total_pkts = 0
    deliv_pkts = 0
    intercepted_pkts = 0
    total_hops = 0
    first_death = rounds
    alphas_recorded = []

    for r in range(rounds):
        for _ in range(pkts_per_round):
            s, m, acts = env.reset_packet()
            if s is None: break
            total_pkts += 1
            done = False
            while not done:
                if hasattr(router, 'get_adaptive_alpha') and getattr(router, 'adaptive_cvar', False):
                    alphas_recorded.append(router.get_adaptive_alpha(s))

                a = router.select_action(s, m)
                step_res, r_val, done, info = env.step(a)
                if done:
                    if info.get('delivered', False):
                        deliv_pkts += 1
                        total_hops += info.get('hops', 0)
                    if info.get('intercepted', False):
                        intercepted_pkts += 1
                    break
                s, m = step_res

        env.service_queues()
        env.apply_mobility()

        alive = sum(1 for n in nodes if n.alive and not n.is_sink)
        if alive < (len(nodes) - 1) and first_death == rounds:
            first_death = r + 1

    pdr = (deliv_pkts / total_pkts * 100.0) if total_pkts > 0 else 0.0
    interception_rate = (intercepted_pkts / total_pkts * 100.0) if total_pkts > 0 else 0.0
    avg_hops = (total_hops / deliv_pkts) if deliv_pkts > 0 else 0.0
    init_e = float(cfg.get('init_energy', 1.0))
    consumed = [max(0.0, init_e - n.energy) for n in nodes if not n.is_sink]
    gini = gini_coefficient(consumed)
    avg_alpha = float(np.mean(alphas_recorded)) if alphas_recorded else 0.25

    return {
        'pdr': pdr,
        'interception_rate': interception_rate,
        'hops': avg_hops,
        'fnd': first_death,
        'gini': gini,
        'avg_alpha': avg_alpha,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=200, help='RL training episodes')
    parser.add_argument('--rounds', type=int, default=40, help='Evaluation rounds')
    parser.add_argument('--pkts_per_round', type=int, default=25, help='Packets per round')
    parser.add_argument('--seeds', type=str, default='42,101,2024', help='Evaluation seeds')
    args = parser.parse_args()

    ensure_dirs()
    seeds = [int(s) for s in args.seeds.split(',')]

    print("=" * 85)
    print("  PHASE 8: DYNAMIC ADAPTIVE CVaR-alpha & BYZANTINE ADVERSARIAL RESILIENCE")
    print(f"  Device: {DEVICE} | Episodes: {args.episodes} | Rounds: {args.rounds} | Seeds: {seeds}")
    print("=" * 85)
    sys.stdout.flush()

    # Step 1: Prepare Datasets & Link Models
    print("\n[Step 1] Loading Dataset & Initializing Physical Link Predictors...")
    df = load_or_create_dataset(CFG['data_path'], seed=CFG['global_seed'])
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df, seed=CFG['global_seed'])
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_val_s = scaler.transform(X_train), scaler.transform(X_val)

    X_seq_train, y_seq_train = generate_temporal_sequences(X_train_s, y_train, seq_len=CFG['seq_len'], rho=CFG['ar1_rho'], seed=CFG['global_seed'])
    X_seq_val, y_seq_val = generate_temporal_sequences(X_val_s, y_val, seq_len=CFG['seq_len'], rho=CFG['ar1_rho'], seed=CFG['global_seed'])

    print("  -> Training Static Point MLP (LinkUQNet)...")
    net_mlp = train_point_model(LinkUQNet(hidden=CFG['link_hidden'], dropout=CFG['link_dropout']),
                                X_train_s, y_train, X_val_s, y_val, epochs=40, lr=CFG['link_lr'], device=str(DEVICE))
    uq_mlp = UncertaintyModel(net_mlp, scaler, device=str(DEVICE), mc_samples=15)

    print("  -> Training Temporal Link Transformer (20 epochs)...")
    net_trans = train_temporal_model(TemporalLinkTransformer(d_model=CFG['trans_d_model'], nhead=CFG['trans_nhead'], num_layers=CFG['trans_layers']),
                                     X_seq_train, y_seq_train, X_seq_val, y_seq_val, epochs=20, lr=CFG['trans_lr'], device=str(DEVICE))
    temporal_model = TemporalUncertaintyModel(net_trans, scaler, device=str(DEVICE), mc_samples=15, seq_len=CFG['seq_len'])

    # Step 2: Training Routing Policies
    print("\n[Step 2] Training Routing Policies (Standard Grid Baseline)...")
    cfg_base = dict(CFG)
    cfg_base['enable_phase4'] = False

    cfg_p4 = dict(CFG)
    cfg_p4['enable_phase4'] = True

    print("  -> Training Model A (Alanazi Tabular Q)...")
    pol_a = train_model_a(cfg_base, episodes=args.episodes, seed=CFG['global_seed'])

    print("  -> Training Model B (Rahul Anand Double-DQN)...")
    pol_b = train_model_b(cfg_base, uq_mlp, episodes=args.episodes, seed=CFG['global_seed'])

    print("  -> Training Model C+ QR-DQN Network...")
    qrdqn_net = train_model_c(cfg_p4, temporal_model, episodes=args.episodes, seed=CFG['global_seed'])

    # Instantiate Routers
    router_c_static = QRDQNRouter(qrdqn_net, alpha=0.25, device=str(DEVICE), adaptive_cvar=False)
    router_c_dynamic = QRDQNRouter(qrdqn_net, alpha=0.25, device=str(DEVICE), adaptive_cvar=True)

    # =========================================================================
    # PART 1: Dynamic State-Adaptive CVaR vs Static CVaR Benchmark
    # =========================================================================
    print("\n" + "=" * 85)
    print("  PART 1: STATIC CVaR (alpha=0.25) vs DYNAMIC STATE-ADAPTIVE CVaR (alpha(t, s))")
    print("=" * 85)
    sys.stdout.flush()

    dynamic_configs = [
        ('Static CVaR (alpha=0.25)', router_c_static, 1.0),
        ('Dynamic Adaptive CVaR (alpha(t))', router_c_dynamic, 1.0),
        ('Static CVaR [Energy-Depleted E0=0.4J]', router_c_static, 0.4),
        ('Dynamic Adaptive CVaR [Energy-Depleted E0=0.4J]', router_c_dynamic, 0.4),
    ]

    dyn_records = []
    for d_name, router, e_init in dynamic_configs:
        cfg_eval = dict(cfg_p4)
        cfg_eval['init_energy'] = e_init

        res_list = []
        for s in seeds:
            res = evaluate_policy(router, cfg_eval, {'temporal_model': temporal_model},
                                  rounds=args.rounds, pkts_per_round=args.pkts_per_round, seed=s)
            res_list.append(res)

        pdr_m, pdr_s = float(np.mean([r['pdr'] for r in res_list])), float(np.std([r['pdr'] for r in res_list]))
        hop_m, hop_s = float(np.mean([r['hops'] for r in res_list])), float(np.std([r['hops'] for r in res_list]))
        fnd_m, fnd_s = float(np.mean([r['fnd'] for r in res_list])), float(np.std([r['fnd'] for r in res_list]))
        gini_m, gini_s = float(np.mean([r['gini'] for r in res_list])), float(np.std([r['gini'] for r in res_list]))
        alpha_m = float(np.mean([r['avg_alpha'] for r in res_list]))

        print(f"  -> {d_name:<46} | PDR: {pdr_m:5.1f}% +/- {pdr_s:4.1f}% | Hops: {hop_m:4.2f} | FND: {fnd_m:4.1f} | Gini: {gini_m:.3f} | Avg alpha: {alpha_m:.2f}")
        sys.stdout.flush()

        dyn_records.append({
            'Configuration': d_name,
            'Initial_Energy': e_init,
            'PDR_mean': pdr_m,
            'PDR_std': pdr_s,
            'Latency_hops_mean': hop_m,
            'Latency_hops_std': hop_s,
            'FND_rounds_mean': fnd_m,
            'FND_rounds_std': fnd_s,
            'Gini_mean': gini_m,
            'Gini_std': gini_s,
            'Avg_Alpha': alpha_m,
        })

    df_dyn = pd.DataFrame(dyn_records)
    dyn_csv_path = os.path.join(CFG['artifacts_dir'], 'phase8_dynamic_cvar_benchmark.csv')
    df_dyn.to_csv(dyn_csv_path, index=False)
    print(f"[Saved Artifact] {dyn_csv_path}")

    # =========================================================================
    # PART 2: Byzantine Black-Hole Adversarial Resilience Benchmark
    # =========================================================================
    print("\n" + "=" * 85)
    print("  PART 2: BYZANTINE BLACK-HOLE ADVERSARIAL RESILIENCE BENCHMARK")
    print("  (Adversary Nodes: 4 & 12 | Drop Probability: 100% | Black-Hole Interception)")
    print("=" * 85)
    sys.stdout.flush()

    adversary_nodes = [4, 12]  # Two strategic intermediate relays
    adversary_models = [
        ('Model A (Alanazi Tabular Q)', pol_a, cfg_base, {}),
        ('Model B (Rahul Anand Double-DQN)', pol_b, cfg_base, {'uq': uq_mlp}),
        ('Model C+ (Static CVaR-0.25)', router_c_static, cfg_p4, {'temporal_model': temporal_model}),
        ('Model C+ (Dynamic Adaptive CVaR)', router_c_dynamic, cfg_p4, {'temporal_model': temporal_model}),
    ]

    adv_records = []
    for m_name, router, cfg_m, l_kw in adversary_models:
        res_list = []
        for s in seeds:
            res = evaluate_policy(
                router, cfg_m, l_kw,
                rounds=args.rounds, pkts_per_round=args.pkts_per_round, seed=s,
                adversary_ids=adversary_nodes, adversary_drop_prob=1.0
            )
            res_list.append(res)

        pdr_m, pdr_s = float(np.mean([r['pdr'] for r in res_list])), float(np.std([r['pdr'] for r in res_list]))
        ir_m, ir_s = float(np.mean([r['interception_rate'] for r in res_list])), float(np.std([r['interception_rate'] for r in res_list]))
        hop_m, hop_s = float(np.mean([r['hops'] for r in res_list])), float(np.std([r['hops'] for r in res_list]))
        fnd_m, fnd_s = float(np.mean([r['fnd'] for r in res_list])), float(np.std([r['fnd'] for r in res_list]))
        gini_m, gini_s = float(np.mean([r['gini'] for r in res_list])), float(np.std([r['gini'] for r in res_list]))

        print(f"  -> {m_name:<34} | PDR: {pdr_m:5.1f}% +/- {pdr_s:4.1f}% | Black-Hole Intercepted: {ir_m:5.1f}% +/- {ir_s:4.1f}% | Hops: {hop_m:4.2f} | Gini: {gini_m:.3f}")
        sys.stdout.flush()

        adv_records.append({
            'Model': m_name,
            'PDR_mean': pdr_m,
            'PDR_std': pdr_s,
            'Interception_Rate_mean': ir_m,
            'Interception_Rate_std': ir_s,
            'Latency_hops_mean': hop_m,
            'Latency_hops_std': hop_s,
            'FND_rounds_mean': fnd_m,
            'FND_rounds_std': fnd_s,
            'Gini_mean': gini_m,
            'Gini_std': gini_s,
        })

    df_adv = pd.DataFrame(adv_records)
    adv_csv_path = os.path.join(CFG['artifacts_dir'], 'phase8_adversarial_resilience.csv')
    df_adv.to_csv(adv_csv_path, index=False)
    print(f"[Saved Artifact] {adv_csv_path}")

    # =========================================================================
    # Step 5: Publication-Grade 4-Panel Visualization
    # =========================================================================
    print("\n[Step 5] Generating Publication-Grade Phase 8 Figure...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Dynamic CVaR vs Static CVaR under Normal vs Depleted Energy
    labels_dyn = ['Normal (E0=1.0J)', 'Depleted (E0=0.4J)']
    pdr_static = [df_dyn.loc[df_dyn['Configuration'].str.startswith('Static CVaR (alpha=0.25)'), 'PDR_mean'].values[0],
                  df_dyn.loc[df_dyn['Configuration'].str.startswith('Static CVaR [Energy-Depleted'), 'PDR_mean'].values[0]]
    pdr_dyn = [df_dyn.loc[df_dyn['Configuration'].str.startswith('Dynamic Adaptive CVaR (alpha(t))'), 'PDR_mean'].values[0],
               df_dyn.loc[df_dyn['Configuration'].str.startswith('Dynamic Adaptive CVaR [Energy-Depleted'), 'PDR_mean'].values[0]]

    x = np.arange(len(labels_dyn))
    width = 0.35
    axes[0, 0].bar(x - width/2, pdr_static, width, label='Static CVaR (α=0.25)', color='#3b82f6', alpha=0.85, edgecolor='black')
    axes[0, 0].bar(x + width/2, pdr_dyn, width, label='Dynamic State-Adaptive CVaR α(t)', color='#10b981', alpha=0.85, edgecolor='black')
    axes[0, 0].set_title('Dynamic Risk Adaptation: PDR under Energy Stress', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Packet Delivery Ratio (PDR %)')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels_dyn, fontsize=10)
    axes[0, 0].set_ylim(0, 105)
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.5)
    axes[0, 0].legend()

    # Panel 2: Theoretical Dynamic Alpha Function Mapping
    batt_sweep = np.linspace(0.0, 1.0, 100)
    alpha_mapped = np.clip(0.10 + 0.65 * (batt_sweep ** 1.5), 0.10, 0.75)
    axes[0, 1].plot(batt_sweep * 100, alpha_mapped, color='#8b5cf6', linewidth=2.8)
    axes[0, 1].axhline(y=0.25, color='#ef4444', linestyle='--', label='Static Baseline α = 0.25')
    axes[0, 1].axvspan(0, 30, color='#fee2e2', alpha=0.5, label='Critical Battery Region (α -> 0.10)')
    axes[0, 1].axvspan(70, 100, color='#dcfce7', alpha=0.5, label='Healthy Battery Region (α -> 0.75)')
    axes[0, 1].set_title('Autonomous Risk Function: α(s) vs Battery Level', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Node Remaining Battery (%)')
    axes[0, 1].set_ylabel('CVaR Quantile Fraction α (Risk Tolerance)')
    axes[0, 1].set_ylim(0.05, 0.85)
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)
    axes[0, 1].legend(loc='upper left')

    # Panel 3: Byzantine Black-Hole PDR Comparison
    labels_adv = ['Model A\n(Tabular Q)', 'Model B\n(Senior DQN)', 'Model C+\n(Static CVaR)', 'Model C+\n(Dynamic CVaR)']
    pdr_adv = df_adv['PDR_mean'].values
    pdr_err = df_adv['PDR_std'].values
    colors_adv = ['#ef4444', '#f59e0b', '#3b82f6', '#10b981']

    bars3 = axes[1, 0].bar(labels_adv, pdr_adv, yerr=pdr_err, capsize=5, color=colors_adv, alpha=0.85, edgecolor='black')
    axes[1, 0].set_title('Byzantine Black-Hole Resilience: Packet Delivery Ratio', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Packet Delivery Ratio (PDR %)')
    axes[1, 0].set_ylim(0, 105)
    axes[1, 0].grid(axis='y', linestyle='--', alpha=0.5)
    for b in bars3:
        h = b.get_height()
        axes[1, 0].text(b.get_x() + b.get_width()/2.0, h + 3.0, f'{h:.1f}%', ha='center', fontweight='bold', fontsize=10)

    # Panel 4: Adversary Interception Rate (Black-Hole Traffic Captured)
    ir_adv = df_adv['Interception_Rate_mean'].values
    ir_err = df_adv['Interception_Rate_std'].values
    bars4 = axes[1, 1].bar(labels_adv, ir_adv, yerr=ir_err, capsize=5, color=['#b91c1c', '#d97706', '#2563eb', '#059669'], alpha=0.85, edgecolor='black')
    axes[1, 1].set_title('Adversary Attack Impact: Interception Rate (% of Packets)', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Packets Intercepted by Black-Holes (%)')
    axes[1, 1].set_ylim(0, 80)
    axes[1, 1].grid(axis='y', linestyle='--', alpha=0.5)
    for b in bars4:
        h = b.get_height()
        axes[1, 1].text(b.get_x() + b.get_width()/2.0, h + 2.0, f'{h:.1f}%', ha='center', fontweight='bold', fontsize=10)

    plt.suptitle('Phase 8: Dynamic State-Adaptive Risk & Byzantine Adversarial Resilience Benchmark', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_path = os.path.join(CFG['figures_dir'], 'phase8_dynamic_and_adversarial.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[Saved Figure] {fig_path}")

    print("\n[Phase 8 Complete] Both suites finished successfully.")


if __name__ == '__main__':
    main()
