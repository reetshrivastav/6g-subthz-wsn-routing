"""
Master Execution Script for Review 2 Comparative Benchmark:
A (Alanazi et al., 2025) vs B (Rahul Anand, 2026) vs C (Extension MVP)
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

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss

from src.config import CFG, STATE_DIM, ensure_dirs
from src.dataset import (
    load_or_create_dataset, split_dataset, save_scaler_npz,
    generate_temporal_sequences
)
from src.models_link import (
    LinkMLP, LinkUQNet, UncertaintyModel,
    TemporalLinkTransformer, TemporalUncertaintyModel,
    train_point_model, train_temporal_model
)
from src.env import build_env
from src.models_rl import (
    TabularQRouter, DQN, DoubleDQNRouter,
    QRDQN, QRDQNRouter, ReplayBuffer,
    compute_dqn_loss, compute_qrdqn_loss, masked_argmax
)
from src.eval import run_benchmark_leaderboard, plot_comparative_charts

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    ensure_dirs()
    print("=" * 80)
    print(f"  WSN 6G COMPARATIVE BENCHMARK: A vs B vs C (Device: {DEVICE})")
    print("=" * 80)

    # 1. Dataset & Scaling
    df = load_or_create_dataset(CFG['data_path'], seed=CFG['global_seed'])
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        df, test_size=CFG['test_size'], val_size=CFG['val_size'], seed=CFG['global_seed']
    )
    scaler = StandardScaler().fit(X_train)
    save_scaler_npz(scaler, f"{CFG['artifacts_dir']}/scaler.npz")
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Sequence Generation for Model C
    X_seq_train, y_seq_train = generate_temporal_sequences(
        X_train_s, y_train, seq_len=CFG['seq_len'], rho=CFG['ar1_rho'], seed=CFG['global_seed']
    )
    X_seq_val, y_seq_val = generate_temporal_sequences(
        X_val_s, y_val, seq_len=CFG['seq_len'], rho=CFG['ar1_rho'], seed=CFG['global_seed']
    )
    X_seq_test, y_seq_test = generate_temporal_sequences(
        X_test_s, y_test, seq_len=CFG['seq_len'], rho=CFG['ar1_rho'], seed=CFG['global_seed']
    )

    # 2. Train Link Models
    print("\n[Tier 1] Training Link Predictor Models...")
    # Model A
    print("  -> Training Model A: LinkMLP (10 neurons, deterministic)...")
    net_a = train_point_model(
        LinkMLP(hidden=CFG['ann_hidden']), X_train_s, y_train, X_val_s, y_val,
        epochs=CFG['link_epochs'], lr=CFG['link_lr'], device=str(DEVICE)
    )

    # Model B
    print("  -> Training Model B: LinkUQNet (32 neurons, MC Dropout)...")
    net_b = train_point_model(
        LinkUQNet(hidden=CFG['link_hidden'], dropout=CFG['link_dropout']),
        X_train_s, y_train, X_val_s, y_val,
        epochs=CFG['link_epochs'], lr=CFG['link_lr'], device=str(DEVICE)
    )
    uq_b = UncertaintyModel(net_b, scaler, device=str(DEVICE), mc_samples=CFG['mc_samples'])

    # Model C
    print("  -> Training Model C: TemporalLinkTransformer (2 layers, 2 heads, d=32)...")
    net_c = train_temporal_model(
        TemporalLinkTransformer(d_model=CFG['trans_d_model'], nhead=CFG['trans_nhead'], num_layers=CFG['trans_layers']),
        X_seq_train, y_seq_train, X_seq_val, y_seq_val,
        epochs=CFG['trans_epochs'], lr=CFG['trans_lr'], device=str(DEVICE)
    )
    uq_c = TemporalUncertaintyModel(net_c, scaler, device=str(DEVICE), mc_samples=CFG['mc_samples'], seq_len=CFG['seq_len'])

    # 3. Train RL Agents
    print("\n[Tier 2] Training Reinforcement Learning Routing Policies...")
    # Model A: Tabular Q
    print("  -> Training Model A: Tabular Q-Router...")
    router_a = TabularQRouter(max_a=CFG['MAX_A'], lr=CFG['tabular_lr'], tau=CFG['tabular_temp'])
    env_a, _ = build_env(CFG, uq=None, seed=CFG['global_seed'])
    for ep in range(1500):
        s, m, acts = env_a.reset_packet()
        if s is None:
            env_a, _ = build_env(CFG, uq=None, seed=CFG['global_seed'] + ep)
            s, m, acts = env_a.reset_packet()
        done = False
        while not done:
            a = router_a.select_action(s, m)
            step_res, r, done, _ = env_a.step(a)
            sn, mn = step_res if not done else (None, None)
            router_a.update(s, a, r, sn, mn, done)
            if not done: s, m = sn, mn
        if ep % 200 == 0: env_a.apply_mobility()

    # Model B: Double-DQN
    print("  -> Training Model B: Double-DQN (Risk-Neutral)...")
    pol_b = DQN(STATE_DIM, CFG['MAX_A'], hidden=CFG['dqn_hidden']).to(DEVICE)
    tgt_b = DQN(STATE_DIM, CFG['MAX_A'], hidden=CFG['dqn_hidden']).to(DEVICE)
    tgt_b.load_state_dict(pol_b.state_dict())
    opt_b = torch.optim.Adam(pol_b.parameters(), lr=CFG['lr_dqn'])
    rb_b = ReplayBuffer(CFG['replay_cap'])
    env_b, _ = build_env(CFG, uq=uq_b, seed=CFG['global_seed'])

    eps = CFG['eps_start']
    for ep in range(CFG['dqn_episodes']):
        s, m, acts = env_b.reset_packet()
        if s is None or ep % 500 == 0:
            env_b, _ = build_env(CFG, uq=uq_b, seed=CFG['global_seed'] + ep)
            s, m, acts = env_b.reset_packet()
        done = False
        while not done:
            valid_acts = [i for i in range(CFG['MAX_A']) if m[i] == 1.0]
            if np.random.uniform(0, 1) < eps:
                a = np.random.choice(valid_acts) if valid_acts else 0
            else:
                with torch.no_grad():
                    q = pol_b(torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(DEVICE))[0]
                    a = masked_argmax(q, torch.tensor(m).to(DEVICE))
            step_res, r, done, _ = env_b.step(a)
            sn = step_res[0] if not done else np.zeros(STATE_DIM, dtype=np.float32)
            mn = step_res[1] if not done else np.zeros(CFG['MAX_A'], dtype=np.float32)
            rb_b.push(s, m, a, r, sn, mn, float(done))
            if not done: s, m = sn, mn

            if len(rb_b) >= CFG['batch_size']:
                batch = rb_b.sample(CFG['batch_size'])
                loss = compute_dqn_loss(pol_b, tgt_b, batch, gamma=CFG['gamma'], use_double=True, device=str(DEVICE))
                opt_b.zero_grad()
                loss.backward()
                opt_b.step()

        eps = max(CFG['eps_end'], eps * CFG['eps_decay'])
        if ep % 500 == 0: tgt_b.load_state_dict(pol_b.state_dict())
        if ep % 100 == 0: env_b.apply_mobility()
    router_b = DoubleDQNRouter(pol_b, device=str(DEVICE))

    # Model C: QR-DQN with CVaR
    print("  -> Training Model C: Distributional QR-DQN with CVaR-alpha...")
    pol_c = QRDQN(STATE_DIM, CFG['MAX_A'], num_quantiles=CFG['qr_quantiles'], hidden=CFG['dqn_hidden']).to(DEVICE)
    tgt_c = QRDQN(STATE_DIM, CFG['MAX_A'], num_quantiles=CFG['qr_quantiles'], hidden=CFG['dqn_hidden']).to(DEVICE)
    tgt_c.load_state_dict(pol_c.state_dict())
    opt_c = torch.optim.Adam(pol_c.parameters(), lr=CFG['qr_lr'])
    rb_c = ReplayBuffer(CFG['replay_cap'])
    env_c, _ = build_env(CFG, temporal_model=uq_c, seed=CFG['global_seed'])

    eps = CFG['eps_start']
    for ep in range(CFG['dqn_episodes']):
        s, m, acts = env_c.reset_packet()
        if s is None or ep % 500 == 0:
            env_c, _ = build_env(CFG, temporal_model=uq_c, seed=CFG['global_seed'] + ep)
            s, m, acts = env_c.reset_packet()
        done = False
        while not done:
            valid_acts = [i for i in range(CFG['MAX_A']) if m[i] == 1.0]
            if np.random.uniform(0, 1) < eps:
                a = np.random.choice(valid_acts) if valid_acts else 0
            else:
                with torch.no_grad():
                    q_dist = pol_c(torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(DEVICE))[0]
                    q_sorted, _ = torch.sort(q_dist, dim=-1)
                    k = max(1, int(np.ceil(CFG['cvar_alpha'] * CFG['qr_quantiles'])))
                    cvar = q_sorted[:, :k].mean(dim=-1)
                    a = masked_argmax(cvar, torch.tensor(m).to(DEVICE))
            step_res, r, done, _ = env_c.step(a)
            sn = step_res[0] if not done else np.zeros(STATE_DIM, dtype=np.float32)
            mn = step_res[1] if not done else np.zeros(CFG['MAX_A'], dtype=np.float32)
            rb_c.push(s, m, a, r, sn, mn, float(done))
            if not done: s, m = sn, mn

            if len(rb_c) >= CFG['batch_size']:
                batch = rb_c.sample(CFG['batch_size'])
                loss = compute_qrdqn_loss(pol_c, tgt_c, batch, gamma=CFG['gamma'], alpha=CFG['cvar_alpha'], device=str(DEVICE))
                opt_c.zero_grad()
                loss.backward()
                opt_c.step()

        eps = max(CFG['eps_end'], eps * CFG['eps_decay'])
        if ep % 500 == 0: tgt_c.load_state_dict(pol_c.state_dict())
        if ep % 100 == 0: env_c.apply_mobility()
    router_c = QRDQNRouter(pol_c, alpha=CFG['cvar_alpha'], device=str(DEVICE))

    # 4. Multi-Seed Leaderboard
    print("\n[Tier 3] Running Multi-Seed Comparative Leaderboards across Seeds [42, 43, 44]...")
    routers = {
        'Model A (Base Paper - Alanazi)': {'router': router_a, 'uq': None, 'temporal_model': None},
        'Model B (Senior - Rahul Anand)': {'router': router_b, 'uq': uq_b, 'temporal_model': None},
        'Model C (Extension MVP - Ours)': {'router': router_c, 'uq': None, 'temporal_model': uq_c},
    }

    print("\nEvaluating Standard Topology (40 Nodes, 40m Range)...")
    summary_std, _ = run_benchmark_leaderboard(routers, CFG, seeds=CFG['eval_seeds'], scenario_name="Standard (40 Nodes)")
    print(summary_std.to_string(index=False))

    print("\nEvaluating Sparse Stress Topology (15 Nodes, 30m Range)...")
    summary_sparse, _ = run_benchmark_leaderboard(
        routers, CFG, seeds=CFG['eval_seeds'], scenario_name="Sparse Stress (15 Nodes)",
        n_nodes=CFG['sparse_n_nodes'], comm_range=CFG['sparse_comm_range']
    )
    print(summary_sparse.to_string(index=False))

    # 5. Save Artifacts & Figures
    summary_std.to_csv(f"{CFG['artifacts_dir']}/leaderboard_standard.csv", index=False)
    summary_sparse.to_csv(f"{CFG['artifacts_dir']}/leaderboard_sparse.csv", index=False)
    plot_comparative_charts(summary_std, summary_sparse, f"{CFG['figures_dir']}/review2_comparative_benchmark.png")
    print(f"\nAll benchmark leaderboards and figures saved to {CFG['artifacts_dir']} and {CFG['figures_dir']}!")

if __name__ == '__main__':
    main()
