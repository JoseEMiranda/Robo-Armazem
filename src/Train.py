"""Treina o DQN no ambiente do robô de armazém.

Exemplos:
    python train.py --no-battery --episodes 1000    # V0: valida o DQN sem bateria
    python train.py --episodes 8000 --delivery-reward 50 --eps-decay 150000 \
        --buffer-size 200000 --lr 0.0005 --target-update 1000   # V1 (mais estável)

Para combater a recompensa esparsa, parte dos episódios de TREINO começa em
posições/bateria/estado de carga sorteados ("inícios variados"). A AVALIAÇÃO
sempre parte da estação, como no problema real.

Arquivos gerados: dqn_robo.pt (rede do último episódio), dqn_robo_best.pt
(melhor rede na avaliação), curva_treino.png e retornos_treino.npy.
Ctrl+C interrompe o treino e ainda salva tudo o que existe até ali.
"""
import argparse
import random

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from warehouse_env import WarehouseEnv
from dqn_agent import DQNAgent


def random_start(env, rng):
    """Sorteia um início para o episódio de treino (usado via reset(options=...))."""
    free = [(r, c) for r in range(env.h) for c in range(env.w) if (r, c) not in env.walls]
    return {
        "pos": rng.choice(free),
        "carrying": rng.random() < 0.5,
        "battery": rng.randint(10, env.battery_max),
    }


def run_episode(env, policy):
    """Roda um episódio com `policy(obs) -> ação`. Devolve (retorno, entregas, morreu)."""
    obs, info = env.reset()
    total, done = 0.0, False
    while not done:
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        total += reward
        done = terminated or truncated
    died = terminated and info["deliveries"] < env.max_deliveries
    return total, info["deliveries"], died


def evaluate(env, policy, episodes=20):
    results = [run_episode(env, policy) for _ in range(episodes)]
    returns, deliveries, deaths = zip(*results)
    return float(np.mean(returns)), float(np.mean(deliveries)), float(np.mean(deaths))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--no-battery", action="store_true", help="V0: sem bateria")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--explore-prob", type=float, default=0.7,
                   help="fração dos episódios de treino com início aleatório")
    p.add_argument("--delivery-reward", type=float, default=20.0,
                   help="recompensa por entrega (padrão do projeto: 20)")
    p.add_argument("--eps-decay", type=int, default=100_000,
                   help="passos até o ε chegar ao mínimo")
    p.add_argument("--lr", type=float, default=1e-3, help="taxa de aprendizado")
    p.add_argument("--buffer-size", type=int, default=50_000, help="tamanho do replay buffer")
    p.add_argument("--target-update", type=int, default=500,
                   help="passos entre cópias da rede principal para a rede alvo")
    args = p.parse_args()

    rng = random.Random(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = WarehouseEnv(use_battery=not args.no_battery, delivery_reward=args.delivery_reward)
    eval_env = WarehouseEnv(use_battery=not args.no_battery, delivery_reward=args.delivery_reward)
    obs_dim = env.observation_space.shape[0]
    agent = DQNAgent(obs_dim, env.action_space.n, lr=args.lr,
                     buffer_size=args.buffer_size, target_update=args.target_update,
                     eps_decay_steps=args.eps_decay)

    # Baseline: agente aleatório (partindo da estação)
    rand = evaluate(eval_env, lambda o: eval_env.action_space.sample(), episodes=50)
    print(f"[aleatório] retorno={rand[0]:.1f} entregas={rand[1]:.2f} mortes={rand[2]:.0%}")

    train_returns, eval_points = [], []   # eval_points: (episódio, retorno, entregas)
    best = (-1.0, -1e9)                   # (entregas, retorno) da melhor avaliação
    try:
        for ep in range(1, args.episodes + 1):
            options = random_start(env, rng) if rng.random() < args.explore_prob else None
            obs, _ = env.reset(seed=args.seed if ep == 1 else None, options=options)
            total, done = 0.0, False
            while not done:
                action = agent.act(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                agent.remember(obs, action, reward, next_obs, terminated)
                agent.learn()
                obs, total = next_obs, total + reward
                done = terminated or truncated
            train_returns.append(total)

            if ep % args.eval_every == 0:
                # Ambiente determinístico + política gulosa: poucos episódios bastam
                ret, deliv, dead = evaluate(eval_env, lambda o: agent.act(o, greedy=True), episodes=3)
                eval_points.append((ep, ret, deliv))
                marker = ""
                if (deliv, ret) > best:
                    best = (deliv, ret)
                    torch.save(agent.policy_net.state_dict(), "dqn_robo_best.pt")
                    marker = "  <- melhor até agora (salvo)"
                print(f"ep {ep:5d} | ε={agent.epsilon():.2f} | treino(média 100)="
                      f"{np.mean(train_returns[-100:]):7.1f} | avaliação: retorno={ret:7.1f} "
                      f"entregas={deliv:.2f} mortes={dead:.0%}{marker}")
    except KeyboardInterrupt:
        print("\nTreino interrompido (Ctrl+C); salvando o que existe...")

    # Salva modelo final, dados e gráficos
    torch.save(agent.policy_net.state_dict(), "dqn_robo.pt")
    np.save("retornos_treino.npy", np.array(train_returns))
    if eval_points:
        eps_, rets_, delivs_ = zip(*eval_points)
        window = 50
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        if len(train_returns) >= window:
            smooth = np.convolve(train_returns, np.ones(window) / window, mode="valid")
            ax1.plot(range(window - 1, len(train_returns)), smooth, alpha=0.5,
                     label=f"treino (média móvel {window}, inícios variados)")
        ax1.plot(eps_, rets_, marker="o", label="avaliação (início na estação)")
        ax1.axhline(rand[0], color="gray", linestyle="--", label="agente aleatório")
        ax1.set_xlabel("episódio"); ax1.set_ylabel("retorno"); ax1.legend()
        ax2.plot(eps_, delivs_, marker="o", color="tab:green")
        ax2.set_xlabel("episódio"); ax2.set_ylabel("entregas por episódio (avaliação)")
        plt.tight_layout()
        plt.savefig("curva_treino.png", dpi=150)
    print("Salvos: dqn_robo.pt (último), dqn_robo_best.pt (melhor), curva_treino.png")


if __name__ == "__main__":
    main()