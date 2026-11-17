"""Carrega o modelo treinado (dqn_robo.pt) e mostra o robô jogando no grid.

Exemplos:
    python play.py                  # modelo treinado com bateria (V1)
    python play.py --no-battery     # modelo treinado sem bateria (V0)
    python play.py --delay 0.3      # mais devagar

No mapa: R = robô, r = robô carregando o item (maiúsculo/minúsculo).
O robô usa só a política aprendida (sem ações aleatórias).
"""
import argparse
import os
import time

import torch

from warehouse_env import WarehouseEnv
from dqn_agent import QNetwork

ACTION_NAMES = ["cima", "baixo", "esquerda", "direita", "interagir", "recarregar"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="dqn_robo.pt")
    p.add_argument("--no-battery", action="store_true",
                   help="use o mesmo valor do treino (V0 = sem bateria)")
    p.add_argument("--delay", type=float, default=0.15, help="segundos entre quadros")
    args = p.parse_args()

    env = WarehouseEnv(use_battery=not args.no_battery, render_mode="ansi")
    net = QNetwork(env.observation_space.shape[0], env.action_space.n)
    net.load_state_dict(torch.load(args.model, map_location="cpu"))
    net.eval()

    obs, info = env.reset()
    total, step, done, last = 0.0, 0, False, "-"
    while not done:
        os.system("cls" if os.name == "nt" else "clear")
        print(env.render())
        print(f"\npasso {step:3d} | última ação: {last} | retorno: {total:7.1f} | "
              f"entregas: {info['deliveries']} | bateria: {info['battery']}")
        time.sleep(args.delay)

        with torch.no_grad():
            q = net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        action = int(q.argmax(dim=1).item())
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        last = ACTION_NAMES[action]
        step += 1
        done = terminated or truncated

    os.system("cls" if os.name == "nt" else "clear")
    print(env.render())
    motivo = "bateria zerada" if terminated and info["deliveries"] < env.max_deliveries \
        else "meta de entregas" if terminated else "limite de passos"
    print(f"\nFim ({motivo}) em {step} passos | retorno: {total:.1f} | "
          f"entregas: {info['deliveries']}")


if __name__ == "__main__":
    main()