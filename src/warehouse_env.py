"""Ambiente do robô de armazém (grid 6x6) no formato Gymnasium.

Legenda do mapa:  # parede | C coleta | E entrega | S estação de recarga | . livre
O robô começa na estação, com a bateria cheia.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

DEFAULT_LAYOUT = [
    "C..#.E",
    ".#.#..",
    ".#...#",
    "...#..",
    "##.##.",
    "S.....",
]

# Ações
UP, DOWN, LEFT, RIGHT, INTERACT, CHARGE = range(6)
MOVES = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}

# Recompensas
STEP_REWARD = -1.0       # custo de cada passo (dá "pressa" ao agente)
DELIVERY_REWARD = 20.0   # entrega concluída
INVALID_REWARD = -5.0    # interagir/recarregar no lugar errado
DEAD_REWARD = -50.0      # bateria chegou a 0


class WarehouseEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, layout=None, battery_max=45, carry_cost=2,
                 use_battery=True, max_deliveries=5, max_steps=200,
                 delivery_reward=DELIVERY_REWARD, render_mode=None):
        super().__init__()
        self.layout = layout or DEFAULT_LAYOUT
        self.h, self.w = len(self.layout), len(self.layout[0])
        self.walls = set()
        for r, row in enumerate(self.layout):
            for c, ch in enumerate(row):
                if ch == "#":
                    self.walls.add((r, c))
                elif ch == "C":
                    self.pickup = (r, c)
                elif ch == "E":
                    self.delivery = (r, c)
                elif ch == "S":
                    self.station = (r, c)

        self.battery_max = battery_max
        self.carry_cost = carry_cost      # gasto por passo quando carrega o item
        self.use_battery = use_battery    # False = versão V0, sem bateria
        self.max_deliveries = max_deliveries
        self.max_steps = max_steps
        self.delivery_reward = delivery_reward
        self.render_mode = render_mode

        # Observação: [linha, coluna, carregando, bateria], tudo normalizado em [0, 1]
        self.observation_space = spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(6)

    # ------------------------------------------------------------------
    def _obs(self):
        r, c = self.pos
        battery = self.battery / self.battery_max if self.use_battery else 1.0
        return np.array([r / (self.h - 1), c / (self.w - 1),
                         float(self.carrying), battery], dtype=np.float32)

    def _info(self):
        return {"deliveries": self.deliveries, "battery": self.battery}

    def _is_free(self, r, c):
        return 0 <= r < self.h and 0 <= c < self.w and (r, c) not in self.walls

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Início padrão: na estação, bateria cheia, sem item.
        self.pos = self.station
        self.battery = self.battery_max
        self.carrying = False
        self.deliveries = 0
        self.steps = 0
        # `options` permite forçar cenários de teste, ex.:
        # env.reset(options={"pos": (3, 2), "battery": 3, "carrying": True})
        if options:
            self.pos = tuple(options.get("pos", self.pos))
            self.battery = options.get("battery", self.battery)
            self.carrying = options.get("carrying", self.carrying)
        return self._obs(), self._info()

    def step(self, action):
        self.steps += 1
        reward = STEP_REWARD
        terminated = truncated = False
        r, c = self.pos

        if action in MOVES:
            dr, dc = MOVES[action]
            if self._is_free(r + dr, c + dc):
                self.pos = (r + dr, c + dc)
                if self.use_battery:
                    cost = self.carry_cost if self.carrying else 1
                    self.battery = max(self.battery - cost, 0)
            # bater na parede: não move e não gasta bateria (só o -1 do passo)

        elif action == INTERACT:
            if self.pos == self.pickup and not self.carrying:
                self.carrying = True
            elif self.pos == self.delivery and self.carrying:
                self.carrying = False
                self.deliveries += 1
                reward += self.delivery_reward
                if self.deliveries >= self.max_deliveries:
                    terminated = True
            else:
                reward += INVALID_REWARD

        elif action == CHARGE:
            if self.pos == self.station and self.use_battery:
                self.battery = self.battery_max
            else:
                reward += INVALID_REWARD

        # Bateria zerada: fim do episódio (terminated)
        if self.use_battery and self.battery <= 0 and not terminated:
            reward += DEAD_REWARD
            terminated = True

        # Limite de tempo: corte (truncated), não faz parte do MDP
        if not terminated and self.steps >= self.max_steps:
            truncated = True

        return self._obs(), reward, terminated, truncated, self._info()

    # ------------------------------------------------------------------
    def render(self):
        rows = [list(row) for row in self.layout]
        r, c = self.pos
        rows[r][c] = "r" if self.carrying else "R"   # r minúsculo = carregando item
        text = "\n".join("".join(row) for row in rows)
        if self.render_mode == "ansi":
            return text
        print(text)