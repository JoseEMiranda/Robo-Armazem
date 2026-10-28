# Robô de Armazém com Aprendizado por Reforço

Projeto da disciplina **[Nome da matéria]**, em que um agente aprende por reforço a operar um robô de armazém em um ambiente customizado, construído sobre o [Gymnasium](https://gymnasium.farama.org/).

> **Status:** em desenvolvimento.

## Ideia

Um robô se move em um grid com obstáculos. Ele precisa **pegar itens em um ponto de coleta e levá-los até um ponto de entrega**, sem ficar sem bateria. Para isso, ele tem que decidir quando vale a pena ir até a estação de recarga: recarregar cedo demais desperdiça tempo, tarde demais zera a bateria.

## Modelagem do MDP

### Ambiente

Grid 6x6 com:

- paredes (obstáculos fixos)
- um ponto de coleta
- um ponto de entrega
- uma estação de recarga

### Estado

O estado precisa ser markoviano: dado o estado atual, o histórico não traz informação extra.

| Variável | Valores | Por que está no estado |
|---|---|---|
| Posição (x, y) | células do grid | onde o robô está |
| Carregando item | 0 ou 1 | muda o que é uma boa ação na mesma célula |
| Bateria | 0 a 20 | define se ele alcança o destino ou precisa recarregar |

São cerca de 1.500 estados (36 × 2 × 21), o que permite Q-learning tabular. As posições de coleta, entrega e recarga são fixas e por isso não entram no estado.

### Ações

`Discrete(6)`:

| Ação | Efeito |
|---|---|
| Cima, baixo, esquerda, direita | move uma célula (bater na parede não move) |
| Interagir | pega o item na coleta ou solta na entrega |
| Recarregar | repõe a bateria, apenas na estação |

### Recompensas

| Evento | Valor |
|---|---|
| Cada passo | −1 |
| Entrega concluída | +20 |
| Ação inválida | −5 |
| Bateria chega a 0 | −50 e fim do episódio |

Não há recompensa direta por recarregar, para evitar que o agente fique carregando sem entregar nada. Recarregar já é incentivado indiretamente, porque evita a penalidade de −50.

### Fim de episódio

- `terminated`: a bateria zerou ou o robô concluiu 5 entregas.
- `truncated`: o limite de 200 passos foi atingido.

## Etapas do desenvolvimento

1. **V0:** grid, coleta e entrega, sem bateria (validação do ambiente e do agente).
2. **V1:** adiciona bateria e estação de recarga.
3. **V2 (opcional):** item em posição aleatória ou obstáculos móveis.

## Instalação

```bash
git clone [URL do repositório]
cd [nome-da-pasta]
pip install gymnasium numpy matplotlib
```

## Como usar

Estrutura prevista (ajustar conforme a implementação):

```
.
├── README.md
├── [arquivo do ambiente].py     # classe do ambiente (gym.Env)
├── [arquivo do agente].py       # Q-learning
├── [arquivo de treino].py       # loop de treinamento e gráficos
└── resultados/                  # curvas de recompensa e figuras
```

Interface padrão do Gymnasium:

```python
import gymnasium as gym

env = gym.make("[id-do-ambiente]")
obs, info = env.reset()

episode_over = False
while not episode_over:
    action = env.action_space.sample()  # trocar pela política do agente
    obs, reward, terminated, truncated, info = env.step(action)
    episode_over = terminated or truncated

env.close()
```

## Experimentos planejados

- Agente aleatório × Q-learning (curva de recompensa por episódio).
- Efeito dos hiperparâmetros (ε, α, γ).
- Comparação entre diferentes desenhos de recompensa (por exemplo, com e sem penalidade por ação inválida).
- Opcional: DQN na versão V2, para discutir o limite da abordagem tabular.

## Resultados

_A preencher após os experimentos._

## Integrantes

- [Nome 1]
- [Nome 2]
- [Nome 3]

## Licença

[Definir licença, por exemplo MIT]
