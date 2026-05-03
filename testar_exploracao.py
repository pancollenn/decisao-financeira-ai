from env.market_env import MarketEnv
from agent.q_learning import QLearningAgent
from utils.plotter import plotar_comparacao_exploracao
import numpy as np

# Cria uma senoide simples para testar
t = np.linspace(0, 50, 200)
precos_teste = 100 + 20 * np.sin(t)

episodes = 500
resultados_exploracao = {}

# Configurações das estratégias: APENAS DECAIMENTO VS FIXO
estrategias = [
    {"name": "epsilon_decay", "kwargs": {"strategy": "epsilon_decay", "epsilon": 1.0, "epsilon_decay": 0.98}},
    {"name": "epsilon_fixed", "kwargs": {"strategy": "epsilon_fixed", "epsilon": 0.2}} # 20% de exploração fixa
]

print("Iniciando comparação de estratégias de exploração (Decaimento vs Fixo)...")

for config in estrategias:
    nome = config["name"]
    print(f"Treinando estratégia: {nome}")
    
    env = MarketEnv(prices=precos_teste, state_type="trend", window_size=3)
    agent = QLearningAgent(actions=[0, 1, 2], **config["kwargs"])
    
    historico = []
    for ep in range(episodes):
        estado = env.reset()
        done = False
        lucro_ep = 0
        
        while not done:
            acao = agent.choose_action(estado)
            prox_estado, recompensa, done = env.step(acao)
            agent.learn(estado, acao, recompensa, prox_estado)
            estado = prox_estado
            lucro_ep += recompensa
            
        agent.atualizar_epsilon()
        historico.append(lucro_ep)
            
    resultados_exploracao[nome] = historico

# Gera o gráfico final
plotar_comparacao_exploracao(resultados_exploracao, output_dir="plots/exploracao")