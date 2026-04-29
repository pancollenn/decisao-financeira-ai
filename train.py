import os

import numpy as np
import matplotlib.pyplot as plt
from env.market_env import MarketEnv
from agent.q_learning import QLearningAgent
from utils.plotter import plot_learning_curve, plot_learning_with_epsilon, plot_q_table_heatmap, plot_trading_results

def suavizar_curva(dados, janela=20):
    """Suaviza uma curva de dados usando média móvel."""
    pesos = np.ones(janela) / janela
    return np.convolve(dados, pesos, mode='valid')

# Cria o diretório de plots caso não exista
PASTA_PLOTS = "plots/ambiente_sintetico"
os.makedirs(PASTA_PLOTS, exist_ok=True)

# 1. Gerando dados sintéticos (uma senoide para testar o aprendizado)
t = np.linspace(0, 50, 200)
precos_sinteticos = 100 + 20 * np.sin(t) + np.random.normal(0, 2, 200)

# 2. Função encapsulada de treinamento
def treinar_agente(env, agent, episodes=1000):
    historico_recompensas = []
    historico_epsilon = []
    
    for ep in range(episodes):
        estado = env.reset()
        done = False
        recompensa_total_episodio = 0
        
        while not done:
            # Agente observa o estado e escolhe uma ação
            acao = agent.choose_action(estado)
            
            # O ambiente processa a ação e retorna as consequências
            proximo_estado, recompensa, done = env.step(acao)
            
            # Agente aprende com a consequência
            agent.learn(estado, acao, recompensa, proximo_estado)
            
            # Atualiza o estado atual
            estado = proximo_estado
            recompensa_total_episodio += recompensa
            
        # Ao final do episódio, diminui a aleatoriedade (Epsilon)   
        agent.atualizar_epsilon()
        historico_recompensas.append(recompensa_total_episodio)
        historico_epsilon.append(agent.epsilon)
        
        # Print a cada 200 episódios para não poluir o terminal
        if (ep + 1) % 200 == 0:
            print(f"  Episódio: {ep+1} | Recompensa Total: {recompensa_total_episodio:.2f} | Epsilon: {agent.epsilon:.3f}")
            
    print(f"  -> Tamanho final da Tabela Q: {len(agent.q_table)} estados mapeados.")
    return historico_recompensas, historico_epsilon

# 3. Executando o treinamento
episodes = 1000

# --- 1. Modelo Simples (Apenas 1 dia) ---
print("\n--- Treinando: Modelo Simples ---")
env_simple = MarketEnv(prices=precos_sinteticos, state_type="simple")
agent_simple = QLearningAgent(actions=[0, 1, 2])
rw_simple, _ = treinar_agente(env_simple, agent_simple, episodes)

# --- 2. Modelo Janela (3 dias) ---
print("\n--- Treinando: Janela 3 Dias ---")
env_window = MarketEnv(prices=precos_sinteticos, state_type="trend", window_size=3)
agent_window = QLearningAgent(actions=[0, 1, 2])
rw_window, _ = treinar_agente(env_window, agent_window, episodes)

# --- 3. Modelo Médias Móveis ---
print("\n--- Treinando: Médias Móveis ---")
env_ma = MarketEnv(prices=precos_sinteticos, state_type="ma", fast_period=5, slow_period=20)
agent_ma = QLearningAgent(actions=[0, 1, 2])
rw_ma, _ = treinar_agente(env_ma, agent_ma, episodes)

# --- NOVA VISUALIZAÇÃO: Comparação de Lucro Acumulado ---
plt.figure(figsize=(12, 6))
plt.plot(suavizar_curva(rw_simple), label="Simples (1 dia)")
plt.plot(suavizar_curva(rw_window), label="Janela (3 dias)")
plt.plot(suavizar_curva(rw_ma), label="Médias Móveis")
plt.title("Evolução do Desempenho (Lucro + Penalidades)")
plt.xlabel("Episódios")
plt.ylabel("Recompensa Suavizada")
plt.legend()
plt.savefig(f"{PASTA_PLOTS}/comparacao_3_modelos.png")

# --- NOVA VISUALIZAÇÃO: Resultado de Trading (Subplots) ---
fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
# Aqui chamamos a lógica de plot_trading_results adaptada para subplots
# Ou chamamos separadamente:
plot_trading_results(env_simple, agent_simple, output_dir=PASTA_PLOTS, filename="trade_simples.png")
plot_trading_results(env_window, agent_window, output_dir=PASTA_PLOTS, filename="trade_janela.png")
plot_trading_results(env_ma, agent_ma, output_dir=PASTA_PLOTS, filename="trade_ma.png")