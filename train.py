import os

import numpy as np
import matplotlib.pyplot as plt
from env.market_env import MarketEnv
from agent.q_learning import QLearningAgent
from utils.plotter import plot_learning_curve, plot_learning_with_epsilon, plot_q_table_heatmap, plot_trading_results

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

# --- Modelo 1: Janela de Tendências ---
print("\n--- Iniciando Treinamento: Janela de Tendências (Window=3) ---")
env_trend = MarketEnv(prices=precos_sinteticos, state_type="trend", window_size=3)
agent_trend = QLearningAgent(actions=[0, 1, 2])
rw_trend, eps_trend = treinar_agente(env_trend, agent_trend, episodes)

# --- Modelo 2: Médias Móveis ---
print("\n--- Iniciando Treinamento: Médias Móveis (Fast=5, Slow=20) ---")
env_ma = MarketEnv(prices=precos_sinteticos, state_type="ma", fast_period=5, slow_period=20)
agent_ma = QLearningAgent(actions=[0, 1, 2])
rw_ma, eps_ma = treinar_agente(env_ma, agent_ma, episodes)

# 4. Plotando os resultados

print("\nGerando gráficos de comparação...")

# Gráfico Extra: Comparando a Convergência dos dois métodos juntos
plt.figure(figsize=(10, 5))

# Criamos uma função simples para calcular a média móvel das recompensas e suavizar o gráfico de aprendizado
def suavizar_curva(dados, janela=20):
    pesos = np.ones(janela) / janela
    return np.convolve(dados, pesos, mode='valid')

plt.plot(suavizar_curva(rw_trend), label="Estado: Tendências (Janela 3)", color='blue', alpha=0.8)
plt.plot(suavizar_curva(rw_ma), label="Estado: Médias Móveis (Cruzamento)", color='orange', alpha=0.8)

plt.title("Comparação de Aprendizado: Tendência vs. Médias Móveis")
plt.xlabel("Episódios")
plt.ylabel("Recompensa Total (Suavizada)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(f"{PASTA_PLOTS}/comparacao_convergencia.png")
plt.show()

# Gerando os plots individuais que você já tinha, mas salvando com prefixos diferentes
print("\nSalvando plots do Modelo Trend...")
# Aqui você pode precisar ajustar o plotter.py se ele não aceitar prefixos de nome de arquivo, 
# mas em teoria, a chamada continuaria a mesma:
plot_trading_results(env_trend, agent_trend, output_dir=PASTA_PLOTS)

print("Salvando plots do Modelo MA...")
plot_trading_results(env_ma, agent_ma, output_dir=PASTA_PLOTS)

print("\nProcesso concluído com sucesso!")



# # 2. Instanciando Ambiente e Agente
# env = MarketEnv(prices=precos_sinteticos)
# agent = QLearningAgent(actions=[0, 1, 2])



# print("\nTreinamento Concluído!")
# print(f"Tamanho final da Tabela Q: {len(agent.q_table)} estados mapeados.")

# # Chamada das funções especificando a pasta de saída
# PASTA_PLOTS = "plots/ambiente_sintetico"

# plot_learning_curve(historico_recompensas, output_dir=PASTA_PLOTS)
# plot_trading_results(env, agent, output_dir=PASTA_PLOTS)
# plot_q_table_heatmap(agent, output_dir=PASTA_PLOTS)
# plot_learning_with_epsilon(historico_recompensas, historico_epsilon, output_dir=PASTA_PLOTS)