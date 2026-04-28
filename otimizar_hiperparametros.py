import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from env.market_env import MarketEnv
from agent.q_learning import QLearningAgent

def _garantir_pasta(pasta):
    if not os.path.exists(pasta):
        os.makedirs(pasta)

# ==========================================
# 1. CONFIGURAÇÃO DO GRID SEARCH
# ==========================================
# Vetores de hiperparâmetros que queremos testar
alphas = [0.01, 0.05, 0.1, 0.2, 0.5]
gammas = [0.1, 0.5, 0.8, 0.95, 0.99]

# Matriz vazia para guardar o lucro médio de cada combinação
resultados = np.zeros((len(alphas), len(gammas)))

# Criamos UMA ÚNICA curva de preços para ser justo com todos os agentes
t = np.linspace(0, 50, 200)
precos_fixos = 100 + 20 * np.sin(t) + np.random.normal(0, 2, 200)

episodios = 400
janela_avaliacao = 50 # Vamos avaliar a média de lucro dos últimos 50 episódios

print("Iniciando Grid Search. Treinando múltiplos agentes...")

# ==========================================
# 2. LOOP DE OTIMIZAÇÃO
# ==========================================
for i, alpha in enumerate(alphas):
    for j, gamma in enumerate(gammas):
        # Instancia um ambiente e um novo agente com a configuração da vez
        env = MarketEnv(prices=precos_fixos)
        agent = QLearningAgent(actions=[0, 1, 2], alpha=alpha, gamma=gamma)
        
        historico_recompensas = []
        
        # Treinamento rápido deste agente específico
        for ep in range(episodios):
            estado = env.reset()
            done = False
            recompensa_total_episodio = 0
            
            while not done:
                acao = agent.choose_action(estado)
                proximo_estado, recompensa, done = env.step(acao)
                agent.learn(estado, acao, recompensa, proximo_estado)
                estado = proximo_estado
                recompensa_total_episodio += recompensa
                
            agent.atualizar_epsilon()
            historico_recompensas.append(recompensa_total_episodio)
        
        # Calcula o desempenho do agente quando ele já estava maduro (final do treino)
        lucro_final_medio = np.mean(historico_recompensas[-janela_avaliacao:])
        resultados[i, j] = lucro_final_medio
        
        print(f"Alpha: {alpha:.2f} | Gamma: {gamma:.2f} --> Lucro Médio: ${lucro_final_medio:.2f}")

# ==========================================
# 3. GERAÇÃO DO MAPA DE CALOR (HEATMAP)
# ==========================================
PASTA_RESULTADOS = "grid_search_results"
_garantir_pasta(PASTA_RESULTADOS)
caminho_final = os.path.join(PASTA_RESULTADOS, "grid_search_hiperparametros.png")

plt.figure(figsize=(10, 8))

# O seaborn.heatmap facilita muito a visualização de matrizes 2D
sns.heatmap(resultados, annot=True, fmt=".0f", cmap="viridis",
            xticklabels=gammas, yticklabels=alphas)

plt.title("Grid Search: Lucro Médio Final por Hiperparâmetro", fontsize=14)
plt.xlabel("Fator de Desconto ($\gamma$)", fontsize=12)
plt.ylabel("Taxa de Aprendizado ($\\alpha$)", fontsize=12)
plt.tight_layout()

plt.savefig(caminho_final, dpi=300)
plt.close()

print(f"\nBusca concluída! O Heatmap de otimização foi salvo em: {caminho_final}")