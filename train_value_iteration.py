import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from env.market_env import MarketEnv
from agent.policy_evaluation import ModelBasedEvaluator

# Garante que o diretório de plots exista
PASTA_PLOTS = "plots/ambiente_sintetico"
os.makedirs(PASTA_PLOTS, exist_ok=True)

# 1. Gerando os mesmos dados sintéticos do train.py original
t = np.linspace(0, 50, 200)
precos_sinteticos = 100 + 20 * np.sin(t) + np.random.normal(0, 2, 200)


def executar_value_iteration(env, nome_modelo, episodes_amostragem=1000):
    """
    Executa a amostragem, resolve as equações de Bellman e avalia a política.
    Retorna o lucro total do episódio de teste.
    """
    print(f"\n{'=' * 40}")
    print(f"--- Iniciando: {nome_modelo} ---")

    # 1. Construção do Modelo Empírico (T e R)
    evaluator = ModelBasedEvaluator(env, gamma=0.95)
    evaluator.build_empirical_model(episodes=episodes_amostragem)

    # 2. Resolução das Equações de Otimalidade de Bellman
    print("Executando Value Iteration...")
    _, optimal_policy = evaluator.value_iteration(theta=1e-5)

    # 3. Avaliação da Política Ótima no Ambiente
    estado = env.reset()
    done = False
    recompensa_total = 0
    acoes_tomadas = {0: 0, 1: 0, 2: 0}

    while not done:
        # Fallback para ação 0 (manter) se o estado for desconhecido
        acao = optimal_policy.get(estado, 0)
        acoes_tomadas[acao] += 1

        proximo_estado, recompensa, done = env.step(acao)
        estado = proximo_estado
        recompensa_total += recompensa

    print(f"  -> Recompensa Total (Lucro): ${recompensa_total:.2f}")
    print(f"  -> Ações: Manter: {acoes_tomadas[0]}, Comprar: {acoes_tomadas[1]}, Vender: {acoes_tomadas[2]}")

    return recompensa_total


# ==========================================
# EXECUÇÃO DOS 3 MODELOS
# ==========================================
nomes_modelos = ["Simples (1 Dia)", "Janela (3 Dias)", "Médias Móveis"]
lucros_finais = []

# --- 1. Modelo Simples ---
env_simple = MarketEnv(prices=precos_sinteticos, state_type="simple")
lucros_finais.append(executar_value_iteration(env_simple, nomes_modelos[0]))

# --- 2. Modelo Janela ---
env_window = MarketEnv(prices=precos_sinteticos, state_type="trend", window_size=3)
lucros_finais.append(executar_value_iteration(env_window, nomes_modelos[1]))

# --- 3. Modelo Médias Móveis ---
env_ma = MarketEnv(prices=precos_sinteticos, state_type="ma", fast_period=5, slow_period=20)
lucros_finais.append(executar_value_iteration(env_ma, nomes_modelos[2]))

# ==========================================
# GERAÇÃO DO GRÁFICO COMPARATIVO
# ==========================================
print("\nGerando gráfico comparativo de lucros...")

plt.figure(figsize=(10, 6))
# Define cores distintas para facilitar a visualização no relatório
cores = ['#ff9999', '#66b3ff', '#99ff99']

# Cria o gráfico de barras
barras = plt.bar(nomes_modelos, lucros_finais, color=cores, edgecolor='black')

# Adiciona os rótulos com os valores exatos em cima de cada barra
for barra in barras:
    altura = barra.get_height()
    plt.text(barra.get_x() + barra.get_width() / 2., altura + 1,
             f'${altura:.2f}',
             ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.title("Comparação de Desempenho: Value Iteration (Políticas Ótimas)", fontsize=14)
plt.ylabel("Lucro Total Acumulado ($)", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adiciona uma linha no eixo zero para destacar prejuízos, se houver
plt.axhline(0, color='black', linewidth=1)

caminho_grafico = os.path.join(PASTA_PLOTS, "comparacao_lucro_dp.png")
plt.tight_layout()
plt.savefig(caminho_grafico, dpi=300)
plt.close()

print(f"Gráfico salvo com sucesso em: {caminho_grafico}")