import os
import numpy as np
import matplotlib.pyplot as plt

from env.market_env import MarketEnv
from agent.q_learning import QLearningAgent
from agent.policy_evaluation import ModelBasedEvaluator

def plot_comparacao_algoritmos(lucro_q, lucro_vi, output_dir="plots/comparacao_bellman_qlearning"):
    os.makedirs(output_dir, exist_ok=True)
    caminho = os.path.join(output_dir, "bellman_vs_qlearning.png")

    plt.figure(figsize=(8, 6))
    nomes = ['Q-Learning\n(Aprendizado)', 'Value Iteration\n(Planejamento)']
    valores = [lucro_q, lucro_vi]
    cores = ['#ff9999', '#66b3ff']

    barras = plt.bar(nomes, valores, color=cores, edgecolor='black', width=0.6)

    for barra in barras:
        altura = barra.get_height()
        plt.text(
            barra.get_x() + barra.get_width() / 2.,
            altura + (abs(altura) * 0.02 + 1),
            f'${altura:.2f}',
            ha='center',
            va='bottom',
            fontweight='bold',
            fontsize=12,
        )

    plt.title("Comparação de Desempenho: Bellman vs Q-Learning", fontsize=14)
    plt.ylabel("Lucro Total no Episódio de Teste ($)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.axhline(0, color='black', linewidth=1)
    plt.tight_layout()
    plt.savefig(caminho, dpi=300)
    plt.close()
    print(f"Gráfico comparativo salvo em: {caminho}")

def testar_agente(env, policy_func):
    """
    Roda um episódio de teste no ambiente usando uma função de política fornecida.
    Retorna o lucro final.
    """
    estado = env.reset()
    done = False
    recompensa_total = 0
    
    while not done:
        acao = policy_func(estado)
        proximo_estado, recompensa, done = env.step(acao)
        estado = proximo_estado
        recompensa_total += recompensa
        
    return recompensa_total

if __name__ == "__main__":
    # 1. Configurando o dataset sintético (senoide)
    t = np.linspace(0, 50, 200)
    precos_teste = 100 + 20 * np.sin(t) + np.random.normal(0, 2, 200)
    
    # Usando o estado de Janela (trend_3)
    env = MarketEnv(prices=precos_teste, state_type="trend", window_size=3)
    
    print("=== 1. Treinando Q-Learning ===")
    agent_q = QLearningAgent(actions=[0, 1, 2], strategy="epsilon_decay")
    episodes = 100
    
    for ep in range(episodes):
        estado = env.reset()
        done = False
        while not done:
            acao = agent_q.choose_action(estado)
            prox_estado, recompensa, done = env.step(acao)
            agent_q.learn(estado, acao, recompensa, prox_estado)
            estado = prox_estado
        agent_q.atualizar_epsilon()
        
    # Teste final do Q-Learning (Exploração desligada)
    agent_q.epsilon = 0.0
    lucro_qlearning = testar_agente(env, lambda s: agent_q.choose_action(s))
    print(f"Lucro Q-Learning no Teste: ${lucro_qlearning:.2f}")

    print("\n=== 2. Executando Value Iteration (Bellman) ===")
    evaluator = ModelBasedEvaluator(env, gamma=0.95)
    evaluator.build_empirical_model(episodes)
    _, optimal_policy = evaluator.value_iteration(theta=1e-5)
    
    # Teste final do Value Iteration
    # Se o estado não estiver na política ótima, a ação default é 0 (Manter)
    lucro_value_iteration = testar_agente(env, lambda s: optimal_policy.get(s, 0))
    print(f"Lucro Value Iteration no Teste: ${lucro_value_iteration:.2f}")

    print("\n=== 3. Gerando Gráfico Comparativo ===")
    plot_comparacao_algoritmos(lucro_qlearning, lucro_value_iteration)