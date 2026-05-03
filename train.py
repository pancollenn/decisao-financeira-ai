import os
import numpy as np
import matplotlib.pyplot as plt

from env.market_env import MarketEnv
from agent.q_learning import QLearningAgent

# Importando as funções de plotagem (sem policy e sem V values)
from utils.plotter import (
    plot_trading_results, 
    plot_learning_with_epsilon, 
    plot_q_table_heatmap
)

from utils.data_loader import obter_dados_reais

def suavizar_curva(dados, janela=20):
    """Suaviza uma curva de dados usando média móvel."""
    pesos = np.ones(janela) / janela
    return np.convolve(dados, pesos, mode='valid')

def plotar_comparacao_sintetico_vs_real_qlearning(rewards_sint, rewards_real, output_dir, janela=20):
    """Gera 3 gráficos (um por modelo) comparando senoide vs real."""
    modelos = [
        ("simple", "Simples (1 dia)"),
        ("trend_3", "Janela (3 dias)"),
        ("ma_5_20", "Médias Móveis"),
    ]

    for model_key, model_name in modelos:
        plt.figure(figsize=(12, 6))
        plt.plot(suavizar_curva(rewards_sint[model_key], janela=janela), label=f"Senoide - {model_name}")
        plt.plot(suavizar_curva(rewards_real[model_key], janela=janela), label=f"Real - {model_name}")
        plt.title(f"Comparação de Desempenho | {model_name}")
        plt.xlabel("Episódios")
        plt.ylabel("Recompensa Suavizada")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"comparacao_sintetico_vs_real_{model_key}.png"), dpi=300)
        plt.close()

# Pastas de saída
PASTA_PLOTS_SINTETICO = "plots/ambiente_sintetico_qlearning"
PASTA_PLOTS_REAL = "plots/ambiente_real_qlearning"
PASTA_PLOTS_COMPARACOES = "plots/comparacoes_qlearning_sintetico_vs_real"

os.makedirs(PASTA_PLOTS_SINTETICO, exist_ok=True)
os.makedirs(PASTA_PLOTS_REAL, exist_ok=True)
os.makedirs(PASTA_PLOTS_COMPARACOES, exist_ok=True)

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

def treinar_tres_modelos(prices, dataset_label, output_dir, episodes):
    """Treina os 3 modelos (simple, trend-3, ma) e salva os gráficos organizados."""
    configs = [
        ("simple", "Simples (1 dia)", dict(state_type="simple")),
        ("trend_3", "Janela (3 dias)", dict(state_type="trend", window_size=3)),
        ("ma_5_20", "Médias Móveis", dict(state_type="ma", fast_period=5, slow_period=20)),
    ]

    recompensas_por_modelo = {}
    envs_por_modelo = {}
    agents_por_modelo = {}

    print(f"\n=== Dataset: {dataset_label} | episódios={episodes} | n_steps={len(prices)} ===")
    for model_key, model_name, env_kwargs in configs:
        print(f"\n--- Treinando: {model_name} ({dataset_label}) ---")
        env = MarketEnv(prices=prices, **env_kwargs)
        agent = QLearningAgent(actions=[0, 1, 2])
        
        # Coleta o histórico de recompensas e do epsilon
        rw, eps = treinar_agente(env, agent, episodes)

        recompensas_por_modelo[model_key] = rw
        envs_por_modelo[model_key] = env
        agents_por_modelo[model_key] = agent

        # Cria subpasta para este modelo específico
        model_output_dir = os.path.join(output_dir, model_key)
        os.makedirs(model_output_dir, exist_ok=True)

        # ---------------------------------------------------------
        # GERAÇÃO DOS GRÁFICOS
        # ---------------------------------------------------------
        
        # 1. Gráfico das ações de Trading sobre o preço
        plot_trading_results(env, agent, output_dir=model_output_dir, filename=f"trade_{model_key}.png")
        
        # 2. Gráfico da Curva de Aprendizado e Decaimento do Epsilon
        plot_learning_with_epsilon(rw, eps, output_dir=model_output_dir)
        
        # 3. Heatmap da Tabela Q
        plot_q_table_heatmap(agent, output_dir=model_output_dir)

    # Comparação interna dos 3 modelos para este dataset
    plt.figure(figsize=(12, 6))
    plt.plot(suavizar_curva(recompensas_por_modelo["simple"]), label="Simples (1 dia)")
    plt.plot(suavizar_curva(recompensas_por_modelo["trend_3"]), label="Janela (3 dias)")
    plt.plot(suavizar_curva(recompensas_por_modelo["ma_5_20"]), label="Médias Móveis")
    plt.title(f"Evolução do Desempenho (Lucro) | {dataset_label}")
    plt.xlabel("Episódios")
    plt.ylabel("Recompensa Suavizada")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparacao_3_modelos.png"), dpi=300)
    plt.close()

    return recompensas_por_modelo, envs_por_modelo, agents_por_modelo

if __name__ == "__main__":
    # 1) Dados sintéticos (senoide)
    t = np.linspace(0, 50, 200)
    precos_sinteticos = 100 + 20 * np.sin(t) + np.random.normal(0, 2, 200)

    # 2) Dados reais (Yahoo Finance)
    print("Carregando dados reais do mercado...")
    TICKER = "PETR4.SA"
    START_DATE = "2020-01-01"
    END_DATE = "2023-01-01"
    precos_reais = obter_dados_reais(TICKER, START_DATE, END_DATE)

    # 3) Executando o treinamento
    episodes = 1000

    rewards_sint, _, _ = treinar_tres_modelos(
        prices=precos_sinteticos,
        dataset_label="Senoide (Sintético)",
        output_dir=PASTA_PLOTS_SINTETICO,
        episodes=episodes,
    )

    rewards_real, _, _ = treinar_tres_modelos(
        prices=precos_reais,
        dataset_label=f"Real ({TICKER})",
        output_dir=PASTA_PLOTS_REAL,
        episodes=episodes,
    )

    # 4) Comparações entre datasets (sintético vs real)
    plotar_comparacao_sintetico_vs_real_qlearning(
        rewards_sint=rewards_sint,
        rewards_real=rewards_real,
        output_dir=PASTA_PLOTS_COMPARACOES,
        janela=20,
    )