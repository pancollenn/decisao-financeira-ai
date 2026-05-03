import os
import numpy as np
import matplotlib.pyplot as plt

from env.market_env import MarketEnv
from agent.policy_evaluation import ModelBasedEvaluator

from utils.data_loader import obter_dados_reais
from utils.plotter import plotar_comparacao_sintetico_vs_real_value_iteration, plot_v_values_heatmap, plot_learned_policy_dict

# Pastas de saída
PASTA_PLOTS_SINTETICO = "plots/ambiente_sintetico_value_iteration"
PASTA_PLOTS_REAL = "plots/ambiente_real_value_iteration"
PASTA_PLOTS_COMPARACOES = "plots/comparacoes_value_iteration_sintetico_vs_real"
os.makedirs(PASTA_PLOTS_SINTETICO, exist_ok=True)
os.makedirs(PASTA_PLOTS_REAL, exist_ok=True)
os.makedirs(PASTA_PLOTS_COMPARACOES, exist_ok=True)


def executar_value_iteration(env, nome_modelo, episodes_amostragem=1000, output_dir="resultados", model_key="default"):
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
    V, optimal_policy = evaluator.value_iteration(theta=1e-5)

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

    # Geração do Heatmap V(s)
    model_output_dir = os.path.join(output_dir, model_key)
    plot_v_values_heatmap(V, output_dir=model_output_dir, filename=f"heatmap_v_values_{model_key}.png")

    # Geração do Heatmap da Política Ótima
    plot_learned_policy_dict(optimal_policy, output_dir=model_output_dir)

    return recompensa_total


def executar_tres_modelos(prices, dataset_label, output_dir, episodes_amostragem=1000):
    """Executa Value Iteration para 3 modelos (simple, trend_3, ma_5_20) em um dataset."""
    configs = [
        ("simple", "Simples (1 dia)", dict(state_type="simple")),
        ("trend_3", "Janela (3 dias)", dict(state_type="trend", window_size=3)),
        ("ma_5_20", "Médias Móveis", dict(state_type="ma", fast_period=5, slow_period=20)),
    ]

    lucros_por_modelo = {}
    print(f"\n=== Dataset: {dataset_label} | amostragem={episodes_amostragem} | n_steps={len(prices)} ===")

    for model_key, model_name, env_kwargs in configs:
        env = MarketEnv(prices=prices, **env_kwargs)
        
        lucro = executar_value_iteration(
            env, 
            f"{model_name} ({dataset_label})", 
            episodes_amostragem=episodes_amostragem,
            output_dir=output_dir,
            model_key=model_key
        )
        lucros_por_modelo[model_key] = lucro

    # Gráfico interno dos 3 modelos para este dataset
    nomes = ["Simples (1 dia)", "Janela (3 dias)", "Médias Móveis"]
    valores = [lucros_por_modelo["simple"], lucros_por_modelo["trend_3"], lucros_por_modelo["ma_5_20"]]
    cores = ['#ff9999', '#66b3ff', '#99ff99']

    plt.figure(figsize=(10, 6))
    barras = plt.bar(nomes, valores, color=cores, edgecolor='black')

    for barra in barras:
        altura = barra.get_height()
        plt.text(
            barra.get_x() + barra.get_width() / 2.,
            altura + (abs(altura) * 0.02 + 1),
            f'${altura:.2f}',
            ha='center',
            va='bottom',
            fontweight='bold',
            fontsize=11,
        )

    plt.title(f"Comparação de Desempenho: Value Iteration | {dataset_label}", fontsize=14)
    plt.ylabel("Lucro Total Acumulado ($)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.axhline(0, color='black', linewidth=1)

    caminho_grafico = os.path.join(output_dir, "comparacao_3_modelos.png")
    plt.tight_layout()
    plt.savefig(caminho_grafico, dpi=300)
    plt.close()

    print(f"Gráfico salvo em: {caminho_grafico}")
    return lucros_por_modelo

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

    # 3) Execução
    episodes_amostragem = 1000
    lucros_sint = executar_tres_modelos(
        prices=precos_sinteticos,
        dataset_label="Senoide (Sintético)",
        output_dir=PASTA_PLOTS_SINTETICO,
        episodes_amostragem=episodes_amostragem,
    )

    lucros_real = executar_tres_modelos(
        prices=precos_reais,
        dataset_label=f"Real ({TICKER})",
        output_dir=PASTA_PLOTS_REAL,
        episodes_amostragem=episodes_amostragem,
    )

    # 4) Comparação entre datasets
    plotar_comparacao_sintetico_vs_real_value_iteration(
        lucros_sint=lucros_sint,
        lucros_real=lucros_real,
        output_dir=PASTA_PLOTS_COMPARACOES,
    )