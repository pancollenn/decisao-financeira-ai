import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import matplotlib.colors as mcolors

def _garantir_pasta(pasta):
    """Cria a pasta se ela ainda não existir."""
    if not os.path.exists(pasta):
        os.makedirs(pasta)
        print(f"Pasta '{pasta}' criada com sucesso.")

def plot_learning_curve(historico_recompensas, output_dir="resultados"):
    """Gera o gráfico da evolução do lucro e guarda na pasta especificada."""
    _garantir_pasta(output_dir)
    caminho_final = os.path.join(output_dir, "curva_aprendizado.png")
    
    plt.figure(figsize=(12, 6))
    plt.plot(historico_recompensas, label='Lucro por Episódio', color='royalblue', alpha=0.4)

    window_size = 50
    if len(historico_recompensas) >= window_size:
        media_movel = np.convolve(historico_recompensas, np.ones(window_size)/window_size, mode='valid')
        eixo_x_mm = range(window_size - 1, len(historico_recompensas))
        plt.plot(eixo_x_mm, media_movel, label=f'Tendência (Média Móvel {window_size} ep)', color='red', linewidth=2)

    plt.title("Curva de Aprendizado: Q-Learning no Mercado Financeiro", fontsize=14)
    plt.xlabel("Episódios de Treinamento", fontsize=12)
    plt.ylabel("Lucro Total Acumulado ($)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(caminho_final, dpi=300)
    plt.close()
    print(f"Gráfico guardado em: {caminho_final}")

def plot_trading_results(env, agent, output_dir="resultados", filename="resultado_trading.png"):
    """
    Simula um episódio final para avaliar a política aprendida e plota as 
    ações de Compra (Triângulo Verde) e Venda (Triângulo Vermelho) sobre o preço.
    """
    _garantir_pasta(output_dir)
    caminho_final = os.path.join(output_dir, filename)
    
    # Roda uma simulação limpa para registrar as ações
    estado = env.reset()
    done = False
    
    historico_precos = []
    historico_compras = []
    historico_vendas = []
    
    # Desliga a exploração (epsilon = 0) para ver o que o agente realmente aprendeu
    epsilon_original = agent.epsilon
    agent.epsilon = 0.0 
    
    while not done:
        historico_precos.append(env.prices[env.current_step])
        
        acao = agent.choose_action(estado)
        
        if acao == 1 and env.position == 0: # Comprou
            historico_compras.append((env.current_step, env.prices[env.current_step]))
        elif acao == 2 and env.position == 1: # Vendeu
            historico_vendas.append((env.current_step, env.prices[env.current_step]))
            
        estado, _, done = env.step(acao)
        
    # Restaura o epsilon do agente (boa prática)
    agent.epsilon = epsilon_original
        
    # --- Criação do Gráfico ---
    plt.figure(figsize=(12, 6))
    plt.plot(historico_precos, label="Preço do Ativo", color="black", alpha=0.7, linewidth=1.5)
    
    # Plotando os marcadores
    if historico_compras:
        x_compra, y_compra = zip(*historico_compras)
        plt.scatter(x_compra, y_compra, marker="^", color="green", s=100, label="Compra", zorder=5)
        
    if historico_vendas:
        x_venda, y_venda = zip(*historico_vendas)
        plt.scatter(x_venda, y_venda, marker="v", color="red", s=100, label="Venda", zorder=5)

    # Pegamos as informações do env para formatar o título (para sabermos qual modelo estamos vendo)
    plt.title(f"Ações do Agente no Episódio de Teste | Patrimônio Final: ${env.portfolio_value:.2f}")
    plt.xlabel("Tempo (Dias)")
    plt.ylabel("Preço")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    plt.savefig(caminho_final, dpi=300)
    plt.close()
    
def plot_q_table_heatmap(agent, output_dir="resultados"):
    """Gera um mapa de calor visualizando os valores aprendidos na Tabela Q."""
    _garantir_pasta(output_dir)
    caminho_final = os.path.join(output_dir, "heatmap_q_table.png")
    
    # Extrai os estados e a matriz de valores Q
    estados = list(agent.q_table.keys())
    q_values = np.array(list(agent.q_table.values()))
    
    # Nova formatação dinâmica que aceita tuplas de qualquer tamanho
    labels_estados = []
    for s in estados:
        pos = s[0]
        # Se o estado tiver mais de 2 elementos, pega tudo do índice 1 em diante como uma tupla
        tendencia = tuple(s[1:]) if len(s) > 2 else s[1]
        labels_estados.append(f"Pos: {pos}, Tend.: {tendencia}")
    
    labels_acoes = ["Manter", "Comprar", "Vender"]
    
    # Ajusta o tamanho da figura dinamicamente dependendo da quantidade de estados
    altura_figura = max(6, len(estados) * 0.4)
    plt.figure(figsize=(8, altura_figura))
    
    sns.heatmap(q_values, annot=True, fmt=".2f", cmap="coolwarm", 
                xticklabels=labels_acoes, yticklabels=labels_estados)
    
    plt.title("Valores Aprendidos na Tabela Q", fontsize=14)
    plt.xlabel("Ações", fontsize=12)
    plt.ylabel("Estados (Posição, Tendência)", fontsize=12)
    plt.tight_layout()
    plt.savefig(caminho_final, dpi=300)
    plt.close()
    print(f"Gráfico guardado em: {caminho_final}")

def plot_learning_with_epsilon(historico_recompensas, historico_epsilon, output_dir="resultados"):
    """Plota a recompensa e o decaimento de epsilon no mesmo gráfico."""
    _garantir_pasta(output_dir)
    caminho_final = os.path.join(output_dir, "aprendizado_vs_epsilon.png")
    
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Eixo Y primário (Lucro)
    ax1.set_xlabel("Episódios", fontsize=12)
    ax1.set_ylabel("Lucro Total", color="tab:blue", fontsize=12)
    media_movel = np.convolve(historico_recompensas, np.ones(50)/50, mode='valid')
    ax1.plot(range(49, len(historico_recompensas)), media_movel, color="tab:blue", linewidth=2, label="Média de Lucro")
    ax1.tick_params(axis='y', labelcolor="tab:blue")

    # Eixo Y secundário (Epsilon)
    ax2 = ax1.twinx()
    ax2.set_ylabel(r"Taxa de Exploração ($\epsilon$)", color="tab:red", fontsize=12)
    ax2.plot(historico_epsilon, color="tab:red", linestyle='--', linewidth=2, label="Epsilon")
    ax2.tick_params(axis='y', labelcolor="tab:red")

    plt.title("Evolução do Lucro vs. Decaimento da Exploração", fontsize=14)
    fig.tight_layout()
    plt.savefig(caminho_final, dpi=300)
    plt.close()
    print(f"Gráfico guardado em: {caminho_final}")

def plotar_comparacao_sintetico_vs_real_value_iteration(lucros_sint, lucros_real, output_dir):
    """Gera 3 gráficos (um por modelo) comparando senoide vs real."""
    modelos = [
        ("simple", "Simples (1 dia)"),
        ("trend_3", "Janela (3 dias)"),
        ("ma_5_20", "Médias Móveis"),
    ]

    for model_key, model_name in modelos:
        plt.figure(figsize=(8, 5))
        valores = [lucros_sint[model_key], lucros_real[model_key]]
        barras = plt.bar(["Senoide", "Real"], valores, color=['#66b3ff', '#ff9999'], edgecolor='black')

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

        plt.title(f"Comparação Senoide vs Real | {model_name}", fontsize=13)
        plt.ylabel("Lucro Total Acumulado ($)", fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.axhline(0, color='black', linewidth=1)

        caminho = os.path.join(output_dir, f"comparacao_sintetico_vs_real_{model_key}.png")
        plt.tight_layout()
        plt.savefig(caminho, dpi=300)
        plt.close()

def plot_v_values_heatmap(V, output_dir="resultados", filename="heatmap_v_values.png"):
    """
    Gera um mapa de calor visualizando os Valores V(s) aprendidos pelo Value Iteration.
    Assume que os estados são tuplas de 2 dimensões: (posição, sinal/tendência).
    """
    _garantir_pasta(output_dir)
    caminho_final = os.path.join(output_dir, filename)
    
    # Filtra apenas os estados que têm 2 elementos (posição, sinal)
    estados_2d = {s: v for s, v in V.items() if isinstance(s, tuple) and len(s) == 2}
    
    if not estados_2d:
        print("Aviso: Nenhum estado 2D encontrado para plotar o heatmap de V(s).")
        return

    # Descobre os valores únicos de posição (eixo Y) e sinal (eixo X)
    posicoes = sorted(list(set(s[0] for s in estados_2d.keys())))
    sinais = sorted(list(set(s[1] for s in estados_2d.keys())))
    
    # Cria a matriz de valores
    matriz_v = np.zeros((len(posicoes), len(sinais)))
    for i, pos in enumerate(posicoes):
        for j, sinal in enumerate(sinais):
            matriz_v[i, j] = estados_2d.get((pos, sinal), 0.0)

    # Nomes para os eixos (adaptável caso usem tendência ou média móvel)
    labels_posicao = [f"Pos {p} (Líquido)" if p == 0 else f"Pos {p} (Comprado)" for p in posicoes]
    labels_sinal = [f"Sinal {s} (Baixa)" if s == 0 else f"Sinal {s} (Alta)" for s in sinais]

    plt.figure(figsize=(8, 6))
    sns.heatmap(matriz_v, annot=True, fmt=".2f", cmap="viridis", 
                xticklabels=labels_sinal, yticklabels=labels_posicao)
    
    plt.title("Mapa de Valores V(s) - Equações de Bellman", fontsize=14)
    plt.xlabel("Sinal / Tendência", fontsize=12)
    plt.ylabel("Posição no Mercado", fontsize=12)
    plt.tight_layout()
    plt.savefig(caminho_final, dpi=300)
    plt.close()
    print(f"Gráfico V(s) guardado em: {caminho_final}")

