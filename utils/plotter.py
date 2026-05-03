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
    Suporta estados no formato:
      - (posição, sinal)                  -> modelos "simple" e "ma"
      - (posição, t1, t2, ..., tk)        -> modelo "trend" com janela k
    """
    _garantir_pasta(output_dir)
    caminho_final = os.path.join(output_dir, filename)

    # Normaliza estados para 2D: (posicao, sinal/tendencia)
    # - Se o estado tiver mais de 2 elementos, o "sinal" vira a tupla (t1, t2, ..., tk).
    valores_por_estado = {}
    for s, v in V.items():
        if not (isinstance(s, tuple) and len(s) >= 2):
            continue

        pos = s[0]
        sinal = tuple(s[1:]) if len(s) > 2 else s[1]
        valores_por_estado[(pos, sinal)] = float(v)

    if not valores_por_estado:
        print("Aviso: Nenhum estado válido encontrado para plotar o heatmap de V(s).")
        return

    # Descobre os valores únicos de posição (eixo Y) e sinal/tendência (eixo X)
    posicoes = sorted(list(set(pos for pos, _ in valores_por_estado.keys())))
    sinais = sorted(list(set(sinal for _, sinal in valores_por_estado.keys())))

    # Cria a matriz de valores
    matriz_v = np.zeros((len(posicoes), len(sinais)))
    for i, pos in enumerate(posicoes):
        for j, sinal in enumerate(sinais):
            matriz_v[i, j] = valores_por_estado.get((pos, sinal), 0.0)

    labels_posicao = [f"Pos {p} (Líquido)" if p == 0 else f"Pos {p} (Comprado)" for p in posicoes]

    # Labels do eixo X: depende do tipo do sinal
    if all(isinstance(s, (int, np.integer)) for s in sinais) and set(int(s) for s in sinais).issubset({0, 1}):
        labels_sinal = [f"Sinal {int(s)} (Baixa)" if int(s) == 0 else f"Sinal {int(s)} (Alta)" for s in sinais]
    elif all(isinstance(s, tuple) for s in sinais):
        labels_sinal = [f"Tend {s}" for s in sinais]
    else:
        labels_sinal = [str(s) for s in sinais]

    largura = max(8, 1.2 * len(sinais))
    plt.figure(figsize=(largura, 6))
    sns.heatmap(
        matriz_v,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        xticklabels=labels_sinal,
        yticklabels=labels_posicao,
    )
    
    plt.title("Mapa de Valores V(s) - Equações de Bellman", fontsize=14)
    plt.xlabel("Sinal / Tendência", fontsize=12)
    plt.ylabel("Posição no Mercado", fontsize=12)
    plt.tight_layout()
    plt.savefig(caminho_final, dpi=300)
    plt.close()
    print(f"Gráfico V(s) guardado em: {caminho_final}")

def plotar_comparacao_exploracao(recompensas_dict, output_dir="resultados"):
    """
    Recebe um dicionário com os históricos de recompensa de diferentes estratégias
    e plota as médias móveis em um único gráfico para comparação.
    """
    _garantir_pasta(output_dir)
    caminho_final = os.path.join(output_dir, "comparacao_exploracao.png")
    
    plt.figure(figsize=(12, 6))
    
    cores = {"epsilon_decay": "blue", "epsilon_fixed": "orange"}
    nomes = {
        "epsilon_decay": "Epsilon-Greedy (Decaimento)", 
        "epsilon_fixed": "Epsilon-Greedy (Fixo = 0.2)"
    }
    
    window_size = 30 
    
    for strategy_name, recompensas in recompensas_dict.items():
        if len(recompensas) >= window_size:
            media_movel = np.convolve(recompensas, np.ones(window_size)/window_size, mode='valid')
            eixo_x = range(window_size - 1, len(recompensas))
            plt.plot(eixo_x, media_movel, label=nomes[strategy_name], color=cores.get(strategy_name, "black"), linewidth=2)
            
    plt.title("Comparação de Estratégias de Exploração (Q-Learning)", fontsize=14)
    plt.xlabel("Episódios", fontsize=12)
    plt.ylabel("Lucro Médio Suavizado ($)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(caminho_final, dpi=300)
    plt.close()
    print(f"Gráfico de comparação de exploração guardado em: {caminho_final}")

def plot_learned_policy(agent, output_dir="resultados"):
    """
    Gera um mapa de calor mostrando a política final aprendida (a melhor ação para cada estado).
    """
    _garantir_pasta(output_dir)
    caminho_final = os.path.join(output_dir, "politica_aprendida.png")
    
    estados = list(agent.q_table.keys())
    # Em vez de pegar os valores Q, pegamos o índice da maior ação (0, 1 ou 2)
    melhores_acoes = np.array([np.argmax(agent.q_table[s]) for s in estados])
    
    # Reformata para uma matriz 2D (Nx1) para o heatmap do Seaborn aceitar
    matriz_politica = melhores_acoes.reshape(-1, 1)
    
    labels_estados = []
    for s in estados:
        pos = s[0]
        tendencia = tuple(s[1:]) if len(s) > 2 else s[1]
        labels_estados.append(f"Pos: {pos}, Tend.: {tendencia}")
        
    altura_figura = max(6, len(estados) * 0.4)
    plt.figure(figsize=(6, altura_figura))
    
    # Usamos um colormap discreto para as 3 ações
    cmap_discreto = mcolors.ListedColormap(['#cccccc', '#2ca02c', '#d62728'])
    
    ax = sns.heatmap(matriz_politica, annot=True, fmt="d", cmap=cmap_discreto,
                     xticklabels=["Ação Escolhida"], yticklabels=labels_estados,
                     cbar=False)
    
    # Ajusta as anotações para mostrar o nome da ação em vez do número
    mapa_nomes = {0: "Manter", 1: "Comprar", 2: "Vender"}
    for t in ax.texts:
        t.set_text(mapa_nomes[int(t.get_text())])
        t.set_fontsize(10)
        t.set_fontweight('bold')
    
    plt.title(r"Política Aprendida $\pi(s)$", fontsize=14)
    plt.ylabel("Estados (Posição, Tendência)", fontsize=12)
    plt.tight_layout()
    plt.savefig(caminho_final, dpi=300)
    plt.close()
    print(f"Gráfico da política aprendida guardado em: {caminho_final}")

def plot_learned_policy_dict(policy_dict, output_dir="resultados", titulo=r"Política Aprendida $\pi(s)$"):
    """
    Gera um mapa de calor mostrando a política ótima a partir de um dicionário.
    Ideal para usar com o retorno do Value Iteration.
    """
    _garantir_pasta(output_dir)
    caminho_final = os.path.join(output_dir, "politica_aprendida_vi.png")
    
    # Filtra apenas estados que têm o formato de tupla com 2 ou mais elementos
    estados = [s for s in policy_dict.keys() if isinstance(s, tuple) and len(s) >= 2]
    
    if not estados:
        print("Aviso: Nenhum estado válido encontrado para plotar a política.")
        return

    melhores_acoes = np.array([policy_dict[s] for s in estados])
    matriz_politica = melhores_acoes.reshape(-1, 1)
    
    labels_estados = []
    for s in estados:
        pos = s[0]
        tendencia = tuple(s[1:]) if len(s) > 2 else s[1]
        labels_estados.append(f"Pos: {pos}, Tend.: {tendencia}")
        
    altura_figura = max(6, len(estados) * 0.4)
    plt.figure(figsize=(6, altura_figura))
    
    cmap_discreto = mcolors.ListedColormap(['#cccccc', '#2ca02c', '#d62728'])
    
    ax = sns.heatmap(matriz_politica, annot=True, fmt="d", cmap=cmap_discreto,
                     xticklabels=["Ação Escolhida"], yticklabels=labels_estados,
                     cbar=False)
    
    mapa_nomes = {0: "Manter", 1: "Comprar", 2: "Vender"}
    for t in ax.texts:
        texto_atual = t.get_text()
        if texto_atual.isdigit():
            t.set_text(mapa_nomes[int(texto_atual)])
            t.set_fontsize(10)
            t.set_fontweight('bold')
    
    plt.title(titulo, fontsize=14)
    plt.ylabel("Estados (Posição, Tendência)", fontsize=12)
    plt.tight_layout()
    plt.savefig(caminho_final, dpi=300)
    plt.close()
    print(f"Gráfico da política guardado em: {caminho_final}")

