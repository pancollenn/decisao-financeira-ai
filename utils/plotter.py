import matplotlib.pyplot as plt
import numpy as np
import os

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

def plot_trading_results(env, agent, output_dir="resultados"):
    """Executa um teste final e guarda o gráfico de trading na pasta especificada."""
    _garantir_pasta(output_dir)
    caminho_final = os.path.join(output_dir, "resultado_trading.png")
    
    # --- Lógica de teste (Epsilon = 0) ---
    estado = env.reset()
    done = False
    agent.epsilon = 0.0 

    historico_precos = []
    compras_x, compras_y = [], []
    vendas_x, vendas_y = [], []

    passo = 0
    while not done:
        acao = agent.choose_action(estado)
        preco_atual = env.prices[env.current_step]
        historico_precos.append(preco_atual)

        if acao == 1 and env.position == 0:
            if env.balance // preco_atual > 0:
                compras_x.append(passo)
                compras_y.append(preco_atual)
        elif acao == 2 and env.position == 1:
            vendas_x.append(passo)
            vendas_y.append(preco_atual)

        proximo_state, reward, done = env.step(acao)
        estado = proximo_state
        passo += 1

    plt.figure(figsize=(15, 7))
    plt.plot(historico_precos, label='Preço do Ativo', color='gray', alpha=0.6)
    
    if compras_x:
        plt.scatter(compras_x, compras_y, marker='^', color='green', s=100, label='Compra')
    if vendas_x:
        plt.scatter(vendas_x, vendas_y, marker='v', color='red', s=100, label='Venda')

    plt.title("Estratégia do Agente Treinado", fontsize=14)
    plt.xlabel("Tempo", fontsize=12)
    plt.ylabel("Preço ($)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plt.savefig(caminho_final, dpi=300)
    plt.close()
    print(f"Gráfico guardado em: {caminho_final}")