import numpy as np
import matplotlib.pyplot as plt
from env.market_env import MarketEnv
from agent.q_learning import QLearningAgent
from utils.plotter import plot_learning_curve, plot_learning_with_epsilon, plot_q_table_heatmap, plot_trading_results

# 1. Gerando dados sintéticos (uma senoide para testar o aprendizado)
t = np.linspace(0, 50, 200)
precos_sinteticos = 100 + 20 * np.sin(t) + np.random.normal(0, 2, 200)

# 2. Instanciando Ambiente e Agente
env = MarketEnv(prices=precos_sinteticos)
agent = QLearningAgent(actions=[0, 1, 2])

episodes = 1000
historico_recompensas = []
historico_epsilon = []

# 3. O Loop Principal de Treinamento
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
    
    # Print de acompanhamento a cada 50 episódios
    if (ep + 1) % 50 == 0:
        print(f"Episódio: {ep+1} | Recompensa Total: {recompensa_total_episodio:.2f} | Epsilon: {agent.epsilon:.3f}")

print("\nTreinamento Concluído!")
print(f"Tamanho final da Tabela Q: {len(agent.q_table)} estados mapeados.")

# Chamada das funções especificando a pasta de saída
PASTA_PLOTS = "plots"

plot_learning_curve(historico_recompensas, output_dir=PASTA_PLOTS)
plot_trading_results(env, agent, output_dir=PASTA_PLOTS)
plot_q_table_heatmap(agent, output_dir=PASTA_PLOTS)
plot_learning_with_epsilon(historico_recompensas, historico_epsilon, output_dir=PASTA_PLOTS)