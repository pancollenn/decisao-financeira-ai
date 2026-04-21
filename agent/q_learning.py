import numpy as np
import random

class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        actions: 0 (Manter), 1 (Comprar) ou 2 (Vender)
        """
        self.actions = actions
        self.alpha = alpha                      # Taxa de aprendizado
        self.gamma = gamma                      # Fator de desconto
        self.epsilon = epsilon                  # Taxa de exploração
        self.epsilon_decay = epsilon_decay      # Decaimento da taxa de exploração
        self.epsilon_min = epsilon_min          # Taxa de exploração mínima

        # Tabela Q: dicionário em que a chave é a tupla do estado (posicao, tendencia)
        # e o valor é um array do NumPy contendo os Valores Q para cada ação.
        self.q_table = {}

    def _get_q_values(self, state):
        """
        Busca os valores Q para um estado. Se o estado for novo, inicializa com zeros.
        """
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        return self.q_table[state]
    
    def choose_action(self, state):
        """
        Implementa a estratégia Epsilon-Greedy.
        """
        if random.uniform(0, 1) < self.epsilon:
            # Exploração: escolhe uma ação aleatória
            return random.choice(self.actions)

        else:
            # Explotação: escolhe a ação com maior valor Q conhecido
            q_values = self._get_q_values(state)
            return np.argmax(q_values)
        
    def learn(self, state, action, reward, next_state):
        """
        Aplica a equação de atualização de Bellman
        """
        q_values = self._get_q_values(state)
        next_q_values = self._get_q_values(next_state)

        best_next_q = np.max(next_q_values)

        # Regra de atualização do Q-Learning
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - q_values[action]

        q_values[action] += self.alpha * td_error

        # Atualiza a tabela (redundante devido à mutabilidade do numpy, mas explícito)
        self.q_table[state] = q_values


    def atualizar_epsilon(self):
        """
        Reduz a exploração ao longo do tempo (chamado ao final de cada episódio).
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
