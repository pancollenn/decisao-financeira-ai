import numpy as np
import random

class QLearningAgent:
    def __init__(self, actions, alpha=0.01, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.005, strategy="epsilon_decay"):
        """
        actions: 0 (Manter), 1 (Comprar) ou 2 (Vender)
        strategy: "epsilon_decay" ou "epsilon_fixed"
        """
        self.actions = actions
        self.alpha = alpha                      
        self.gamma = gamma                      
        self.epsilon = epsilon                  
        self.epsilon_decay = epsilon_decay      
        self.epsilon_min = epsilon_min          
        self.strategy = strategy                

        self.q_table = {}

    def _get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        return self.q_table[state]
    
    def choose_action(self, state):
        """Implementa a estratégia Epsilon-Greedy com base no parâmetro strategy."""
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        else:
            q_values = self._get_q_values(state)
            return np.argmax(q_values)
        
    def learn(self, state, action, reward, next_state):
        q_values = self._get_q_values(state)
        next_q_values = self._get_q_values(next_state)

        best_next_q = np.max(next_q_values)

        td_target = reward + self.gamma * best_next_q
        td_error = td_target - q_values[action]

        q_values[action] += self.alpha * td_error

        self.q_table[state] = q_values

    def atualizar_epsilon(self):
        """Reduz a exploração apenas se a estratégia for de decaimento."""
        if self.strategy == "epsilon_decay":
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay