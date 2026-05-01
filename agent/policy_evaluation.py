import numpy as np
from collections import defaultdict


class ModelBasedEvaluator:
    def __init__(self, env, gamma=0.95):
        """
        Inicializa o avaliador que construirá um modelo estocástico a partir do simulador.
        """
        self.env = env
        self.gamma = gamma

        # Estruturas de dados para o Modelo do MDP
        # transitions[estado][acao][proximo_estado] = probabilidade P(s'|s,a)
        self.transitions = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

        # rewards[estado][acao][proximo_estado] = recompensa R(s,a,s')
        self.rewards = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

        self.states = set()
        self.actions = [0, 1, 2]  # 0: Manter, 1: Comprar, 2: Vender

    def build_empirical_model(self, episodes=500):
        """
        Fase 1: Mapeamento do Ambiente.
        Executa episódios simulados executando ações aleatórias para construir o Grafo do MDP
        (frequências de transição) a partir da série temporal histórica de preços.
        """
        print("Construindo matrizes de transição T(s,a,s') e recompensas R(s,a,s')...")
        transition_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        reward_sums = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

        for _ in range(episodes):
            estado = self.env.reset()
            self.states.add(estado)
            done = False

            while not done:
                # Amostra ações para explorar todas as ramificações possíveis
                acao = np.random.choice(self.actions)
                proximo_estado, recompensa, done = self.env.step(acao)

                self.states.add(proximo_estado)

                # Contabiliza a transição
                transition_counts[estado][acao][proximo_estado] += 1
                reward_sums[estado][acao][proximo_estado] += recompensa

                estado = proximo_estado

        # Converte as contagens brutas em probabilidades (Normalização)
        for s in transition_counts:
            for a in transition_counts[s]:
                total_transitions = sum(transition_counts[s][a].values())
                for next_s, count in transition_counts[s][a].items():
                    # P(s'|s,a)
                    self.transitions[s][a][next_s] = count / total_transitions
                    # R(s,a,s') médio
                    self.rewards[s][a][next_s] = reward_sums[s][a][next_s] / count

        print(f"Modelo construído! {len(self.states)} estados discretos mapeados.")

    def evaluate_policy(self, policy, theta=1e-5):
        """
        Fase 2: Policy Evaluation (Avaliação de Política).
        Aplica iterativamente a Equação de Expectativa de Bellman para encontrar V(s).

        policy: Dicionário mapeando estado -> lista de probabilidades [P(a=0), P(a=1), P(a=2)].
                Exemplo de política aleatória uniforme: {s: [0.33, 0.33, 0.33]}
        theta: Limiar de convergência. O loop para quando a maior alteração em V(s) for menor que theta.
        """
        print("Iniciando avaliação da política (Programação Dinâmica)...")
        # Inicializa a Tabela de Valores V(s) com zeros
        V = {s: 0.0 for s in self.states}
        iteracao = 0

        while True:
            delta = 0

            for s in self.states:
                v_old = V[s]
                v_new = 0.0

                # Recupera a distribuição de probabilidade das ações para este estado, dada a política atual pi(a|s).
                # Se o estado não estiver na política, assume distribuição uniforme provisória.
                action_probs = policy.get(s, [1 / 3, 1 / 3, 1 / 3])

                # Somatório sobre todas as ações 'a' no estado 's'
                for a in self.actions:
                    prob_a = action_probs[a]

                    if prob_a == 0:
                        continue  # Pula cálculos desnecessários se a política nunca escolhe esta ação

                    sum_transitions = 0.0

                    # Somatório sobre todos os próximos estados possíveis 's''
                    for next_s, prob_t in self.transitions[s][a].items():
                        reward = self.rewards[s][a][next_s]

                        # O cerne da Equação de Bellman: T(s,a,s') * [R(s,a,s') + gamma * V(s')]
                        sum_transitions += prob_t * (reward + self.gamma * V[next_s])

                    # Pondera o valor esperado pela probabilidade de tomar a ação 'a'
                    v_new += prob_a * sum_transitions

                # Atualiza o valor do estado
                V[s] = v_new

                # Rastreia a maior mudança observada nesta iteração (para critério de parada)
                delta = max(delta, abs(v_old - v_new))

            iteracao += 1
            if delta < theta:
                print(f"Convergência alcançada após {iteracao} iterações. (Delta = {delta:.6f})")
                break

        return V

    def value_iteration(self, theta=1e-5):
        """
        Aplica a Iteração de Valor para encontrar a política ótima.
        Retorna a Função de Valor ótima V*(s) e a Política Ótima pi*(s).
        """
        V = {s: 0.0 for s in self.states}

        # 1. Encontra os Valores Ótimos V*(s)
        while True:
            delta = 0
            for s in self.states:
                v_old = V[s]
                max_v = float('-inf')

                for a in self.actions:
                    sum_transitions = 0.0
                    for next_s, prob in self.transitions[s][a].items():
                        reward = self.rewards[s][a][next_s]
                        sum_transitions += prob * (reward + self.gamma * V[next_s])

                    if sum_transitions > max_v:
                        max_v = sum_transitions

                # Se não houver transições (estado terminal no modelo), mantemos 0
                if max_v != float('-inf'):
                    V[s] = max_v

                delta = max(delta, abs(v_old - V[s]))

            if delta < theta:
                break

        # 2. Extrai a Política Ótima (Greedy policy em relação a V*)
        optimal_policy = {}
        for s in self.states:
            best_a = 0
            max_v = float('-inf')

            for a in self.actions:
                sum_transitions = 0.0
                for next_s, prob in self.transitions[s][a].items():
                    reward = self.rewards[s][a][next_s]
                    sum_transitions += prob * (reward + self.gamma * V[next_s])

                if sum_transitions > max_v:
                    max_v = sum_transitions
                    best_a = a

            optimal_policy[s] = best_a

        return V, optimal_policy

if __name__ == "__main__":
    from env.market_env import MarketEnv
    import numpy as np

    # Dados de teste (deve ser substituído pelos dados do seu gerador)
    t = np.linspace(0, 50, 200)
    precos_teste = 100 + 20 * np.sin(t) + np.random.normal(0, 2, 200)

    # Instancia o ambiente no modo MA (Médias Móveis) pois gera um espaço de estados menor e mais fácil de avaliar
    env_ma = MarketEnv(prices=precos_teste, state_type="ma", fast_period=5, slow_period=20)

    # 1. Constrói o modelo do MDP
    evaluator = ModelBasedEvaluator(env_ma, gamma=0.95)
    evaluator.build_empirical_model(episodes=1000)

    # 2. Define uma política a ser avaliada.
    # Neste exemplo, uma política que compra(1) se MA rápida cruzar lenta para cima, e vende(2) caso contrário.
    # Lembre-se: Estado no MA = (posicao, sinal_ma). Ações = [0:Manter, 1:Comprar, 2:Vender]
    minha_politica = {}
    for s in evaluator.states:
        posicao, sinal_ma = s
        if sinal_ma == 1 and posicao == 0:
            minha_politica[s] = [0.0, 1.0, 0.0]  # 100% de chance de comprar
        elif sinal_ma == 0 and posicao == 1:
            minha_politica[s] = [0.0, 0.0, 1.0]  # 100% de chance de vender
        else:
            minha_politica[s] = [1.0, 0.0, 0.0]  # 100% de chance de manter

    # 3. Executa a avaliação da política
    valores_estimados = evaluator.evaluate_policy(minha_politica)

    print("\nValores Estimados V(s) para os estados mapeados:")
    for estado, valor in valores_estimados.items():
        print(f"Estado {estado} -> V(s) = {valor:.4f}")