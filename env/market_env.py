import numpy as np

class MarketEnv:
    def __init__(self, prices, state_type="trend", window_size=3, fast_period=5, slow_period=20):
        """
        prices: array ou lista de preços do ativo ao longo do tempo
        state_type: tipo de abordagem para os estados
        window_size: tamanho da janela para calcular a tendência
        fast_period: período para a média móvel rápida
        slow_period: período para a média móvel lenta
        """
        self.prices = prices
        self.n_steps = len(prices)
        self.current_step = 0

        # Estado do portfólio
        self.initial_balance = 1000.0
        self.balance = self.initial_balance
        self.shares = 0
        self.portfolio_value = self.initial_balance

        # 0 = Líquido (todo o dinheiro está no saldo), 1 = Comprado (todo o dinheiro está em ações)
        self.position = 0

        # Configurações do Estado
        self.state_type = state_type
        self.window_size = window_size

        self.fast_period = fast_period
        self.slow_period = slow_period

    def reset(self):
        """
        Reinicia o ambiente para um novo treinamento
        """
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares = 0
        self.portfolio_value = self.initial_balance
        self.position = 0

        return self._get_state()
    
    def _get_state(self):
        """
        Constrói o estado discretizado.
        Retorna: (posicao_atual, tendencia_preco)
        Tendência: 1 se o preço subiu em relação a ontem, 0 se caiu.
        """
        if self.state_type == "trend":
            trends = []
            
            # Se estamos muito no começo, preenchemos com tendência de "alta" (1) por padrão
            if self.current_step < self.window_size:
                trends = [1] * self.window_size
            else:
                # Pega os preços da janela atual + 1 dia anterior para calcular a variação
                start_idx = self.current_step - self.window_size
                window_prices = self.prices[start_idx : self.current_step + 1]
                
                # Calcula a tendência dia a dia dentro da janela
                for i in range(1, len(window_prices)):
                    if window_prices[i] >= window_prices[i - 1]:
                        trends.append(1) # Subiu ou manteve
                    else:
                        trends.append(0) # Caiu
            
            # Estado vira uma tupla longa: ex (0, 1, 0, 1) -> Posição=0, Subiu, Caiu, Subiu
            return tuple([self.position] + trends)

        elif self.state_type == "ma":
            # Lógica de Médias Móveis
            # Precisamos de dados suficientes para a média lenta
            if self.current_step < self.slow_period:
                ma_signal = 1 # Sinal padrão inicial
            else:
                # Calcula Média Rápida
                fast_ma = np.mean(self.prices[self.current_step - self.fast_period : self.current_step + 1])
                # Calcula Média Lenta
                slow_ma = np.mean(self.prices[self.current_step - self.slow_period : self.current_step + 1])
                
                ma_signal = 1 if fast_ma >= slow_ma else 0
            
            # Estado: (Posição, Cruzamento_MA) -> Apenas 4 estados possíveis (2x2)
            return (self.position, ma_signal)

        else: # Fallback para o modo antigo "simple" (apenas 1 dia)
            if self.current_step == 0:
                trend = 1
            else:
                trend = 1 if self.prices[self.current_step] >= self.prices[self.current_step - 1] else 0
            return (self.position, trend)
    
    def step(self, action):
        """
        Executa a ação escolhida pelo agente.
        Ações: 0 (Manter), 1 (Comprar), 2 (Vender)
        """
        current_price = self.prices[self.current_step]
        # a ser testado: penalidade por não conseguir comprar ou tentar comprar quando já está comprado
        # step_penalty = 0 # Inicializa penalidade extra zerada   

        # Lógica da ação
        if action == 1 and self.position == 0: # Tenta comprar o que der
            n_purchased = self.balance // current_price

            if n_purchased > 0:
                self.shares = n_purchased
                self.balance -= n_purchased * current_price
                self.position = 1
            else:
                # O agente tentou comprar, mas o caixa não compra sequer 1 ação.
                # A ação é tratada como "Manter" ou reward negativo DECIDIR DPS
                # step_penalty = -5
                pass
            
        elif action == 2 and self.position == 1: # Vende o que der
            self.balance += self.shares * current_price
            self.shares = 0
            self.position = 0

        elif action == 1 and self.position == 1:
            # Punição por tentar comprar quando já está comprado (opcional, mas ajuda)
            # step_penalty = -5
            pass

        # Ação 0 não altera nada

        # Avança no tempo
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1

        # Calcula a recompensa
        new_price = self.prices[self.current_step]
        new_portfolio_value = self.balance + (self.shares * new_price)

        reward = new_portfolio_value - self.portfolio_value
        # reward += step_penalty # Aplica a penalidade extra, se houver
        self.portfolio_value = new_portfolio_value

        next_state = self._get_state()

        return next_state, reward, done

