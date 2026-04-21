import numpy as np

class MarketEnv:
    def __init__(self, prices):
        """
        prices: array ou lista de preços do ativo ao longo do tempo
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
        if self.current_step == 0:
            trend = 1

        else:
            if self.prices[self.current_step] >= self.prices[self.current_step - 1]:
                trend = 1
            else:
                trend = 0
        
        return (self.position, trend)
    
    def step(self, action):
        """
        Executa a ação escolhida pelo agente.
        Ações: 0 (Manter), 1 (Comprar), 2 (Vender)
        """
        current_price = self.prices[self.current_step]

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
                pass
            
        elif action == 2 and self.position == 1: # Vende o que der
            self.balance += self.shares * current_price
            self.shares = 0
            self.position = 0

        # Ação 0 não altera nada

        # Avança no tempo
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1

        # Calcula a recompensa
        new_price = self.prices[self.current_step]
        new_portfolio_value = self.balance + (self.shares * new_price)

        reward = new_portfolio_value - self.portfolio_value
        self.portfolio_value = new_portfolio_value

        next_state = self._get_state()

        return next_state, reward, done

