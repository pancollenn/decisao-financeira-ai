import yfinance as yf
import numpy as np

def obter_dados_reais(ticker="^BVSP", start_date="2020-01-01", end_date="2023-01-01"):
    """
    Baixa o histórico de preços de fechamento de um ativo usando o Yahoo Finance.
    
    Parâmetros:
    - ticker: O símbolo do ativo (ex: '^BVSP' para Ibovespa, 'PETR4.SA' para Petrobras, 'AAPL' para Apple).
    - start_date: Data de início (AAAA-MM-DD).
    - end_date: Data de fim (AAAA-MM-DD).
    
    Retorna:
    - Um numpy array com os preços de fechamento.
    """
    print(f"Baixando dados reais para {ticker} de {start_date} até {end_date}...")
    
    # Baixa os dados
    dados = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    # Pega apenas a coluna 'Close' (Preço de Fechamento) e remove valores nulos (NaN)
    precos_fechamento = dados['Close'].dropna().values
    
    # yfinance pode retornar um array 2D dependendo da versão, garantimos que seja 1D
    precos_1d = precos_fechamento.flatten()
    
    print(f"Sucesso! {len(precos_1d)} dias de dados carregados.")
    return precos_1d

if __name__ == "__main__":
    # Teste rápido
    precos = obter_dados_reais("PETR4.SA", "2022-01-01", "2022-12-31")
    print(f"Primeiros 5 preços: {precos[:5]}")