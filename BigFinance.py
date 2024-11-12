import yfinance as yf
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings
import pandas as pd
import mplfinance as mpf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np


# Ignorar avisos
warnings.filterwarnings("ignore")

# Dicionário com os nomes completos das commodities e suas unidades
commodity_names = {
    'CC=F': 'Cacau (USD/tonelada)',
    'KC=F': 'Café (USD/libra-peso)',
    'SB=F': 'Açúcar (USD/libra-peso)',  
    'CT=F': 'Algodão (USD/libra-peso)',
    'ZS=F': 'Soja (centavos de USD/bushel)',
    'LE=F': 'Carne Bovina (USD/libra-peso)',
    'TIO=F': 'Minério de Ferro (USD/tonelada)'
}

# Função para calcular o RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    data['RSI'] = rsi
    return data


# Função para obter dados do Yahoo Finance com velas diárias
def get_data(tickers, interval="1d"):
    data = {}
    for ticker in tickers:
        stock_data = yf.Ticker(ticker)
        df = stock_data.history(period="1y", interval=interval)
        if df.empty:
            print(f"Nenhum dado retornado para {ticker}.")
        else:
            data[ticker] = df[['Open', 'High', 'Low', 'Volume', 'Close']]
    return data


# Função de aprendizado de máquina para prever sinais de compra e venda com base no RSI
# Função de aprendizado de máquina para prever sinais de compra e venda com base no RSI
def machine_learning_rsi(data, ticker):
    prices = data[ticker]
    
    # Calcular o RSI
    prices = calculate_rsi(prices)
    
    # Criar variáveis preditoras: RSI, e preço de fechamento
    prices['Prev_Close'] = prices['Close'].shift(1)
    prices.dropna(inplace=True)
    
    # Definir sinais de compra e venda
    prices['Signal'] = 0  # Nenhuma ação
    prices['Signal'][prices['RSI'] <= 30] = 1  # Compra
    prices['Signal'][prices['RSI'] >= 70] = -1  # Venda

    # Variáveis independentes (features)
    X = prices[['Prev_Close', 'RSI']]

    # Variável dependente (target)
    y = prices['Signal']

    # Dividir dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Treinar o modelo de classificação
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Fazer previsões
    y_pred = model.predict(X_test)

    # Avaliar o modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Acurácia do modelo: {accuracy:.2f}')

    # Adicionar sinais de compra e venda no gráfico
    return prices, model, accuracy  # Agora a função retorna três valores

# Função para plotar gráficos de velas com volume e RSI, incluindo sinais de compra e venda
def plot_candlestick(data, title, frame):
    for widget in frame.winfo_children():
        widget.destroy()

    for ticker, prices in data.items():
        # Obter o modelo de Machine Learning e prever os sinais
        prices, model, accuracy = machine_learning_rsi(data, ticker)  # Passando a acurácia

        # Criar o gráfico de velas
        fig, axes = mpf.plot(prices, type='candle', volume=True, show_nontrading=True,
                             title=title, style='charles', returnfig=True)

        # Ajustar o limite do eixo x para mostrar mais velas
        ax_candle = fig.axes[0]
        ax_candle.set_xlim(prices.index[-120], prices.index[-1])  # Mostra as últimas 40 velas para um melhor zoom

        # Adicionar o gráfico de RSI como o segundo subplot
        ax_rsi = fig.add_subplot(3, 1, 2, sharex=ax_candle)
        ax_rsi.plot(prices.index, prices['RSI'], color='orange', label='RSI')
        ax_rsi.axhline(70, color='red', linestyle='--', label='Overbought')
        ax_rsi.axhline(30, color='green', linestyle='--', label='Oversold')
        ax_rsi.set_ylabel('RSI')
        ax_rsi.set_ylim(0, 100)  # Limitar o eixo y do RSI

        # Adicionar sinais de compra e venda com base no modelo de ML
        for i in range(1, len(prices['RSI'])):
            if prices['Signal'][i] == 1:
                ax_rsi.annotate('COMPRA', xy=(prices.index[i], prices['RSI'][i]),
                                xytext=(prices.index[i], prices['RSI'][i]-10),
                                arrowprops=dict(facecolor='green', shrink=0.05),
                                fontsize=10, color='green')
            elif prices['Signal'][i] == -1:
                ax_rsi.annotate('VENDA', xy=(prices.index[i], prices['RSI'][i]),
                                xytext=(prices.index[i], prices['RSI'][i]+10),
                                arrowprops=dict(facecolor='red', shrink=0.05),
                                fontsize=10, color='red')

        ax_rsi.legend(loc='upper left')

        # Adicionar a acurácia no gráfico
        ax_rsi.text(0.5, 0.1, f"Precisão 1=100%     : {accuracy:.2f}", transform=ax_rsi.transAxes,
                    fontsize=12, color='blue', ha='center', va='top', bbox=dict(facecolor='white', alpha=0.7))

        # Adicionar o gráfico de volume como o terceiro subplot
        ax_volume = fig.add_subplot(3, 1, 3, sharex=ax_candle)
        ax_volume.bar(prices.index, prices['Volume'], color='blue')
        ax_volume.set_ylabel('')
        ax_volume.set_aspect(2)  # Ajusta a altura do gráfico de volume

        # Ajustar layout
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)

        # Obter o último preço de fechamento e exibir no gráfico de velas
        last_price = prices['Close'][-1]
        ax_candle.text(0.01, 0.95, f"{ticker}: {last_price:.2f}", transform=ax_candle.transAxes, fontsize=12,
                       color='black', ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7))

        # Criar o canvas para o gráfico
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Função para plotar gráficos de linhas com bordas arredondadas
def plot_line(data, title, frame, labels=None, unit=''):
    for widget in frame.winfo_children():
        widget.destroy()

    fig, ax = plt.subplots(figsize=(10, 5))

    for ticker, prices in data.items():
        if not prices.empty:  # Verifica se o DataFrame não está vazio
            last_price = prices['Close'][-1]  # Obtém a última cotação
            label = f"{labels[ticker]}: {last_price:.2f} {unit}" if labels and ticker in labels else f"{ticker}: {last_price:.2f} {unit}"  # Inclui o valor e unidade no label
            ax.plot(prices.index, prices['Close'], label=label)

    ax.set_title(title + f" ({unit})")
    ax.legend(loc='upper left')
    
    # Adiciona régua de preços à direita
    ax.yaxis.set_ticks_position('right')
    ax.yaxis.set_label_position("right")

    # Adiciona um retângulo com bordas arredondadas
    rect = patches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.03", color="lightgrey", transform=ax.transAxes, zorder=-1)
    ax.add_patch(rect)

    fig.suptitle(title, fontsize=10)
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def plot_pie(frame):
    for widget in frame.winfo_children():
        widget.destroy()

    # Dados para o gráfico de capitalização de mercado das empresas de tecnologia
    companies = ['Apple', 'Microsoft', 'Amazon', 'Alphabet', 'Meta', 'Tesla', 'NVIDIA']
    market_caps = [2.8, 2.5, 1.3, 1.6, 0.7, 0.8, 1.0]
    company_info = [
        "Apple: Líder em eletrônicos e tecnologia móvel.",
        "Microsoft: Dominante em software e serviços de nuvem.",
        "Amazon: Maior varejista online do mundo.",
        "Alphabet: Empresa-mãe do Google, foco em IA e publicidade.",
        "Meta: Expansão em redes sociais e metaverso.",
        "Tesla: Pioneira em veículos elétricos e energia renovável.",
        "NVIDIA: Líder em GPUs e computação gráfica."
    ]

    # Dados para o gráfico de distribuição de setores no S&P 500
    sectors = ['Tecnologia', 'Saúde', 'Financeiro', 'Consumo', 'Industrial', 'Outros']
    sector_distribution = [27, 14, 13, 10, 8, 28]
    sector_info = [
        "Tecnologia: Maior participação no índice, impulsionada pela inovação.",
        "Saúde: Representa farmacêuticas e biotecnologia.",
        "Financeiro: Inclui bancos e seguradoras.",
        "Consumo: Empresas de bens e serviços ao consumidor.",
        "Industrial: Englobando manufatura e infraestrutura.",
        "Outros: Diversos setores com menor impacto individual."
    ]

    # Dados para o gráfico de participação regional de mercado
    regions = ['América do Norte', 'Europa', 'Ásia', 'América Latina', 'Outros']
    regional_distribution = [45, 25, 20, 5, 5]
    region_info = [
        "América do Norte: Maior mercado global, liderado por EUA.",
        "Europa: Forte em tecnologia e finanças.",
        "Ásia: Economia em crescimento, foco em manufatura.",
        "América Latina: Mercado emergente com grande potencial.",
        "Outros: Diversos mercados menores ao redor do mundo."
    ]

    # Configurar a grade para os gráficos de pizza
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Gráfico de pizza 1: Capitalização de Mercado
    axs[0].pie(market_caps, labels=companies, autopct='%1.1f%%', startangle=140)
    axs[0].set_title("Participação de Mercado - Principais Empresas de Tecnologia")

    # Gráfico de pizza 2: Distribuição de Setores no S&P 500
    axs[1].pie(sector_distribution, labels=sectors, autopct='%1.1f%%', startangle=140)
    axs[1].set_title("Distribuição de Setores no S&P 500")

    # Gráfico de pizza 3: Participação Regional de Mercado
    axs[2].pie(regional_distribution, labels=regions, autopct='%1.1f%%', startangle=100)
    axs[2].set_title("Participação Regional de Mercado")

    # Criar o canvas para os gráficos e adicionar ao frame
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Frame para texto descritivo ao redor dos gráficos
    info_frame = tk.Frame(frame)
    info_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    # Adicionando rótulos informativos ao lado dos gráficos
    # Informações das empresas
    for i, info in enumerate(company_info):
        tk.Label(info_frame, text=info, font=("Arial", 14), anchor='w').grid(row=i, column=0, sticky='w', pady='5', padx='50')

    # Informações dos setores
    for i, info in enumerate(sector_info):
        tk.Label(info_frame, text=info, font=("Arial", 14), anchor='w').grid(row=i, column=1, sticky='w', pady='5', padx='50')

    # Informações das regiões
    for i, info in enumerate(region_info):
        tk.Label(info_frame, text=info, font=("Arial", 14), anchor='w').grid(row=i, column=2, sticky='w', pady='5', padx='10')

# Função para configurar a interface
def setup_interface(root):
    root.title("Análise de Big Data - Yahoo Finance")
    root.geometry("2000x1050")

    # Cria um notebook para as abas
    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True)

    # Aumenta o tamanho da fonte das abas
    style = ttk.Style()
    style.configure("TNotebook.Tab", font=("Arial", 14))
    style.configure("TNotebook", background="lightgray")

    # Estilo para as abas quando selecionadas
    style.map("TNotebook.Tab", background=[("selected", "green"), ("active", "lightgreen")])

    # Estilo para bordas e arredondamento
    style.configure("TNotebook", borderwidth=5)
    style.configure("TNotebook.Tab", borderwidth=2, relief="solid")

    # 1. Commodities, Bancos Nacionais e Ações (Unidos)
    commodities = ['CC=F', 'KC=F', 'SB=F', 'CT=F', 'ZS=F', 'LE=F', 'TIO=F']
    commodities_data = get_data(commodities)
    
    bancos = ['ITUB4.SA', 'BBDC3.SA', 'BBDC4.SA', 'BBAS3.SA', 'SANB11.SA', 
              'BPAC11.SA', 'BMGB4.SA', 'BRSR6.SA', 'CVCB3.SA', 'PSSA3.SA']
    bancos_data = get_data(bancos)

    # Gráficos das Ações
    brasil_stocks = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'B3SA3.SA', 'MGLU3.SA']
    brasil_stocks_data = get_data(brasil_stocks)

    usa_stocks = ['AAPL', 'AMZN', 'GOOGL', 'TSLA', 'MSFT']
    usa_stocks_data = get_data(usa_stocks)

    combined_frame = tk.Frame(notebook)
    notebook.add(combined_frame, text='Commodities, Bancos e Ações')

    # Dividir o frame em uma grade 2x2
    left_frame_top = tk.Frame(combined_frame)
    left_frame_top.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')
    left_frame_bottom = tk.Frame(combined_frame)
    left_frame_bottom.grid(row=1, column=0, padx=10, pady=10, sticky='nsew')
    right_frame_top = tk.Frame(combined_frame)
    right_frame_top.grid(row=0, column=1, padx=10, pady=10, sticky='nsew')
    right_frame_bottom = tk.Frame(combined_frame)
    right_frame_bottom.grid(row=1, column=1, padx=10, pady=10, sticky='nsew')

    combined_frame.grid_rowconfigure(0, weight=1)
    combined_frame.grid_rowconfigure(1, weight=1)
    combined_frame.grid_columnconfigure(0, weight=1)
    combined_frame.grid_columnconfigure(1, weight=1)

    # Gráficos de Commodities
    plot_line(commodities_data, 'Principais Commodities', left_frame_bottom, labels=commodity_names, unit='USD')

    # Gráficos de Bancos Nacionais
    plot_line(bancos_data, 'Principais Bancos Nacionais', left_frame_top, unit='BRL')

    # Gráficos de Ações Brasileiras
    plot_line(brasil_stocks_data, 'Principais Ações Brasileiras', right_frame_top, unit='BRL')

    # Gráficos de Ações Americanas
    plot_line(usa_stocks_data, 'Principais Ações Americanas', right_frame_bottom, unit='USD')

    # 2. Índices: Bovespa e S&P 500
    indices_frame = tk.Frame(notebook)
    notebook.add(indices_frame, text='Índices: Bovespa e S&P 500')

    # Dividindo o frame em dois subframes
    bovespa_frame = tk.Frame(indices_frame)
    bovespa_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
    sp500_frame = tk.Frame(indices_frame)
    sp500_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Índice Bovespa
    indices_bovespa = ['^BVSP']
    bovespa_data = get_data(indices_bovespa)
    plot_candlestick(bovespa_data, 'Gráfico do Índice Bovespa', bovespa_frame)
    
    # Índice S&P 500
    indices_sp500 = ['^GSPC']
    sp500_data = get_data(indices_sp500)
    plot_candlestick(sp500_data, 'Gráfico do S&P 500', sp500_frame)

    # 3. Corretoras Estrangeiras e índices e criptomoedas
    brokers_frame = tk.Frame(notebook)
    
    notebook.add(brokers_frame, text='Ativos Diversos')

    # Dividir o frame de corretoras em 2x2 subframes
    frame_left_top = tk.Frame(brokers_frame)
    frame_left_top.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')
    frame_right_top = tk.Frame(brokers_frame)
    frame_right_top.grid(row=0, column=1, padx=10, pady=10, sticky='nsew')
    frame_left_bottom = tk.Frame(brokers_frame)
    frame_left_bottom.grid(row=1, column=0, padx=10, pady=10, sticky='nsew')
    frame_right_bottom = tk.Frame(brokers_frame)
    frame_right_bottom.grid(row=1, column=1, padx=10, pady=10, sticky='nsew')

    brokers_frame.grid_rowconfigure(0, weight=1)
    brokers_frame.grid_rowconfigure(1, weight=1)
    brokers_frame.grid_columnconfigure(0, weight=1)
    brokers_frame.grid_columnconfigure(1, weight=1)

    # Gráficos das corretoras brasileiras
    brokers_brazil = ['XPBR31.SA', 'CASH3.SA', 'BPAC11.SA', 'BBSE3.SA']
    brokers_brazil_data = get_data(brokers_brazil)
    plot_line(brokers_brazil_data, 'Corretoras Brasileiras', frame_left_top, unit='BRL')

    # Gráficos das corretoras estrangeiras
    brokers_foreign = ['MS', 'UBS', 'JPM', 'MERR', 'GS']
    brokers_foreign_data = get_data(brokers_foreign)
    plot_line(brokers_foreign_data, 'Corretoras Estrangeiras', frame_right_top, unit='USD')

    # Gráficos dos principais índices estrangeiros
    indices_foreign = ['^DJI', '^IXIC', '^FTSE', '^N225']  # Exemplo de índices
    indices_data = get_data(indices_foreign)
    plot_line(indices_data, 'Principais Índices Estrangeiros', frame_left_bottom, unit='USD')

    # Gráficos de criptomoedas
    cryptos = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'DOGE-USD']  # Exemplo de criptomoedas
    cryptos_data = get_data(cryptos)
    plot_line(cryptos_data, 'Criptomoedas', frame_right_bottom, unit='USD')

    # 4. Nova aba com gráfico de pizza de participação de mercado
    market_share_frame = tk.Frame(notebook)
    notebook.add(market_share_frame, text='Participação de Mercado')
    
    # Chamando a função para plotar o gráfico de pizza
    plot_pie(market_share_frame)

    return notebook

# Função principal para execução da interface\
def main():
    root = tk.Tk()
    setup_interface(root)
    
    root.mainloop()

if __name__ == "__main__":
    main()



