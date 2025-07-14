"""
main.py
Script principal para execução via CLI.
Executa todo o pipeline de análise e previsão de criptomoedas.
"""

import argparse
import logging
import sys

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from rich import box
from rich.console import Console
from rich.table import Table

from src.data_load import load_crypto_data
from src.features import criar_features_basicas_completas
from src.models import train_mlp, train_linear, treinar_regressao_polinomial, validacao_cruzada_kfold
from src.lucro import calcular_lucro_investimento, calcular_estrategia_buy_and_hold
from src.statistics.analysis import compare_dispersion, summary_statistics
from src.statistics.plots import plot_boxplot, plot_histogram, plot_price_with_summary
from src.util.config import LOG_LEVEL
from src.util.utils import setup_logging

console = Console()


def print_dispersion_table(dispersion_table):
    table = Table(title="Dispersão entre criptomoedas")
    for col in dispersion_table.columns:
        table.add_column(str(col), style="cyan")
    for _, row in dispersion_table.iterrows():
        table.add_row(*[f"{x:.6f}" if isinstance(x, float) else str(x) for x in row])
    console.print(table)


def print_stats(stats: dict, crypto: str):
    table = Table(
        title=f"Medidas resumo e de dispersão - {crypto}", box=box.SIMPLE_HEAVY
    )
    table.add_column("Estatística", style="cyan", no_wrap=True)
    table.add_column("Valor", style="magenta")
    for key, value in stats.items():
        table.add_row(str(key), f"{value:.6f}")
    console.print(table)


def print_message(message: str, style: str = "bold green"):
    console.print(message, style=style)


def print_profit_results(crypto: str, profit_model, profit_buyhold):
    table = Table(title=f"Resultados de Lucro - {crypto}", box=box.SIMPLE_HEAVY)
    table.add_column("Estratégia", style="cyan", no_wrap=True)
    table.add_column("Lucro (R$)", style="magenta")
    table.add_column("Melhor?", style="green")
    
    is_model_better = profit_model > profit_buyhold
    
    table.add_row("Modelo ML", f"{profit_model:.2f}", "✓" if is_model_better else "")
    table.add_row("Buy & Hold", f"{profit_buyhold:.2f}", "✓" if not is_model_better else "")
    
    console.print(table)


def treinar_modelo_escolhido(model_type: str, X_train, y_train):
    """Treina o modelo escolhido pelo usuário"""
    if model_type == "linear":
        return train_linear(X_train, y_train)
    elif model_type == "mlp":
        return train_mlp(X_train, y_train)
    elif model_type == "poly":
        model, poly_transformer = treinar_regressao_polinomial(X_train, y_train)
        return model, poly_transformer
    else:
        raise ValueError(f"Modelo {model_type} não reconhecido!")


def preparar_features_para_modelo(df_with_features):
    """Prepara as features e target para treinamento"""
    # Remove linhas com valores NaN (que aparecem devido às médias móveis)
    df_clean = df_with_features.dropna()
    
    # Features: média móvel, volatilidade, retorno diário, preço subiu
    feature_columns = ['media_movel_7d', 'volatilidade_7d', 'retorno_diario', 'preco_subiu']
    X = df_clean[feature_columns].values
    
    # Target: preço de fechamento do próximo dia
    y = df_clean['close'].shift(-1).dropna().values
    
    # Remove a última linha de X para alinhar com y
    X = X[:-1]
    
    return X, y, df_clean[:-1]


def parse_args():
    parser = argparse.ArgumentParser(description="Crypto forecasting")
    parser.add_argument(
        "--model",
        type=str,
        choices=["mlp", "linear", "poly"],
        required=True,
        help="Modelo a ser usado",
    )
    parser.add_argument(
        "--kfolds", type=int, default=5, help="Número de folds para cross-validation"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print_message(
        f"🚀 Executando pipeline usando modelo {args.model} com {args.kfolds} folds", 
        style="bold blue"
    )

    # 1. Defina os arquivos das 10 criptomoedas (ajuste paths conforme sua organização)
    cryptos = {
        "ADA": "data/Poloniex_ADAUSDT_d.csv",
        "AKITA": "data/Poloniex_AKITAUSDT_d.csv",
        "ALICE": "data/Poloniex_ALICEUSDT_d.csv",
        "ALPACA": "data/Poloniex_ALPACAUSDT_d.csv",
        "ALPINE": "data/Poloniex_ALPINEUSDT_d.csv",
        "BICO": "data/Poloniex_BICOUSDT_d.csv",
        "BTC": "data/Poloniex_BTCUSDT_d.csv",
        "CHR": "data/Poloniex_CHRUSDT_d.csv",
        "COOL": "data/Poloniex_COOLUSDT_d.csv",
        "CORN": "data/Poloniex_CORNUSDT_d.csv",
    }

    dfs = {}
    stats_dict = {}
    all_profits_model = []
    all_profits_buyhold = []
    
    for crypto, filepath in cryptos.items():
        try:
            print_message(f"\n📊 Processando {crypto}...", style="bold yellow")
            
            # Carrega dados
            df = load_crypto_data(filepath, parse_dates=["date"])
            dfs[crypto] = df

            # a) Medidas resumo e dispersão
            stats = summary_statistics(df)
            stats_dict[crypto] = stats
            print_stats(stats, crypto)

            # b) Boxplot e histograma
            plot_boxplot(df, crypto=crypto)
            plot_histogram(df, crypto=crypto)

            # d) Gráfico de linha com preço + média, mediana, moda
            plot_price_with_summary(df, crypto=crypto)

            # === NOVO: PIPELINE DE MACHINE LEARNING ===
            
            # 2. Criar features
            print_message("🔧 Criando features...", style="cyan")
            df_with_features = criar_features_basicas_completas(df)
            
            # 3. Preparar dados para modelo
            X, y, df_clean = preparar_features_para_modelo(df_with_features)
            
            if len(X) < 20:  # Precisa de dados suficientes
                print_message(f"❌ {crypto}: Poucos dados para treinamento", style="red")
                continue
            
            # 4. Dividir dados treino/teste
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # 5. Normalizar features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 6. Treinar modelo escolhido
            print_message(f"🤖 Treinando modelo {args.model}...", style="cyan")
            
            if args.model == "poly":
                model, poly_transformer = treinar_modelo_escolhido(args.model, X_train_scaled, y_train)
                # Para poly, precisa aplicar transformação polinomial
                X_test_poly = poly_transformer.transform(X_test_scaled)
                predictions = model.predict(X_test_poly)
            else:
                model = treinar_modelo_escolhido(args.model, X_train_scaled, y_train)
                predictions = model.predict(X_test_scaled)
            
            # 7. Calcular validação cruzada
            print_message(f"✅ Fazendo validação cruzada com {args.kfolds} folds...", style="cyan")
            errors, mean_error = validacao_cruzada_kfold(X, y, args.kfolds)
            
            # 8. Calcular lucros
            print_message("💰 Calculando lucros...", style="green")
            
            # Preços reais do período de teste
            test_prices = y_test
            
            # Lucro do modelo
            _, profit_model, _ = calcular_lucro_investimento(test_prices, predictions)
            
            # Lucro buy-and-hold
            profit_buyhold = calcular_estrategia_buy_and_hold(test_prices)
            
            # Mostrar resultados
            print_profit_results(crypto, profit_model, profit_buyhold)
            
            # Guardar para estatística final
            all_profits_model.append(profit_model)
            all_profits_buyhold.append(profit_buyhold)

            print_message(f"✅ {crypto} processado com sucesso!\n", style="bold green")
            
        except Exception as e:
            print_message(f"❌ [{crypto}] Erro ao processar: {e}", style="red")
            logging.warning(f"[{crypto}] Erro ao processar: {e}")

    # === ESTATÍSTICAS FINAIS ===
    print_message("\n📈 RESUMO GERAL", style="bold magenta")
    
    # c) Comparação de dispersão entre criptomoedas
    if len(dfs) > 1:
        dispersion_df = compare_dispersion(dfs)
        print_dispersion_table(dispersion_df)
    
    # Estatísticas de lucro
    if all_profits_model:
        avg_profit_model = np.mean(all_profits_model)
        avg_profit_buyhold = np.mean(all_profits_buyhold)
        
        table = Table(title="Resumo de Lucros (Média)", box=box.SIMPLE_HEAVY)
        table.add_column("Estratégia", style="cyan")
        table.add_column("Lucro Médio (R$)", style="magenta")
        table.add_column("Melhor?", style="green")
        
        is_model_better = avg_profit_model > avg_profit_buyhold
        
        table.add_row(f"Modelo {args.model}", f"{avg_profit_model:.2f}", "✓" if is_model_better else "")
        table.add_row("Buy & Hold", f"{avg_profit_buyhold:.2f}", "✓" if not is_model_better else "")
        
        console.print(table)
    
    print_message("🎉 Pipeline completo executado!", style="bold green")


if __name__ == "__main__":
    try:
        setup_logging(LOG_LEVEL)
        main()
    except Exception as e:
        logging.error(f"Erro inesperado na execução: {e}", exc_info=True)
        sys.exit(1)
