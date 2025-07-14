"""
main.py
Script principal para execução via CLI.
Executa todo o pipeline de análise e previsão de criptomoedas.
"""

import argparse
import logging
import sys

from rich import box
from rich.console import Console
from rich.table import Table

from src.data_load import load_crypto_data
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
    logging.info(
        f"Executando pipeline para múltiplas moedas usando modelo {args.model} com {args.kfolds} folds."
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
    for crypto, filepath in cryptos.items():
        try:
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

            print_message(f"Análises concluídas para {crypto}\n", style="bold green")
        except Exception as e:
            logging.warning(f"[{crypto}] Erro ao processar: {e}")

    # c) Comparação de dispersão entre criptomoedas
    if len(dfs) > 1:
        dispersion_df = compare_dispersion(dfs)
        print_dispersion_table(dispersion_df)
    else:
        logging.warning(
            "Menos de duas criptomoedas processadas — não é possível comparar dispersão."
        )


if __name__ == "__main__":
    try:
        setup_logging(LOG_LEVEL)
        main()
    except Exception as e:
        logging.error(f"Erro inesperado na execução: {e}", exc_info=True)
        sys.exit(1)
