"""
main.py
Script principal para execução via CLI.
Executa todo o pipeline de análise e previsão de criptomoedas.

Uso:
    python main.py --crypto BTC --model mlp --kfolds 5
"""

import argparse
import logging
import sys

from rich.console import Console
from rich.table import Table
from rich import box
from src.data_load import load_crypto_data
from src.statistics.analysis import summary_statistics
from src.util.utils import setup_logging
from src.util.config import LOG_LEVEL

console = Console()

def print_stats(stats: dict, crypto: str):
    table = Table(title=f"Medidas Resumo - {crypto}", box=box.SIMPLE_HEAVY)
    table.add_column("Estatística", style="cyan", no_wrap=True)
    table.add_column("Valor", style="magenta")

    for key, value in stats.items():
        table.add_row(str(key), f"{value:.6f}")

    console.print(table)

def print_message(message: str, style: str = "bold green"):
    console.print(message, style=style)

def parse_args():
    """Analisa os argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description="Crypto forecasting")
    parser.add_argument("--crypto", type=str, required=True, help="Código da moeda (ex: BTC)")
    parser.add_argument("--model", type=str, choices=["mlp", "linear", "poly"], required=True, help="Modelo a ser usado")
    parser.add_argument("--kfolds", type=int, default=5, help="Número de folds para cross-validation")
    # Adicione outros argumentos aqui conforme necessário
    return parser.parse_args()

def main():
    """
    Função principal do projeto. Executa o pipeline de previsão.
    """
    args = parse_args()
    logging.info(f"Executando pipeline para {args.crypto} usando modelo {args.model} com {args.kfolds} folds.")

    # Exemplo: carregamento de dados
    # from src.data_load import load_crypto_data
    df = load_crypto_data("data/poloniex_aavebtc_d.csv", parse_dates=["date"])
    logging.info(f"Primeiras linhas:\n{df.head()}")

    stats = summary_statistics(df)
    print_stats(stats, "BTC")
    print_message("Análise concluída!", style="bold blue")

    # Continue pipeline: features, modelagem, etc.
    # ...

if __name__ == "__main__":
    try:
        setup_logging(LOG_LEVEL)
        main()
    except Exception as e:
        logging.error(f"Erro inesperado na execução: {e}", exc_info=True)
        sys.exit(1)
