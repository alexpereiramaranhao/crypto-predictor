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

from src.statistics.analysis import anova_between_cryptos, tukey_posthoc

from src.data_load import load_crypto_data
from src.statistics.analysis import summary_statistics, compare_dispersion
from src.statistics.plots import plot_boxplot, plot_histogram, plot_price_with_summary
from src.util.config import LOG_LEVEL
from src.util.utils import setup_logging
from src.features import build_features

SIGNIFICANCE_LEVEL = 0.05

console = Console()

def print_dispersion_table(dispersion_table):
    table = Table(title="Dispersão entre criptomoedas")
    for col in dispersion_table.columns:
        table.add_column(str(col), style="cyan")
    for _, row in dispersion_table.iterrows():
        table.add_row(*[f"{x:.6f}" if isinstance(x, float) else str(x) for x in row])
    console.print(table)

def print_stats(stats: dict, crypto: str):
    table = Table(title=f"Medidas resumo e de dispersão - {crypto}", box=box.SIMPLE_HEAVY)
    table.add_column("Estatística", style="cyan", no_wrap=True)
    table.add_column("Valor", style="magenta")
    for key, value in stats.items():
        table.add_row(str(key), f"{value:.6f}")
    console.print(table)

def print_message(message: str, style: str = "bold green"):
    console.print(message, style=style)

def parse_args():
    parser = argparse.ArgumentParser(description="Crypto forecasting")
    parser.add_argument("--model", type=str, choices=["mlp", "linear", "poly"], required=True, help="Modelo a ser usado")
    parser.add_argument("--kfolds", type=int, default=5, help="Número de folds para cross-validation")
    return parser.parse_args()

def main():
    args = parse_args()
    logging.info(f"Executando pipeline para múltiplas moedas usando modelo {args.model} com {args.kfolds} folds.")

    # 1. Defina os arquivos das 10 criptomoedas (ajuste paths conforme sua organização)
    cryptos = {
        "AAVE": "data/Poloniex_AAVEBTC_d.csv",
        "BTC": "data/Poloniex_BTCUSDD_d.csv",
        "ETH": "data/Poloniex_ETHUSDD_d.csv",
        "LTC": "data/Poloniex_LTCUSDD_d.csv",
        "XRP": "data/Poloniex_XRPUSDD_d.csv",
        "BCH": "data/Poloniex_BCHBTC_d.csv",
        "XMR": "data/Poloniex_XMRBTC_d.csv",
        "DASH": "data/Poloniex_DASHBTC_d.csv",
        "ETC": "data/Poloniex_ETCBTC_d.csv",
        "BAT": "data/Poloniex_BATBTC_d.csv"
    }

    # data statistics
    dfs = {}
    stats_dict = {}
    for crypto, filepath in cryptos.items():
        try:
            df = load_crypto_data(filepath, parse_dates=["date"])
            df = build_features(df, price_col="close", date_col="date")
            print_message(f"Features criadas para {crypto}: {list(df.columns)}", style="dim")
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

    if len(dfs) > 1:
        dispersion_df = compare_dispersion(dfs)
        print_dispersion_table(dispersion_df)

        print_message("\n[ANOVA] Comparando os retornos médios diários das criptomoedas...", style="bold yellow")
        returns_dict = {}
        for name, df in dfs.items():
            df = df.copy()
            df["return"] = df["close"].pct_change()
            returns_dict[name] = df["return"].dropna().values

        f_stat, p_value, labels, groups = anova_between_cryptos(returns_dict)
        print_message(f"ANOVA F={f_stat:.4f} | p-value={p_value:.4g}", style="bold cyan")
        if p_value < SIGNIFICANCE_LEVEL:
            print_message("Há diferença significativa entre os retornos médios das moedas. Teste post hoc de Tukey:", style="bold green")
            tukey = tukey_posthoc(returns_dict)
            # Print tabela do Tukey formatada com Rich
            table = Table(title="Tukey HSD Post Hoc", box=box.SIMPLE_HEAVY)
            table.add_column("Grupo 1", style="cyan")
            table.add_column("Grupo 2", style="cyan")
            table.add_column("Diferença", style="magenta")
            table.add_column("p-adj", style="green")
            table.add_column("Inferior", style="yellow")
            table.add_column("Superior", style="yellow")
            table.add_column("Rejeita H0?", style="red")
            for row in tukey.summary().data[1:]:
                table.add_row(*[str(x) for x in row])
            console.print(table)
        else:
            print_message("Não há diferença significativa entre os retornos médios das moedas (p >= 0.05).", style="bold red")

        print_message("\n[ANOVA] Comparando retornos diários por 3 grupos de volume médio negociado...", style="bold yellow")

        # 1. Calcule volume médio negociado de cada moeda
        mean_volume_dict = {}
        for name, df in dfs.items():
            volume_col = next((col for col in df.columns if 'Volume' in col or 'volume' in col), None)
            if not volume_col:
                logging.warning(f"[{name}] Nenhuma coluna de volume encontrada.")
                continue
            mean_volume = df[volume_col].mean()
            mean_volume_dict[name] = mean_volume

        if len(mean_volume_dict) < 3:
            print_message("Menos de três grupos possíveis para ANOVA de volume médio negociado.", style="bold red")
        else:
            import numpy as np
            from scipy.stats import f_oneway
            from statsmodels.stats.multicomp import pairwise_tukeyhsd

            # 2. Separe moedas em 3 grupos por tercis
            vols = np.array(list(mean_volume_dict.values()))
            tercis = np.percentile(vols, [33.33, 66.66])
            grupo_retorno = {"Baixo Volume": [], "Médio Volume": [], "Alto Volume": []}
            labels = {}
            for name, mean_vol in mean_volume_dict.items():
                df = dfs[name]
                df = df.copy()
                df["return"] = df["close"].pct_change()
                returns = df["return"].dropna().values
                if mean_vol < tercis[0]:
                    grupo = "Baixo Volume"
                elif mean_vol < tercis[1]:
                    grupo = "Médio Volume"
                else:
                    grupo = "Alto Volume"
                grupo_retorno[grupo].extend(returns)
                labels[name] = grupo

            # 3. Rode ANOVA entre grupos
            f_stat, p_value = f_oneway(*grupo_retorno.values())
            print_message(
                f"ANOVA (Volume médio - 3 grupos): F={f_stat:.4f} | p-value={p_value:.4g}",
                style="bold cyan"
            )

            if p_value < SIGNIFICANCE_LEVEL:
                print_message("Diferença significativa entre os 3 grupos de volume médio negociado. Executando teste post hoc (Tukey HSD):", style="bold green")
                # Prepare arrays para Tukey
                all_returns = []
                all_labels = []
                for grupo, returns in grupo_retorno.items():
                    all_returns.extend(returns)
                    all_labels.extend([grupo] * len(returns))
                tukey = pairwise_tukeyhsd(endog=all_returns, groups=all_labels, alpha=SIGNIFICANCE_LEVEL)
                # Print tabela do Tukey formatada com Rich
                table = Table(title="Tukey HSD Post Hoc (por Volume)", box=box.SIMPLE_HEAVY)
                table.add_column("Grupo 1", style="cyan")
                table.add_column("Grupo 2", style="cyan")
                table.add_column("Diferença", style="magenta")
                table.add_column("p-adj", style="green")
                table.add_column("Inferior", style="yellow")
                table.add_column("Superior", style="yellow")
                table.add_column("Rejeita H0?", style="red")
                for row in tukey.summary().data[1:]:
                    table.add_row(*[str(x) for x in row])
                console.print(table)
            else:
                print_message("Não há diferença significativa entre os grupos de volume.", style="bold red")

    else:
        logging.warning("Menos de duas criptomoedas processadas — não é possível comparar dispersão.")

    # training
if __name__ == "__main__":
    try:
        setup_logging(LOG_LEVEL)
        main()
    except Exception as e:
        logging.error(f"Erro inesperado na execução: {e}", exc_info=True)
        sys.exit(1)
