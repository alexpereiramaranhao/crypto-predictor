"""
main.py
Script principal para execução via CLI.
Executa todo o pipeline de análise e previsão de criptomoedas.
"""

import argparse
import logging
import sys
from typing import Tuple, Union, Any, List, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from rich import box
from rich.console import Console
from rich.table import Table

from src.data_load import load_crypto_data
from src.features import criar_features_basicas_completas
from src.models import train_mlp, train_linear, encontrar_melhor_grau_polinomial, validacao_cruzada_kfold
from src.lucro import calcular_lucro_investimento, calcular_estrategia_buy_and_hold
from src.analise_lucro import imprimir_metricas_modelo, comparar_todos_modelos, mostrar_equacao_linear
from src.statistics.analysis import compare_dispersion, summary_statistics, teste_hipotese_retorno, anova_entre_criptos, anova_grupos_caracteristicas
from src.statistics.plots import plot_boxplot, plot_histogram, plot_price_with_summary, plotar_evolucao_lucro, plotar_dispersao_modelos
from src.util.config import LOG_LEVEL
from src.util.utils import setup_logging

console = Console()


def print_dispersion_table(dispersion_table: pd.DataFrame) -> None:
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


def print_resumo_teste_hipotese(resultados_teste: List[Dict]) -> None:
    """
    Mostra tabela resumo dos testes de hipótese para todas as criptomoedas.
    
    Args:
        resultados_teste: Lista com resultados do teste para cada crypto
    """
    if not resultados_teste:
        return
        
    print_message("\n📊 RESUMO DOS TESTES DE HIPÓTESE", style="bold magenta")
    
    table = Table(title="Teste de Hipótese - Retorno Esperado", box=box.SIMPLE_HEAVY)
    table.add_column("Criptomoeda", style="cyan", no_wrap=True)
    table.add_column("Retorno Médio (%)", style="white")
    table.add_column("P-valor", style="white")
    table.add_column("Rejeita H0?", style="white")
    table.add_column("Conclusão", style="white")
    
    for resultado in resultados_teste:
        crypto = resultado['crypto']
        retorno_medio = resultado['retorno_medio']
        p_valor = resultado['p_valor']
        rejeita_h0 = resultado['rejeita_h0']
        percentual_esperado = resultado['percentual_esperado']
        
        # Formatação da conclusão
        if rejeita_h0:
            conclusao = f"Retorno > {percentual_esperado}%"
            conclusao_style = "✓"
        else:
            conclusao = f"Retorno ≤ {percentual_esperado}%"
            conclusao_style = "✗"
        
        table.add_row(
            crypto,
            f"{retorno_medio:.4f}",
            f"{p_valor:.6f}",
            conclusao_style,
            conclusao
        )
    
    console.print(table)
    
    # Estatísticas gerais
    total_cryptos = len(resultados_teste)
    rejeitaram_h0 = sum(1 for r in resultados_teste if r['rejeita_h0'])
    
    print(f"\nResumo: {rejeitaram_h0}/{total_cryptos} criptomoedas rejeitaram H0 (α = 5%)")


def print_profit_results(crypto: str, profit_model: float, profit_buyhold: float) -> None:
    table = Table(title=f"Resultados de Lucro - {crypto}", box=box.SIMPLE_HEAVY)
    table.add_column("Estratégia", style="cyan", no_wrap=True)
    table.add_column("Lucro (R$)", style="magenta")
    table.add_column("Melhor?", style="green")
    
    is_model_better = profit_model > profit_buyhold
    
    table.add_row("Modelo ML", f"{profit_model:.2f}", "✓" if is_model_better else "")
    table.add_row("Buy & Hold", f"{profit_buyhold:.2f}", "✓" if not is_model_better else "")
    
    console.print(table)


def treinar_modelo_escolhido(model_type: str, X_train: np.ndarray, y_train: np.ndarray) -> Union[Any, Tuple[Any, Any]]:
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


def preparar_features_para_modelo(df_with_features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
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


def fazer_analise_completa_lucro(X: np.ndarray, y: np.ndarray, crypto: str) -> None:
    """
    Faz análise completa de lucro comparando MLP, Linear e melhor Polinomial.
    
    Args:
        X: Features preparadas
        y: Target (preços)
        crypto: Nome da criptomoeda
    """
    print_message(f"\n🔍 Análise Completa de Lucro - {crypto}", style="bold magenta")
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    # Normalizar features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Treinar MLP
    print_message("🤖 Treinando MLP...", style="cyan")
    modelo_mlp = train_mlp(X_train_scaled, y_train)
    previsoes_mlp = modelo_mlp.predict(X_test_scaled)
    
    # 2. Treinar Linear
    print_message("📈 Treinando Regressão Linear...", style="cyan")
    modelo_linear = train_linear(X_train_scaled, y_train)
    previsoes_linear = modelo_linear.predict(X_test_scaled)
    
    # 3. Encontrar melhor grau polinomial
    print_message("🔢 Encontrando melhor grau polinomial (2-10)...", style="cyan")
    melhor_grau, _, modelo_poly, transformador_poly = encontrar_melhor_grau_polinomial(X_train_scaled, y_train)
    
    # Fazer previsões com melhor polinomial
    X_test_poly = transformador_poly.transform(X_test_scaled)
    previsoes_poly = modelo_poly.predict(X_test_poly)
    
    # === ANÁLISES ESTATÍSTICAS (requisitos b, c, d) ===
    
    resultados_modelos = {}
    
    # MLP
    metricas_mlp = imprimir_metricas_modelo("MLP", y_test, previsoes_mlp)
    resultados_modelos["MLP"] = metricas_mlp
    
    # Linear
    metricas_linear = imprimir_metricas_modelo("Linear", y_test, previsoes_linear)
    resultados_modelos["Linear"] = metricas_linear
    print(f"Equação Linear: {mostrar_equacao_linear(modelo_linear)}")
    
    # Polinomial
    nome_poly = f"Polinomial Grau {melhor_grau}"
    metricas_poly = imprimir_metricas_modelo(nome_poly, y_test, previsoes_poly)
    resultados_modelos[nome_poly] = metricas_poly
    print(f"Equação Polinomial: {mostrar_equacao_linear(modelo_poly)} (com features transformadas)")
    
    # Comparação final (requisito e)
    comparar_todos_modelos(resultados_modelos)
    
    # === CÁLCULO DE LUCROS ===
    
    print_message("\n💰 Calculando lucros...", style="green")
    
    # Lucro MLP
    _, lucro_mlp, historico_mlp = calcular_lucro_investimento(y_test, previsoes_mlp)
    
    # Lucro Linear
    _, lucro_linear, historico_linear = calcular_lucro_investimento(y_test, previsoes_linear)
    
    # Lucro Polinomial
    _, lucro_poly, historico_poly = calcular_lucro_investimento(y_test, previsoes_poly)
    
    # Lucro Buy-and-Hold
    lucro_buyhold = calcular_estrategia_buy_and_hold(y_test)
    
    # Mostrar resultados de lucro
    print(f"\nRESULTADOS DE LUCRO ({crypto}):")
    print(f"MLP: R$ {lucro_mlp:.2f}")
    print(f"Linear: R$ {lucro_linear:.2f}")
    print(f"{nome_poly}: R$ {lucro_poly:.2f}")
    print(f"Buy-and-Hold: R$ {lucro_buyhold:.2f}")
    
    # === GRÁFICOS (requisitos a e f) ===
    
    print_message("📊 Gerando gráficos...", style="blue")
    
    # a) Diagrama de dispersão (comparando MLP e melhor polinomial)
    plotar_dispersao_modelos(
        y_test, previsoes_mlp, previsoes_poly,
        nome_modelo1="MLP", nome_modelo2=nome_poly
    )
    
    # f) Evolução do lucro (comparando MLP e melhor polinomial)
    plotar_evolucao_lucro(
        historico_mlp, historico_poly,
        nome_modelo1="MLP", nome_modelo2=nome_poly
    )
    
    print_message(f"✅ Análise completa de {crypto} finalizada!", style="bold green")


def parse_args() -> argparse.Namespace:
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
    parser.add_argument(
        "--teste-retorno", 
        type=float, 
        default=None, 
        help="Percentual de retorno esperado para teste de hipótese (ex: 5.0 para 5 porcento)"
    )
    return parser.parse_args()


def main() -> None:
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
    resultados_teste_hipotese = []  # Para guardar resultados do teste de hipótese
    
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

            # === PIPELINE DE MACHINE LEARNING ===
            
            # 2. Criar features
            print_message("🔧 Criando features...", style="cyan")
            df_with_features = criar_features_basicas_completas(df)
            
            # 3. Preparar dados para modelo
            X, y, df_clean = preparar_features_para_modelo(df_with_features)
            
            if len(X) < 20:  # Precisa de dados suficientes
                print_message(f"❌ {crypto}: Poucos dados para treinamento", style="red")
                continue
            
            # === TESTE DE HIPÓTESE (se solicitado pelo usuário) ===
            if args.teste_retorno is not None:
                print_message(f"🧪 Fazendo teste de hipótese para retorno ≥ {args.teste_retorno}%", style="yellow")
                resultado_teste = teste_hipotese_retorno(
                    df_with_features['retorno_diario'], 
                    percentual_esperado=args.teste_retorno,
                    nivel_significancia=0.05
                )
                # Adiciona nome da crypto ao resultado
                resultado_teste['crypto'] = crypto
                resultados_teste_hipotese.append(resultado_teste)
            
            # === ANÁLISE COMPLETA DE LUCRO (Requisito 9) ===
            # Faz análise completa comparando MLP vs Linear vs melhor Polinomial
            fazer_analise_completa_lucro(X, y, crypto)
            
            # === PIPELINE ORIGINAL (para compatibilidade) ===
            
            # 4. Dividir dados treino/teste
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # 5. Normalizar features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 6. Treinar modelo escolhido pelo usuário
            print_message(f"🤖 Treinando modelo escolhido pelo usuário: {args.model}...", style="cyan")
            
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
            
            # 8. Calcular lucros do modelo escolhido
            print_message("💰 Calculando lucros do modelo escolhido...", style="green")
            
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
    
    # Resumo dos testes de hipótese (se foram executados)
    if resultados_teste_hipotese:
        print_resumo_teste_hipotese(resultados_teste_hipotese)
    
    # c) Comparação de dispersão entre criptomoedas
    if len(dfs) > 1:
        dispersion_df = compare_dispersion(dfs)
        print_dispersion_table(dispersion_df)
    
    # === ANÁLISES DE VARIÂNCIA (ANOVA) ===
    print_message("\n📊 ANÁLISES DE VARIÂNCIA (ANOVA)", style="bold blue")
    
    # Preparar dados com features para ANOVA (necessário para ter retorno_diario e volatilidade_7d)
    dfs_com_features = {}
    for crypto, df in dfs.items():
        try:
            df_features = criar_features_basicas_completas(df)
            if 'retorno_diario' in df_features.columns and 'volatilidade_7d' in df_features.columns:
                dfs_com_features[crypto] = df_features
        except Exception as e:
            print_message(f"Erro ao criar features para {crypto}: {e}", style="red")
    
    if len(dfs_com_features) >= 2:
        # A) ANOVA entre criptomoedas (Requisito 11a)
        print_message("\n🔍 A) ANOVA entre criptomoedas", style="cyan")
        resultado_anova_criptos = anova_entre_criptos(dfs_com_features)
        
        if len(dfs_com_features) >= 3:
            # B) ANOVA entre grupos de volatilidade (Requisito 11b)
            print_message("\n🔍 B) ANOVA entre grupos de volatilidade", style="cyan")
            resultado_anova_grupos = anova_grupos_caracteristicas(dfs_com_features)
        else:
            print_message("⚠️ Precisa de pelo menos 3 criptomoedas para ANOVA de grupos", style="yellow")
    else:
        print_message("⚠️ Dados insuficientes para análises de ANOVA", style="yellow")
    
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
