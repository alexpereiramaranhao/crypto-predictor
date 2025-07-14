import numpy as np
import pandas as pd


def calcular_lucro_investimento(precos_reais, previsoes_modelo, investimento_inicial=1000.0):
    """
    Calcula o lucro obtido seguindo as previsões de um modelo.
    
    Estratégia: Se o modelo prevê que o preço vai subir amanhã, 
    investe todo o dinheiro hoje. Senão, não investe.
    
    Args:
        precos_reais: Lista com os preços reais da criptomoeda
        previsoes_modelo: Lista com as previsões do modelo
        investimento_inicial: Quanto dinheiro começamos (padrão: R$ 1000)
    
    Returns:
        dinheiro_final: Quanto dinheiro temos no final
        lucro_total: Quanto ganhamos (pode ser negativo se perdemos)
        historico_dinheiro: Lista mostrando o dinheiro a cada dia
    """
    print(f"Calculando lucro com investimento inicial de R$ {investimento_inicial:.2f}")
    
    # Converte para listas se necessário
    if isinstance(precos_reais, pd.Series):
        precos_reais = precos_reais.tolist()
    if isinstance(previsoes_modelo, pd.Series):
        previsoes_modelo = previsoes_modelo.tolist()
    
    # Verifica se os tamanhos são compatíveis
    if len(precos_reais) != len(previsoes_modelo):
        print("ERRO: Tamanhos diferentes entre preços reais e previsões!")
        return None, None, None
    
    # Variáveis para controle
    dinheiro_atual = investimento_inicial
    historico_dinheiro = [dinheiro_atual]  # Guarda o dinheiro a cada dia
    
    # Para cada dia (exceto o último, pois não há previsão para o dia seguinte)
    for dia in range(len(precos_reais) - 1):
        preco_hoje = precos_reais[dia]
        preco_amanha_real = precos_reais[dia + 1]
        previsao_amanha = previsoes_modelo[dia + 1]
        
        # Decisão: investir ou não?
        if previsao_amanha > preco_hoje:
            # Modelo prevê que vai subir - INVESTE tudo!
            quantidade_moedas = dinheiro_atual / preco_hoje
            dinheiro_amanha = quantidade_moedas * preco_amanha_real
            
            print(f"Dia {dia+1}: Preço R${preco_hoje:.2f} -> Previsão R${previsao_amanha:.2f} -> INVESTE!")
            print(f"  Comprou {quantidade_moedas:.6f} moedas, amanhã valem R${dinheiro_amanha:.2f}")
        else:
            # Modelo prevê que vai cair - NÃO INVESTE (mantém o dinheiro)
            dinheiro_amanha = dinheiro_atual
            print(f"Dia {dia+1}: Preço R${preco_hoje:.2f} -> Previsão R${previsao_amanha:.2f} -> NÃO INVESTE")
        
        dinheiro_atual = dinheiro_amanha
        historico_dinheiro.append(dinheiro_atual)
    
    # Calcula o lucro final
    dinheiro_final = dinheiro_atual
    lucro_total = dinheiro_final - investimento_inicial
    percentual_lucro = (lucro_total / investimento_inicial) * 100
    
    print(f"\n=== RESULTADO FINAL ===")
    print(f"Investimento inicial: R$ {investimento_inicial:.2f}")
    print(f"Dinheiro final: R$ {dinheiro_final:.2f}")
    print(f"Lucro total: R$ {lucro_total:.2f} ({percentual_lucro:.2f}%)")
    
    return dinheiro_final, lucro_total, historico_dinheiro


def comparar_lucro_entre_modelos(precos_reais, previsoes_modelo1, previsoes_modelo2, 
                                nome_modelo1="Modelo 1", nome_modelo2="Modelo 2"):
    """
    Compara o lucro entre dois modelos diferentes.
    
    Args:
        precos_reais: Preços reais da criptomoeda
        previsoes_modelo1: Previsões do primeiro modelo
        previsoes_modelo2: Previsões do segundo modelo
        nome_modelo1: Nome do primeiro modelo
        nome_modelo2: Nome do segundo modelo
    
    Returns:
        resultados: Dicionário com os resultados de cada modelo
    """
    print(f"Comparando lucro entre {nome_modelo1} e {nome_modelo2}...")
    
    # Calcula lucro do modelo 1
    dinheiro1, lucro1, historico1 = calcular_lucro_investimento(
        precos_reais, previsoes_modelo1
    )
    
    # Calcula lucro do modelo 2  
    dinheiro2, lucro2, historico2 = calcular_lucro_investimento(
        precos_reais, previsoes_modelo2
    )
    
    # Determina qual foi melhor
    if lucro1 > lucro2:
        melhor_modelo = nome_modelo1
        diferenca = lucro1 - lucro2
    else:
        melhor_modelo = nome_modelo2
        diferenca = lucro2 - lucro1
    
    print(f"\n=== COMPARAÇÃO DOS MODELOS ===")
    print(f"{nome_modelo1}: R$ {lucro1:.2f} de lucro")
    print(f"{nome_modelo2}: R$ {lucro2:.2f} de lucro")
    print(f"Melhor modelo: {melhor_modelo} (R$ {diferenca:.2f} a mais)")
    
    resultados = {
        nome_modelo1: {"dinheiro_final": dinheiro1, "lucro": lucro1, "historico": historico1},
        nome_modelo2: {"dinheiro_final": dinheiro2, "lucro": lucro2, "historico": historico2},
        "melhor_modelo": melhor_modelo,
        "diferenca_lucro": diferenca
    }
    
    return resultados


def calcular_estrategia_buy_and_hold(precos_reais, investimento_inicial=1000.0):
    """
    Calcula o lucro da estratégia "comprar e segurar" (buy and hold).
    Compra no primeiro dia e vende no último dia.
    
    Args:
        precos_reais: Lista com os preços reais
        investimento_inicial: Quanto investir no início
    
    Returns:
        lucro_buy_hold: Lucro da estratégia buy and hold
    """
    preco_inicial = precos_reais[0]
    preco_final = precos_reais[-1]
    
    # Quantidade de moedas que conseguimos comprar no primeiro dia
    quantidade_moedas = investimento_inicial / preco_inicial
    
    # Valor final das moedas
    valor_final = quantidade_moedas * preco_final
    
    # Lucro
    lucro_buy_hold = valor_final - investimento_inicial
    
    print(f"Estratégia Buy and Hold:")
    print(f"  Comprou no preço: R$ {preco_inicial:.2f}")
    print(f"  Vendeu no preço: R$ {preco_final:.2f}")
    print(f"  Lucro: R$ {lucro_buy_hold:.2f}")
    
    return lucro_buy_hold