import pandas as pd
import pytest

from src.features import (
    calcular_media_movel,
    calcular_volatilidade,
    calcular_retorno,
    criar_features_basicas_completas
)


def test_calcular_media_movel():
    """Testa se a média móvel está sendo calculada corretamente."""
    # Dados simples de teste
    precos = [10, 20, 30, 40, 50]
    
    # Calcula média móvel de 3 dias
    media_movel = calcular_media_movel(precos, janela_dias=3)
    
    # Verifica se os primeiros valores são NaN (esperado)
    assert pd.isna(media_movel.iloc[0])
    assert pd.isna(media_movel.iloc[1])
    
    # Verifica se o terceiro valor é a média de [10, 20, 30] = 20
    assert media_movel.iloc[2] == 20.0
    
    # Verifica se o quarto valor é a média de [20, 30, 40] = 30
    assert media_movel.iloc[3] == 30.0


def test_calcular_volatilidade():
    """Testa se a volatilidade está sendo calculada corretamente."""
    # Dados simples de teste (sem variação = volatilidade zero)
    precos_sem_variacao = [10, 10, 10, 10, 10]
    
    volatilidade = calcular_volatilidade(precos_sem_variacao, janela_dias=3)
    
    # Verifica se a volatilidade é zero quando não há variação
    assert volatilidade.iloc[2] == 0.0


def test_calcular_retorno():
    """Testa se os retornos estão sendo calculados corretamente."""
    # Dados simples: preço dobra a cada dia
    precos = [100, 200, 400, 800]
    
    retornos = calcular_retorno(precos)
    
    # Primeiro valor deve ser NaN (não há dia anterior)
    assert pd.isna(retornos.iloc[0])
    
    # Segundo valor deve ser 100% (de 100 para 200)
    assert retornos.iloc[1] == 100.0
    
    # Terceiro valor deve ser 100% (de 200 para 400)
    assert retornos.iloc[2] == 100.0


def test_criar_features_basicas_completas():
    """Testa se todas as features básicas são criadas corretamente."""
    # Cria um DataFrame simples para teste
    dados_teste = {
        'date': pd.date_range('2024-01-01', periods=10, freq='D'),
        'close': [100, 110, 105, 120, 115, 130, 125, 140, 135, 150]
    }
    df_teste = pd.DataFrame(dados_teste)
    
    # Cria as features
    df_com_features = criar_features_basicas_completas(df_teste)
    
    # Verifica se as novas colunas foram criadas
    assert 'media_movel_7d' in df_com_features.columns
    assert 'volatilidade_7d' in df_com_features.columns
    assert 'retorno_diario' in df_com_features.columns
    assert 'preco_subiu' in df_com_features.columns
    
    # Verifica se o DataFrame não foi modificado no tamanho
    assert len(df_com_features) == len(df_teste)
    
    # Verifica se a coluna 'preco_subiu' tem apenas valores 0 e 1
    valores_unicos = df_com_features['preco_subiu'].dropna().unique()
    for valor in valores_unicos:
        assert valor in [0, 1]


def test_media_movel_com_dados_vazios():
    """Testa o comportamento com dados vazios."""
    precos_vazios = []
    
    # Deve retornar uma Series vazia
    resultado = calcular_media_movel(precos_vazios, janela_dias=5)
    assert len(resultado) == 0


def test_retorno_com_um_valor():
    """Testa o comportamento com apenas um valor."""
    preco_unico = [100]
    
    retornos = calcular_retorno(preco_unico)
    
    # Deve retornar apenas um valor NaN
    assert len(retornos) == 1
    assert pd.isna(retornos.iloc[0])