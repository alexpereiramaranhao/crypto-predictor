import pandas as pd
import pytest

from src.data_load import load_crypto_data


def test_load_crypto_data(tmp_path):
    """
    Testa o carregamento de um CSV de criptomoeda usando a função load_crypto_data.
    """
    # Cria um arquivo CSV temporário de exemplo
    content = """unix,date,symbol,open,high,low,close,Volume AAVE,Volume BTC,buyTakerAmount,buyTakerQuantity,tradeCount,weightedAverage
1751500800000,2025-07-03 00:00:00,AAVE/BTC,0.002621,0.002824,0.002621,0.002824,0.000982,0.35,0.000982,0.35,3,0.002807
1751414400000,2025-07-02 00:00:00,AAVE/BTC,0.002398,0.002398,0.002398,0.002398,0,0,0,0,0,0.002398
"""
    # Cria um arquivo temporário
    temp_file = tmp_path / "crypto.csv"
    temp_file.write_text(content)

    # Testa o carregamento sem skiprows
    df = load_crypto_data(str(temp_file), parse_dates=["date"])
    # Testa se dataframe não está vazio
    assert not df.empty
    # Testa se coluna 'date' foi convertida para datetime
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    # Testa se as colunas corretas estão presentes
    assert "open" in df.columns and "close" in df.columns

    # Testa erro ao passar coluna inválida
    with pytest.raises(ValueError):
        load_crypto_data(str(temp_file), parse_dates=["data_invalida"])


def test_load_crypto_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_crypto_data("data/non_existing_file.csv")


def test_load_crypto_data_empty_file(tmp_path):
    temp_file = tmp_path / "empty.csv"
    temp_file.write_text("")
    with pytest.raises(pd.errors.EmptyDataError):
        load_crypto_data(str(temp_file))
