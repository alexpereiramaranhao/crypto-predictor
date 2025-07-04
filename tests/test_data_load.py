import pytest

from src.data_load import load_crypto_data


def test_load_crypto_data():
    with pytest.raises(FileNotFoundError):
        load_crypto_data("not-found.csv")
