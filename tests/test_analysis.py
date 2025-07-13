import pandas as pd
import pytest

from src.statistics.analysis import summary_statistics


def test_summary_statistics_basic():
    data = {"close": [1.0, 2.0, 3.0, 4.0, 5.0]}
    df = pd.DataFrame(data)
    stats = summary_statistics(df, price_col="close")
    assert stats["mean"] == pytest.approx(3.0)
    assert stats["median"] == pytest.approx(3.0)
    assert stats["min"] == pytest.approx(1.0)
    assert stats["max"] == pytest.approx(5.0)
    assert stats["amplitude"] == pytest.approx(4.0)
    assert stats["iqr"] == pytest.approx(2.0)
