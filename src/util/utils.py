import logging
from rich.logging import RichHandler

def setup_logging(level: str = "INFO"):
    """
    Configura logging centralizado e colorido usando Rich para todo o projeto.
    Args:
        level (str): Nível do log (ex: "INFO", "DEBUG", "WARNING", "ERROR")
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="[%H:%M:%S]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                show_time=True,
                omit_repeated_times=True,
                show_path=True
            )
        ]
    )

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
