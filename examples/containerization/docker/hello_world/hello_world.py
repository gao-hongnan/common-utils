import logging

from rich.console import Console
from rich.logging import RichHandler

console = Console(force_terminal=True)

# Setup logging
logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)

logger = logging.getLogger("rich")

logger.info("Hello world!")
