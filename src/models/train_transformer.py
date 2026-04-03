from __future__ import annotations

"""Plantilla inicial para fine-tuning.

Siguiente paso sugerido:
1. Usar un dataset procesado con columnas `text_clean` y `label`.
2. Mapear etiquetas a ids.
3. Probar primero un modelo en español, por ejemplo BETO.
4. Exportar métricas a outputs/metrics.
"""

from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    logger.info("Plantilla lista para fine-tuning del transformer.")
    logger.info("Aquí se conectará el pipeline con Hugging Face Transformers.")


if __name__ == "__main__":
    main()
