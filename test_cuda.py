import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"Torch version: {torch.__version__}")
logger.info(f"CUDA version: {torch.version.cuda}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"Device name: {torch.cuda.get_device_name(0)}")
