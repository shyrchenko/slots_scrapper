import logging
from .state import is_debug_mode_activated

level = logging.DEBUG if is_debug_mode_activated() else logging.INFO

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('slots_parser')
logger.setLevel(level)
