from config import get_logger


logger = get_logger(__name__)


def test_debug_func():
    print('in debug func')
    logger.debug(f'debug message')


def test_info_func():
    print('in info func')
    logger.info(f'info message')


def test_warning_func():
    print('in warning func')
    logger.warning(f'warning message')

def test_function_call_func():
    print('in function_call func')
    logger.function_call(f'function_call message')
