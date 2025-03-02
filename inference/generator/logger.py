import logging
import colorlog

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set the minimum logging level

# Create a console handler for colored logging
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

DATE_FORMAT = '%H:%M:%S'

LOG_FORMAT = "%(log_color)s%(asctime)s %(levelname)s - [%(filename)s:%(lineno)d] in %(funcName)s\n%(message)s"

# Define a formatter with colors
color_formatter = colorlog.ColoredFormatter(
    fmt=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'bold_red',
    },
    reset=True,
)
# Apply the formatter to the console handler
console_handler.setFormatter(color_formatter)

# Add the console handler to the logger
logger.addHandler(console_handler)

# Optionally, create a file handler (without colors)
file_handler = logging.FileHandler('output.log')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)
