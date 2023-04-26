import logging  # Importing the logging module
import os  # Importing the os module to work with paths
from datetime import datetime  # Importing the datetime module to get the current date and time

# Create a log file with the current timestamp as the filename
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Create the directory to store the log files if it doesn't exist
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(logs_path, exist_ok=True)

# Create the complete path for the log file
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure the logging module
logging.basicConfig(
    filename=LOG_FILE_PATH,  # Set the filename for the log file
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",  # Set the format for the log messages
    level=logging.INFO,  # Set the logging level to INFO
)
