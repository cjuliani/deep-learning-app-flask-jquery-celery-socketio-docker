import os

# MySQL credentials.
USER = 'batman'
PASSWORD = '!abcd1234567'
HOST = os.environ.get("HOST_DB", "localhost")
PORT = 3306
DB_NAME = 'main'

# Directories of training results.
SUMMARY_PATH = 'results/summary'  # folder path of summary results
WEIGHT_PATH = 'results/weights'  # folder path where to save parameters of training models

# Cross-origin resource sharing.
CORS_ADDR = ['http://127.0.0.1:5050',
             'http://127.0.0.1:5051']

# Flask key.
KEYMASTER = 'themasterkey'
