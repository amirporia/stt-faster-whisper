import loggingimport logging.configdef setup_logging():    logging.config.dictConfig({        'version': 1,        'disable_existing_loggers': False,        'formatters': {'standard': {'format': '%(asctime)s %(levelname)s [%(name)s]: %(message)s'}},        'handlers': {            'console': {'level': 'INFO', 'class': 'logging.StreamHandler', 'formatter': 'standard'}        },        'root': {'handlers': ['console'], 'level': 'DEBUG'},        'loggers': {            'uvicorn': {'handlers': ['console'], 'level': 'INFO', 'propagate': False},            'whisper_backend': {'handlers': ['console'], 'level': 'DEBUG', 'propagate': False}        }    })setup_logging()logger = logging.getLogger(__name__)