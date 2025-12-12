import logging


class MyLogger:
    def __init__(self, accelerator=None):
        self.accelerator = accelerator
        self.logger = logging.getLogger(__name__)
        if accelerator.is_main_process:
            logging.basicConfig(level=logging.INFO)
        else:
            # Reduce logging level on non-main ranks
            logging.basicConfig(level=logging.ERROR)
    
    def info(self, message):
        if self.accelerator.is_main_process:
            self.logger.info(message)