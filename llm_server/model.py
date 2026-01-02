import logging
import flask


app = flask.Flask(__name__)

class BaseModel:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def request(self, system_msg: str, prompt: str, temperature: float = 0.0) -> str:
        raise NotImplementedError("Subclasses should implement requesting to LLM.")
