from openai import OpenAI
from openai import AzureOpenAI
from functools import wraps

import logging

logger = logging.getLogger(__name__)

def except_retry_dec(retry_num: int = 3):
    def decorator(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            i = 0
            while True:
                try:
                    logger.info("openai agent post...")
                    ret = func(*args, **kwargs)
                    logger.info("openai agent post finished")
                    return ret
                # error define: https://platform.openai.com/docs/guides/error-codes/python-library-error-types
                except (
                    openai.BadRequestError,
                    openai.AuthenticationError,
                ) as e:
                    raise
                except Exception as e:  # pylint: disable=W0703
                    logger.error(f"{e}")
                    logger.info(f"sleep {i + 1}")
                    time.sleep(i + 1)
                    if i >= retry_num:
                        raise
                    logger.warning(f"do retry, time: {i}")
                    i += 1

        return wrapped_func

    return decorator

class Agent:
    def __init__(
        self, model, source, api_dict, temperature: float = 0.0):
        self.model = model
        self.temperature = temperature

        if source == "azure":
            self.client = AzureOpenAI(
                azure_endpoint = api_dict["endpoint"], 
                api_version=api_dict["api_version"],
                api_key=api_dict["api_key"],
                )
            
        elif source == "openai":
            self.client = OpenAI(
                    # This is the default and can be omitted
                    api_key=api_dict["api_key"],
                )
        elif source == "deepseek":
            self.client = OpenAI(
                    # This is the default and can be omitted
                    base_url=api_dict["base_url"],
                    api_key=api_dict["api_key"],
                )
        print(f"You are using {self.model} from {source}")
        
    @except_retry_dec()
    def generate(self, prompt: str, max_new_tokens:int=None) -> str:
        _completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=self.temperature,
                model=self.model,
            )
        return [_completion.choices[0].message.content]
