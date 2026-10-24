"""OpenAI chat model factory."""

from langchain_openai import ChatOpenAI

import config


def get_llm(model: str | None = None) -> ChatOpenAI:
    kwargs = {"model": model or config.LLM_MODEL, "timeout": 60, "max_retries": 2}
    if config.LLM_TEMPERATURE is not None:
        kwargs["temperature"] = config.LLM_TEMPERATURE
    return ChatOpenAI(**kwargs)