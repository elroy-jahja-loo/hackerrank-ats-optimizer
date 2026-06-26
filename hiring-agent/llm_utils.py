"""
Utility functions for LLM providers.
"""

import logging
from typing import Any, Optional
from models import ModelProvider, OllamaProvider, GeminiProvider, OpenAIProvider, AnthropicProvider
from prompt import MODEL_PROVIDER_MAPPING, GEMINI_API_KEY, OPENAI_API_KEY, OPENAI_BASE_URL, ANTHROPIC_API_KEY

logger = logging.getLogger(__name__)


def extract_json_from_response(response_text: str) -> str:
    """
    Extract JSON content from markdown code blocks.

    Args:
        response_text: Text that may contain JSON wrapped in markdown code blocks

    Returns:
        Text with markdown code block syntax removed
    """

    response_text = response_text.strip()
    if "<think>" in response_text:
        think_start = response_text.find("<think>")
        think_end = response_text.find("</think>")
        if think_start != -1 and think_end != -1:
            response_text = response_text[:think_start] + response_text[think_end + 8 :]

    # Remove leading ```json if present
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    # Remove trailing ``` if present
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    return response_text


def initialize_llm_provider(
    model_name: str,
    provider_name: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Any:
    """
    Initialize the appropriate LLM provider based on the model name.

    Args:
        model_name: The name of the model to use

    Returns:
        An initialized LLM provider (either OllamaProvider or GeminiProvider)
    """
    if provider_name:
        try:
            model_provider = ModelProvider(provider_name)
        except ValueError as exc:
            raise ValueError(f"Unsupported LLM provider: {provider_name}") from exc
    else:
        model_provider = MODEL_PROVIDER_MAPPING.get(model_name, ModelProvider.OLLAMA)

    if model_provider == ModelProvider.ANTHROPIC:
        provider_api_key = api_key or ANTHROPIC_API_KEY
        if not provider_api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set in .env")
        logger.info(f"🔄 Using Anthropic Claude provider with model {model_name}")
        return AnthropicProvider(api_key=provider_api_key)

    if model_provider == ModelProvider.OPENAI:
        provider_api_key = api_key or OPENAI_API_KEY
        provider_base_url = base_url if base_url is not None else OPENAI_BASE_URL
        if not provider_api_key:
            raise ValueError("OPENAI_API_KEY is not set in .env")
        logger.info(f"🔄 Using OpenAI provider with model {model_name}")
        return OpenAIProvider(api_key=provider_api_key, base_url=provider_base_url or None)

    if model_provider == ModelProvider.GEMINI:
        provider_api_key = api_key or GEMINI_API_KEY
        if not provider_api_key:
            raise ValueError("GEMINI_API_KEY is not set in .env")
        logger.info(f"🔄 Using Google Gemini API provider with model {model_name}")
        return GeminiProvider(api_key=provider_api_key)

    logger.info(f"🔄 Using Ollama provider with model {model_name}")
    return OllamaProvider(host=base_url or None)
