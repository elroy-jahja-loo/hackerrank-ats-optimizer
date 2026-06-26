"""
Utility functions for LLM providers.
"""

import logging
from typing import Any, Dict, Optional
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


def initialize_llm_provider(model_name: str) -> Any:
    """
    Initialize the appropriate LLM provider based on the model name.

    Args:
        model_name: The name of the model to use

    Returns:
        An initialized LLM provider (either OllamaProvider or GeminiProvider)
    """
    model_provider = MODEL_PROVIDER_MAPPING.get(model_name, ModelProvider.OLLAMA)

    if model_provider == ModelProvider.ANTHROPIC:
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is not set in .env")
        logger.info(f"🔄 Using Anthropic Claude provider with model {model_name}")
        return AnthropicProvider(api_key=ANTHROPIC_API_KEY)

    if model_provider == ModelProvider.OPENAI:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in .env")
        logger.info(f"🔄 Using OpenAI provider with model {model_name}")
        return OpenAIProvider(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL or None)

    if model_provider == ModelProvider.GEMINI:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set in .env")
        logger.info(f"🔄 Using Google Gemini API provider with model {model_name}")
        return GeminiProvider(api_key=GEMINI_API_KEY)

    logger.info(f"🔄 Using Ollama provider with model {model_name}")
    return OllamaProvider()
