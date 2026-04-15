from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel
from ollama import chat, ChatResponse


def generate_inference(
    model_id: str,
    messages: List[Dict[str, str]],
    response_format: Optional[Type[BaseModel]] = None,
) -> Any:
    """
    Base function to generate inference using Ollama.

    Args:
        model_id: The ID of the model to use.
        messages: A list of message dictionaries (e.g., [{"role": "user", "content": "..."}]).
        response_format: Optional Pydantic BaseModel class for structured output.

    Returns:
        The parsed Pydantic model if response_format is provided, otherwise the raw string content.
    """
    kwargs = {
        "model": model_id,
        "messages": messages,
    }

    if response_format:
        kwargs["format"] = response_format.model_json_schema()

    try:
        response: ChatResponse = chat(**kwargs)

        content = response.message.content or ""

        if response_format:
            return response_format.model_validate_json(content)

        return content
    except Exception as e:
        return f"ERROR: {str(e)}"
