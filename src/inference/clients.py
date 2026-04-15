from pydantic import BaseModel
from src.inference.base import generate_inference
from src.prompts import instruction_v6


class CorrectionResponse(BaseModel):
    corrected_text: str


def correct_text_latest(model_id: str, corrupted_text: str) -> str:
    """
    Inference client for the latest GEC model using structured output.

    Args:
        model_id: The ID of the model to use.
        corrupted_text: The text to correct.

    Returns:
        The corrected text.
    """
    messages = [{"role": "user", "content": corrupted_text}]

    result = generate_inference(
        model_id=model_id, messages=messages, response_format=CorrectionResponse
    )

    if isinstance(result, CorrectionResponse):
        return result.corrected_text

    return str(result)


def correct_text_original(model_id: str, corrupted_text: str) -> str:
    """
    Inference client for original models using instruction_v6.

    Args:
        model_id: The ID of the model to use.
        corrupted_text: The text to correct.

    Returns:
        The model's response (expected to be JSON string based on instruction_v6).
    """
    prompt = instruction_v6.format(input_text=corrupted_text)
    messages = [{"role": "user", "content": prompt}]

    result = generate_inference(model_id=model_id, messages=messages)

    return str(result)
