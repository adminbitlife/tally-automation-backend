import base64
import json
import os
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def _get_client() -> OpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def extract_invoice_data(image_path: str) -> Dict[str, Any]:
    prompt = """
You are an invoice extraction assistant.

Extract all invoice information from the provided invoice image.
Return ONLY valid JSON.

JSON Schema:
{
  "invoice_number": "",
  "invoice_date": "",
  "due_date": "",
  "currency": "",
  "seller": {
      "store_name": "",
      "owner_name": "",
      "email": "",
      "phone": "",
      "website": "",
      "address": ""
  },
  "buyer": {
      "name": "",
      "email": "",
      "phone": "",
      "address": ""
  },
  "shipping": {
      "shipping_address": "",
      "tracking_number": ""
  },
  "payment": {
      "payment_instruction": "",
      "amount_paid": 0,
      "balance_due": 0
  },
  "items": [
      {
          "description": "",
          "rate": 0,
          "quantity": 0,
          "tax_percent": 0,
          "discount_percent": 0,
          "amount": 0
      }
  ],
  "totals": {
      "subtotal": 0,
      "discount": 0,
      "shipping_cost": 0,
      "sales_tax": 0,
      "total": 0,
      "amount_paid": 0,
      "balance_due": 0
  }
}
"""

    client = _get_client()
    if client is None:
        return {"error": "OPENAI_API_KEY is not set"}

    try:
        base64_image = _encode_image(image_path)
        response = client.responses.create(
            model="gpt-5-nano",
            reasoning={"effort": "minimal"},
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    ],
                }
            ],
        )
    except Exception as exc:
        return {"error": str(exc)}

    raw_output = response.output_text.strip()
    usage = getattr(response, "usage", None)
    usage_payload = None
    if usage is not None:
        usage_payload = {
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
            "input_tokens_details": getattr(usage, "input_tokens_details", None),
            "output_tokens_details": getattr(usage, "output_tokens_details", None),
        }

    try:
        parsed = json.loads(raw_output)
        if usage_payload is not None:
            parsed["__usage"] = usage_payload
        return parsed
    except json.JSONDecodeError:
        return {"raw_output": raw_output, "error": "Could not parse JSON", "__usage": usage_payload}
