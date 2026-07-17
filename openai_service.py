import base64
import json
import os
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Pricing (USD per 1K tokens). Values are split for `input` and `output` tokens.
# Update values as needed or override INR rate via `INR_PER_USD` env var.
PRICING_USD_PER_1K = {
    "gpt-5-nano": {"input": 0.0002, "output": 0.00125},
    "gpt-4o": {"input": 0.02, "output": 0.03},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
}

INR_PER_USD = float(os.getenv("INR_PER_USD", "96.0"))


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
        # determine model used (response may include model info)
        model_used = getattr(response, "model", "gpt-5-nano")

        # find a pricing entry that matches the model (allow prefix matches)
        pricing_entry = None
        for key, val in PRICING_USD_PER_1K.items():
            if model_used == key or model_used.startswith(key):
                pricing_entry = val
                break
        if pricing_entry is None:
            pricing_entry = PRICING_USD_PER_1K.get("gpt-5-nano")

        usd_input_per_1k = pricing_entry.get("input")
        usd_output_per_1k = pricing_entry.get("output")

        def _price_for_tokens(tokens: int | None, rate_per_1k: float | None) -> float | None:
            if tokens is None or rate_per_1k is None:
                return None
            return (tokens / 1000.0) * rate_per_1k

        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)

        input_price_usd = _price_for_tokens(input_tokens, usd_input_per_1k)
        output_price_usd = _price_for_tokens(output_tokens, usd_output_per_1k)
        # total price is sum of input+output if available, else computed from total_tokens using average rate
        if input_price_usd is not None and output_price_usd is not None:
            total_price_usd = input_price_usd + output_price_usd
        else:
            # fallback: use average of input/output rates to estimate
            avg_rate = None
            if usd_input_per_1k is not None and usd_output_per_1k is not None:
                avg_rate = (usd_input_per_1k + usd_output_per_1k) / 2.0
            total_price_usd = _price_for_tokens(total_tokens, avg_rate)

        usage_payload = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "input_tokens_details": getattr(usage, "input_tokens_details", None),
            "output_tokens_details": getattr(usage, "output_tokens_details", None),
            "pricing": {
                "model": model_used,
                "usd_input_per_1k": usd_input_per_1k,
                "usd_output_per_1k": usd_output_per_1k,
                "inr_input_per_1k": None if usd_input_per_1k is None else round(usd_input_per_1k * INR_PER_USD, 6),
                "inr_output_per_1k": None if usd_output_per_1k is None else round(usd_output_per_1k * INR_PER_USD, 6),
                "input_price_usd": None if input_price_usd is None else round(input_price_usd, 8),
                "output_price_usd": None if output_price_usd is None else round(output_price_usd, 8),
                "total_price_usd": None if total_price_usd is None else round(total_price_usd, 8),
                "input_price_inr": None if input_price_usd is None else round(input_price_usd * INR_PER_USD, 2),
                "output_price_inr": None if output_price_usd is None else round(output_price_usd * INR_PER_USD, 2),
                "total_price_inr": None if total_price_usd is None else round(total_price_usd * INR_PER_USD, 2),
            },
        }

    try:
        parsed = json.loads(raw_output)
        if usage_payload is not None:
            parsed["__usage"] = usage_payload
        return parsed
    except json.JSONDecodeError:
        return {"raw_output": raw_output, "error": "Could not parse JSON", "__usage": usage_payload}
