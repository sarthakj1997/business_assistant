import os
import json
from openai import OpenAI
from typing import List, Optional
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class LineItem(BaseModel):
    description: str
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    line_total: float

class InvoiceSummary(BaseModel):
    vendor_name: str
    vendor_address: Optional[str]
    customer_name: Optional[str]
    customer_address: Optional[str]
    invoice_number: str
    invoice_date: str
    due_date: Optional[str]
    purchase_order: Optional[str]
    currency: str
    items: List[LineItem]
    subtotal: Optional[float]
    tax: Optional[float]
    shipping: Optional[float]
    total: float
    payment_terms: Optional[str]
    payment_method: Optional[str]
    confidence_score: float

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)

SYSTEM_PROMPT = """
You are an invoice parser.
Input: a JSON object with key "pages", whose value is a list of pages:
  {
    "pages": [
      {
        "page": <int>,
        "blocks": [
          {"text": <string>, "bbox": [x0, y0, x1, y1]},
          …
        ]
      },
      …
    ]
  }

Extract exactly these fields and return a single JSON matching InvoiceSummary:

Parties:
• vendor_name (string)  
• vendor_address (string or null)  
• customer_name (string or null)  
• customer_address (string or null)  

Metadata:
• invoice_number (string)  
• invoice_date (string, format DD‑MM‑YYYY)  
• due_date (string, format DD‑MM‑YYYY or null)  
• purchase_order (string or null)  
• currency (string, any ISO 4217 code, e.g. "USD", "INR", "EUR")  

Line items & totals:
• items: list of objects each with  
    – description (string)  
    – quantity (float or null)  
    – unit_price (float or null)  
    – line_total (float)  
• subtotal (float or null) — sum of all line_total before tax  
• tax (float or null)  
• shipping (float or null)  
• total (float) — **take the exact "Total" or "Amount Due" value printed on the invoice (already includes tax), do not add subtotal + tax yourself**  

Payment:
• payment_terms (string or null)  
• payment_method (string or null)  

AI metadata:
• confidence_score (float between 0.0 and 100.0) — measure how well you extracted all fields; err on the low side  

**Requirements:**
- Return **only** the JSON object—no markdown, no commentary.  
- Use `null` for any missing or optional field.  
- Dates must be in `DD-MM-YYYY` format.  
"""

def process_invoice(pages: List[dict]) -> InvoiceSummary:
    completion = client.beta.chat.completions.parse(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({"pages": pages})}
        ],
        response_format=InvoiceSummary,
    )
    return completion.choices[0].message.parsed