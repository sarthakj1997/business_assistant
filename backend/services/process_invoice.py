import os
import json
from openai import OpenAI
from typing import List, Optional
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from typing import List, Optional
from pydantic import BaseModel, Field

class LineItem(BaseModel):
    product_id: Optional[str] = Field(
        None, description="SKU or product identifier"
    )
    product_name: str = Field(
        ..., description="Name or description of the product"
    )
    quantity: Optional[int] = Field(
        None, description="Number of units"
    )
    unit_price: Optional[float] = Field(
        None, description="Price per single unit"
    )
    line_total: Optional[float] = Field(
        None, description="Total for this line (unit_price * quantity)"
    )
    confidence_score: Optional[float] = Field(
        None, description="Extraction confidence 0–1"
    )

class InvoiceSummary(BaseModel):
    # Invoice / order identifiers
    order_id: str = Field(..., description="Order or invoice number")
    customer_id: Optional[str] = Field(
        None, description="Customer ID from the document"
    )
    invoice_date: str = Field(
        ..., description="Order or invoice date (ISO 8601 YYYY‑MM‑DD)"
    )

    # Customer contact & address
    contact_name: Optional[str] = Field(
        None, description="Customer contact person"
    )
    address: Optional[str] = Field(
        None, description="Street address"
    )
    city: Optional[str] = Field(
        None, description="City"
    )
    postal_code: Optional[str] = Field(
        None, description="Postal or ZIP code"
    )
    country: Optional[str] = Field(
        None, description="Country"
    )
    customer_phone: Optional[str] = Field(
        None, description="Customer phone number"
    )
    customer_fax: Optional[str] = Field(
        None, description="Customer fax number"
    )

    # Line items & totals
    items: List[LineItem] = Field(
        ..., description="List of purchased products or services"
    )
    total_price: float = Field(
        ..., description="Grand total at bottom of invoice"
    )
    confidence_score: float = Field(
        ..., description="Overall extraction confidence 0–1"
    )


client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)

SYSTEM_PROMPT = """
You are an expert invoice‑parsing assistant. You will receive a single JSON object with one key, "pages", whose value is an array of page objects containing all raw text and layout data from an invoice. Your job is to output a single JSON object that exactly matches the following Pydantic schema. Do not output any extra keys, comments, or explanations—only the JSON object.

Pydantic schema to match:

InvoiceSummary {
  order_id: string
  customer_id: string | null
  invoice_date: string            // ISO 8601 date, e.g. "2025-07-30"
  contact_name: string | null
  address: string | null
  city: string | null
  postal_code: string | null
  country: string | null
  customer_phone: string | null
  customer_fax: string | null
  items: [
    {
      product_id: string | null
      product_name: string
      quantity: integer | null
      unit_price: number | null
      line_total: number | null    // unit_price * quantity
      confidence_score: number | null  // 0.0–1.0
    },
    …
  ]
  total_price: number
  confidence_score: number         // 0.0–1.0 overall
}

Rules:
1. Output strictly valid JSON with exactly these keys, in this order.
2. Use null for any optional field you cannot find.
3. Dates must be ISO 8601 format (YYYY‑MM‑DD).
4. Quantities must be integers; unit_price, line_total, total_price, and confidence_score must be numbers.
5. Calculate line_total as unit_price * quantity when both values are available.
6. Infer line items by grouping contiguous rows of product identifier/name/quantity/price data.
7. For each line item, assign a confidence_score between 0.0 and 1.0 reflecting your extraction certainty.
8. Assign an overall confidence_score (0.0–1.0) for the entire invoice extraction.
9. The input "pages" array may have varied layout; locate invoice metadata (order_id, invoice_date, customer_id), customer contact/address block, the itemized products table, and the final total, then map each to the schema above.

Return only the JSON object representing the InvoiceSummary.
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