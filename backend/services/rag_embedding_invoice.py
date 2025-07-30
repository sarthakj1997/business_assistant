import os
from pinecone import Pinecone, ServerlessSpec

from typing import List, Dict

from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


INDEX_NAME = "invoice-rag"
index = pc.Index(INDEX_NAME)

def get_embedding(text: str) -> List[float]:
    response = pc.inference.embed(
        model="llama-text-embed-v2",
        inputs=[text],
        parameters={"input_type": "passage"}
    )
    return response[0]['values']



def store_invoice_vectors(invoice_data, invoice_id: int, user_id: int):
    # Store comprehensive invoice summary
    invoice_text = f"""
    Order ID: {invoice_data.order_id}
    Customer ID: {invoice_data.customer_id or 'N/A'}
    Contact Name: {invoice_data.contact_name or 'N/A'}
    Address: {invoice_data.address or 'N/A'}
    City: {invoice_data.city or 'N/A'}
    Postal Code: {invoice_data.postal_code or 'N/A'}
    Country: {invoice_data.country or 'N/A'}
    Phone: {invoice_data.customer_phone or 'N/A'}
    Fax: {invoice_data.customer_fax or 'N/A'}
    Invoice Date: {invoice_data.invoice_date}
    Total Price: {invoice_data.total_price}
    """.strip()
    
    index.upsert([{
        "id": f"invoice_{invoice_id}",
        "values": get_embedding(invoice_text),
        "metadata": {
            "type": "invoice",
            "user_id": user_id,
            "invoice_id": invoice_id,
            "order_id": invoice_data.order_id,
            "contact_name": invoice_data.contact_name,
            "total_price": invoice_data.total_price,
            "invoice_date": invoice_data.invoice_date
        }
    }])
    
    # Store comprehensive line items
    for i, item in enumerate(invoice_data.items):
        item_text = f"""
        Product ID: {item.product_id or 'N/A'}
        Product Name: {item.product_name}
        Quantity: {item.quantity or 'N/A'}
        Unit Price: {item.unit_price or 'N/A'}
        Line Total: {item.line_total or 'N/A'}
        Order ID: {invoice_data.order_id}
        Invoice Date: {invoice_data.invoice_date}
        """.strip()
        
        index.upsert([{
            "id": f"item_{invoice_id}_{i}",
            "values": get_embedding(item_text),
            "metadata": {
                "type": "line_item",
                "user_id": user_id,
                "invoice_id": invoice_id,
                "order_id": invoice_data.order_id,
                "product_id": item.product_id,
                "product_name": item.product_name,
                "quantity": item.quantity,
                "unit_price": item.unit_price,
                "line_total": item.line_total
            }
        }])
    
    
def search_invoices(query: str, user_id: int = None, top_k: int = 5) -> List[Dict]:
    query_embedding = get_embedding(query)
    
    filter_dict = {}
    if user_id:
        filter_dict["user_id"] = user_id
    
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filter_dict
    )
    
    return [{"score": match.score, "metadata": match.metadata} for match in results.matches]
