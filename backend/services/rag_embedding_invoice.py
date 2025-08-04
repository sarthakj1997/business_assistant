
import os
from pinecone import Pinecone
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
    """Store invoice data optimized for search"""
    
    # 1. Main invoice document - natural language
    invoice_text = f"""
    Invoice Order {invoice_data.order_id} for customer {invoice_data.contact_name or 'Unknown'}.
    Date: {invoice_data.invoice_date}. Total amount: ${invoice_data.total_price}.
    Customer details: {invoice_data.contact_name}, {invoice_data.city}, {invoice_data.country}.
    Order ID {invoice_data.order_id}. Invoice {invoice_data.order_id}.
    """
    
    index.upsert([{
        "id": f"invoice_{invoice_data.order_id}",
        "values": get_embedding(invoice_text),
        "metadata": {
            "type": "invoice",
            "user_id": user_id,
            "invoice_id": invoice_id,
            "order_id": invoice_data.order_id,
            "contact_name": invoice_data.contact_name,
            "total_price": float(invoice_data.total_price),
            "invoice_date": str(invoice_data.invoice_date),
            "city": invoice_data.city,
            "country": invoice_data.country
        }
    }])
    
    # 2. Combined line items document
    if invoice_data.items:
        products_text = f"""
        Order {invoice_data.order_id} contains the following products:
        """
        
        for item in invoice_data.items:
            products_text += f"""
            {item.product_name} - quantity {item.quantity} at ${item.unit_price} each (total ${item.line_total}).
            """
        
        products_text += f"This order {invoice_data.order_id} was placed by {invoice_data.contact_name} on {invoice_data.invoice_date}."
        
        index.upsert([{
            "id": f"products_{invoice_data.order_id}",
            "values": get_embedding(products_text),
            "metadata": {
                "type": "products",
                "user_id": user_id,
                "invoice_id": invoice_id,
                "order_id": invoice_data.order_id,
                "contact_name": invoice_data.contact_name,
                "product_count": len(invoice_data.items),
                "products": [item.product_name for item in invoice_data.items]
            }
        }])
    
    # 3. Individual product documents for specific searches
    for i, item in enumerate(invoice_data.items):
        item_text = f"""
        Product {item.product_name} in order {invoice_data.order_id}.
        Customer {invoice_data.contact_name} ordered {item.quantity} units of {item.product_name} 
        at ${item.unit_price} per unit for a total of ${item.line_total}.
        Order date: {invoice_data.invoice_date}. Order ID {invoice_data.order_id}.
        """
        
        index.upsert([{
            "id": f"item_{invoice_data.order_id}_{i}",
            "values": get_embedding(item_text),
            "metadata": {
                "type": "line_item",
                "user_id": user_id,
                "invoice_id": invoice_id,
                "order_id": invoice_data.order_id,
                "product_name": item.product_name,
                "quantity": int(item.quantity) if item.quantity else 0,
                "unit_price": float(item.unit_price) if item.unit_price else 0,
                "line_total": float(item.line_total) if item.line_total else 0,
                "contact_name": invoice_data.contact_name
            }
        }])

def search_invoices(query: str, user_id: int = None, top_k: int = 10) -> List[Dict]:
    import re
    
    # Enhanced query preprocessing
    enhanced_query = query
    
    # Extract and boost exact order IDs
    order_match = re.search(r'\b(\d{5})\b', query)
    if order_match:
        order_id = order_match.group(1)
        enhanced_query = f"order {order_id} invoice {order_id} order ID {order_id} {query}"
    
    # Extract customer names and boost
    customer_match = re.search(r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b', query)
    if customer_match:
        customer = customer_match.group(1)
        enhanced_query = f"customer {customer} {query}"
    
    query_embedding = get_embedding(enhanced_query)
    
    filter_dict = {}
    if user_id:
        filter_dict["user_id"] = user_id
    
    # Get more results for re-ranking
    results = index.query(
        vector=query_embedding,
        top_k=top_k * 2,  # Get 2x results for re-ranking
        include_metadata=True,
        filter=filter_dict
    )
    
    # Re-rank results based on query type
    ranked_results = _rerank_results(results.matches, query)
    
    return [{"score": match.score, "metadata": match.metadata} for match in ranked_results[:top_k]]

def _rerank_results(matches, query):
    import re
    
    # Boost exact order ID matches
    order_match = re.search(r'\b(\d{5})\b', query)
    if order_match:
        order_id = order_match.group(1)
        for match in matches:
            if match.metadata.get('order_id') == order_id:
                match.score += 0.3  # Significant boost
    
    # Boost invoice type over line items for general queries
    for match in matches:
        if match.metadata.get('type') == 'invoice':
            match.score += 0.1
    
    return sorted(matches, key=lambda x: x.score, reverse=True)
