# backend/services/rag_qa_service.py
import os
from typing import List, Dict
from groq import Groq
from datetime import datetime, timedelta
from collections import defaultdict
from .rag_embedding_invoice import search_invoices

class RAGQAService:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama3-8b-8192"
    
    def answer_question(self, question: str, user_id: int = None) -> str:
        # Retrieve more context for better accuracy
        context_results = search_invoices(question, user_id, top_k=20)
        
        # Enhanced context formatting with aggregation
        context = self._format_enhanced_context(context_results)
        
        # Improved prompt with examples
        prompt = f"""You are a business analyst. Answer questions about invoice data accurately using ONLY the provided context.

Context Data:
{context}

Question: {question}

Instructions:
- Count items carefully when asked about quantities
- For date-based questions, consider the invoice dates provided
- For product questions, aggregate data from all relevant line items
- If data is insufficient, clearly state what's missing
- Provide specific numbers and dates when available

Answer:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300
        )
        
        return response.choices[0].message.content
    
    def _format_enhanced_context(self, results: List[Dict]) -> str:
        invoices = []
        products = defaultdict(lambda: {"quantity": 0, "orders": set()})
        
        for result in results:
            metadata = result['metadata']
            
            if metadata['type'] == 'invoice':
                invoices.append({
                    "order_id": metadata.get('order_id'),
                    "date": metadata.get('invoice_date'),
                    "customer": metadata.get('contact_name'),
                    "total": metadata.get('total_price')
                })
            else:  # line_item
                product_name = metadata.get('product_name', 'Unknown')
                quantity = metadata.get('quantity', 0)
                order_id = metadata.get('order_id')
                
                if quantity and order_id:
                    products[product_name]["quantity"] += int(quantity)
                    products[product_name]["orders"].add(order_id)
        
        # Format context
        context_parts = []
        
        if invoices:
            context_parts.append("INVOICES:")
            for inv in invoices:
                context_parts.append(f"- Order {inv['order_id']}: {inv['date']}, Customer: {inv['customer']}, Total: ${inv['total']}")
        
        if products:
            context_parts.append("\nPRODUCTS:")
            for product, data in products.items():
                context_parts.append(f"- {product}: Total Quantity: {data['quantity']}, Orders: {len(data['orders'])}")
        
        return "\n".join(context_parts)
    
    def _filter_by_date_range(self, results: List[Dict], days_back: int = 7) -> List[Dict]:
        cutoff_date = datetime.now() - timedelta(days=days_back)
        filtered = []
        
        for result in results:
            date_str = result['metadata'].get('invoice_date')
            if date_str:
                try:
                    invoice_date = datetime.strptime(date_str, '%Y-%m-%d')
                    if invoice_date >= cutoff_date:
                        filtered.append(result)
                except:
                    continue
        
        return filtered
