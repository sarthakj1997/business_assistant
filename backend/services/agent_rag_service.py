# backend/services/agent_rag_service.py
import os
import json
from typing import List, Dict, Set
from groq import Groq
from sqlalchemy import text
from database.setup_db import SessionLocal
from .rag_embedding_invoice import search_invoices

class AgentRAGService:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama3-70b-8192"
    
    def answer_question(self, question: str, user_id: int = None) -> Dict:
        """Return structured response with thinking, answer, and sources"""
        order_id = self._is_exact_order_query(question)
        if order_id:
            return self._handle_exact_order_query(order_id, user_id)
        # Step 1: Vector search and metadata extraction
        vector_results = search_invoices(question, user_id, top_k=20)
        metadata_context = self._extract_metadata_context(vector_results)
        
        # Step 2: Determine strategy and execute
        thinking_steps = []
        sources = {"vector_search": [], "database_query": None, "sql_results": []}
        
        thinking_steps.append(f"🔍 Vector search found {len(vector_results)} relevant results")
        thinking_steps.append(f"📊 Extracted metadata: {len(metadata_context['customers'])} customers, {len(metadata_context['order_ids'])} orders, {len(metadata_context['products'])} products")
        
        # Collect vector sources
        sources["vector_search"] = self._format_vector_sources(vector_results[:10])
        
        if self._needs_sql_query(question, metadata_context):
            thinking_steps.append("🎯 Question requires SQL aggregation - building targeted query")
            
            sql_query = self._build_targeted_sql(question, metadata_context, user_id)
            sources["database_query"] = sql_query.strip()
            
            sql_results = self._execute_sql_query(sql_query)
            sources["sql_results"] = sql_results
            
            thinking_steps.append(f"💾 SQL query returned {len(sql_results)} structured results")
            
            answer = self._generate_combined_answer(question, vector_results, sql_results, metadata_context)
        else:
            thinking_steps.append("📝 Using vector search only for semantic answer")
            answer = self._generate_vector_answer(question, vector_results)
        
        return {
            "thinking": thinking_steps,
            "answer": answer,
            "sources": sources
        }
    
    def _format_vector_sources(self, results: List[Dict]) -> List[Dict]:
        """Format vector search results as sources - Updated for new structure"""
        sources = []
        for i, result in enumerate(results):
            metadata = result['metadata']
            source = {
                "rank": i + 1,
                "score": round(result.get('score', 0), 3),
                "type": metadata.get('type'),
                "order_id": metadata.get('order_id')
            }
            
            if metadata['type'] == 'invoice':
                source.update({
                    "customer": metadata.get('contact_name'),
                    "date": metadata.get('invoice_date'),
                    "total": metadata.get('total_price'),
                    "city": metadata.get('city'),
                    "country": metadata.get('country')
                })
            elif metadata['type'] == 'products':
                source.update({
                    "customer": metadata.get('contact_name'),
                    "product_count": metadata.get('product_count'),
                    "products": metadata.get('products', [])
                })
            else:  # line_item
                source.update({
                    "product": metadata.get('product_name'),
                    "quantity": metadata.get('quantity'),
                    "price": metadata.get('unit_price'),
                    "customer": metadata.get('contact_name')
                })
            
            sources.append(source)
        
        return sources
    

    def _generate_combined_answer(self, question: str, vector_results: List[Dict], 
                             sql_results: List[Dict], metadata_context: Dict) -> str:
        """Generate final answer using both sources"""
        
        vector_context = self._format_vector_context(vector_results[:5])
        sql_context = self._format_sql_context(sql_results)
        
        prompt = f"""Answer the business question with ONLY the essential facts. Be direct and concise.

    Vector Context:
    {vector_context}

    SQL Results:
    {sql_context}

    Question: {question}

    Rules:
    - Give only the key facts requested
    - Use bullet points for multiple items
    - No explanations or background
    - Maximum 2-3 sentences

    Answer:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200  # Reduced to force conciseness
        )
        
        return response.choices[0].message.content

    def _generate_vector_answer(self, question: str, vector_results: List[Dict]) -> str:
        """Generate answer using vector results only"""
        
        context = self._format_vector_context(vector_results)
        
        prompt = f"""Answer directly with only the requested information. Be concise.

    Data:
    {context}

    Question: {question}

    Rules:
    - State only the facts asked for
    - No extra details or context
    - Maximum 2 sentences

    Answer:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=150  # Reduced for brevity
        )
        
        return response.choices[0].message.content



    
    def _extract_metadata_context(self, vector_results: List[Dict]) -> Dict:
        """Extract useful metadata for filtering and context"""
        context = {
            'customers': set(),
            'order_ids': set(),
            'products': set(),
            'date_range': {'min': None, 'max': None},
            'invoice_ids': set(),
            'total_results': len(vector_results)
        }
        
        for result in vector_results:
            metadata = result['metadata']
            
            # Collect customers
            if metadata.get('contact_name'):
                context['customers'].add(metadata['contact_name'])
            
            # Collect order IDs
            if metadata.get('order_id'):
                context['order_ids'].add(metadata['order_id'])
            
            # Collect products
            if metadata.get('product_name'):
                context['products'].add(metadata['product_name'])
            
            # Collect invoice IDs
            if metadata.get('invoice_id'):
                context['invoice_ids'].add(metadata['invoice_id'])
            
            # Track date range
            if metadata.get('invoice_date'):
                date = metadata['invoice_date']
                if not context['date_range']['min'] or date < context['date_range']['min']:
                    context['date_range']['min'] = date
                if not context['date_range']['max'] or date > context['date_range']['max']:
                    context['date_range']['max'] = date
        
        return context
    
    def _needs_sql_query(self, question: str, metadata_context: Dict) -> bool:
        """Determine if question needs SQL for accurate answers"""
        sql_keywords = [
            'how many', 'count', 'total', 'sum', 'average', 'most', 'least',
            'top', 'bottom', 'all orders', 'all customers', 'all products'
        ]
        
        # If question asks for aggregation or we have specific entities to filter
        return (any(keyword in question.lower() for keyword in sql_keywords) or 
                len(metadata_context['customers']) > 0 or 
                len(metadata_context['order_ids']) > 0)
    
    def _handle_with_sql_and_vector(self, question: str, vector_results: List[Dict], 
                                   metadata_context: Dict, user_id: int) -> str:
        """Use metadata to create targeted SQL queries + vector context"""
        
        # Build SQL query based on metadata context
        sql_query = self._build_targeted_sql(question, metadata_context, user_id)
        
        # Execute SQL query
        sql_results = self._execute_sql_query(sql_query)
        
        # Combine vector and SQL results
        return self._generate_combined_answer(question, vector_results, sql_results, metadata_context)
    
    def _build_targeted_sql(self, question: str, metadata_context: Dict, user_id: int) -> str:
        """Build SQL query using metadata context for filtering"""
        
        base_conditions = []
        if user_id:
            base_conditions.append(f"i.user_id = {user_id}")
        
        # Filter by customers found in vector search
        if metadata_context['customers']:
            customer_list = "', '".join(metadata_context['customers'])
            base_conditions.append(f"i.contact_name IN ('{customer_list}')")
        
        # Filter by order IDs found in vector search
        if metadata_context['order_ids']:
            order_list = "', '".join(metadata_context['order_ids'])
            base_conditions.append(f"i.order_id IN ('{order_list}')")
        
        where_clause = " AND ".join(base_conditions) if base_conditions else "1=1"
        
        # Choose query type based on question
        if any(word in question.lower() for word in ['how many orders', 'count orders']):
            return f"""
                SELECT COUNT(DISTINCT i.order_id) as order_count,
                       COUNT(DISTINCT i.contact_name) as customer_count
                FROM invoices i WHERE {where_clause}
            """
        
        elif any(word in question.lower() for word in ['total spent', 'how much', 'total amount']):
            return f"""
                SELECT i.contact_name, SUM(i.total_price) as total_spent,
                       COUNT(i.order_id) as order_count
                FROM invoices i WHERE {where_clause}
                GROUP BY i.contact_name
                ORDER BY total_spent DESC
            """
        
        elif any(word in question.lower() for word in ['most ordered', 'popular product', 'top product']):
            return f"""
                SELECT li.product_name, SUM(li.quantity) as total_quantity,
                       COUNT(DISTINCT i.order_id) as order_count
                FROM invoices i
                JOIN line_items li ON i.id = li.invoice_id
                WHERE {where_clause}
                GROUP BY li.product_name
                ORDER BY total_quantity DESC
                LIMIT 10
            """
        
        else:
            # Default: get detailed order information
            return f"""
                SELECT i.order_id, i.contact_name, i.invoice_date, i.total_price,
                       li.product_name, li.quantity, li.unit_price
                FROM invoices i
                LEFT JOIN line_items li ON i.id = li.invoice_id
                WHERE {where_clause}
                ORDER BY i.invoice_date DESC
                LIMIT 50
            """
    
    def _execute_sql_query(self, sql_query: str) -> List[Dict]:
        """Execute SQL query and return results"""
        db = SessionLocal()
        try:
            result = db.execute(text(sql_query))
            return [dict(row._mapping) for row in result]
        except Exception as e:
            return [{"error": str(e)}]
        finally:
            db.close()
    
    def _generate_combined_answer(self, question: str, vector_results: List[Dict], 
                                 sql_results: List[Dict], metadata_context: Dict) -> str:
        """Generate answer using both vector and SQL results"""
        
        # Format context from both sources
        vector_context = self._format_vector_context(vector_results[:5])  # Top 5 for context
        sql_context = self._format_sql_context(sql_results)
        
        prompt = f"""Answer the question using the provided data sources.

Vector Search Context (semantic matches):
{vector_context}

Database Query Results (structured data):
{sql_context}

Metadata Summary:
- Found {len(metadata_context['customers'])} customers: {', '.join(list(metadata_context['customers'])[:3])}
- Found {len(metadata_context['order_ids'])} orders
- Found {len(metadata_context['products'])} products

Question: {question}

Provide a comprehensive answer using both the semantic context and structured data."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=400
        )
        
        return response.choices[0].message.content
    
    def _handle_with_vector_only(self, question: str, vector_results: List[Dict]) -> str:
        """Handle with vector search only"""
        context = self._format_vector_context(vector_results)
        
        prompt = f"""Based on the invoice data below, answer the question accurately.

Data:
{context}

Question: {question}

Answer with specific details from the data provided."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300
        )
        
        return response.choices[0].message.content
    
    def _format_vector_context(self, results: List[Dict]) -> str:
        """Updated context formatting for new vector structure"""
        context_parts = []
        for result in results:
            metadata = result['metadata']
            
            if metadata['type'] == 'invoice':
                context_parts.append(
                    f"Invoice {metadata.get('order_id')}: Customer {metadata.get('contact_name')}, "
                    f"Date: {metadata.get('invoice_date')}, Total: ${metadata.get('total_price')}, "
                    f"Location: {metadata.get('city')}, {metadata.get('country')}"
                )
            elif metadata['type'] == 'products':
                products = metadata.get('products', [])
                context_parts.append(
                    f"Order {metadata.get('order_id')} products: {', '.join(products[:3])}... "
                    f"({metadata.get('product_count')} total items)"
                )
            else:  # line_item
                context_parts.append(
                    f"Product: {metadata.get('product_name')}, "
                    f"Qty: {metadata.get('quantity')}, Price: ${metadata.get('unit_price')}, "
                    f"Order: {metadata.get('order_id')}"
                )
        
        return "\n".join(context_parts)
    
    def _format_sql_context(self, results: List[Dict]) -> str:
        if not results or (len(results) == 1 and 'error' in results[0]):
            return "No structured data available"
        
        return json.dumps(results, indent=2, default=str)

    def _is_exact_order_query(self, question: str) -> str:
        """Check if question asks for specific order ID"""
        import re
        patterns = [
            r"order id (\d+)",
            r"order (\d+)",
            r"invoice (\d+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, question.lower())
            if match:
                return match.group(1)
        return None

    def _handle_exact_order_query(self, order_id: str, user_id: int) -> Dict:
        """Handle exact order ID queries with direct SQL"""
        db = SessionLocal()
        try:
            query = text("""
                SELECT i.order_id, i.contact_name, i.invoice_date, i.total_price,
                    li.product_name, li.quantity, li.unit_price, li.line_total
                FROM invoices i
                LEFT JOIN line_items li ON i.id = li.invoice_id
                WHERE i.order_id = :order_id
                AND (:user_id IS NULL OR i.user_id = :user_id)
            """)
            
            result = db.execute(query, {"order_id": order_id, "user_id": user_id})
            rows = result.fetchall()
            
            if not rows:
                return {
                    "thinking": [f"🔍 Searched for exact order ID {order_id}", "❌ No invoice found with this order ID"],
                    "answer": f"No invoice found with Order ID {order_id}.",
                    "sources": {"database_query": str(query), "sql_results": []}
                }
            
            # Format response
            invoice_info = rows[0]
            answer = f"Invoice {order_id}:\n"
            answer += f"Customer: {invoice_info.contact_name}\n"
            answer += f"Date: {invoice_info.invoice_date}\n"
            answer += f"Total: ${invoice_info.total_price}\n\n"
            
            if invoice_info.product_name:
                answer += "Line Items:\n"
                for row in rows:
                    if row.product_name:
                        answer += f"• {row.product_name}: {row.quantity} units @ ${row.unit_price} = ${row.line_total}\n"
            
            return {
                "thinking": [f"🔍 Direct SQL query for order ID {order_id}", f"✅ Found invoice with {len(rows)} line items"],
                "answer": answer,
                "sources": {"database_query": str(query), "sql_results": [dict(row._mapping) for row in rows]}
            }
            
        finally:
            db.close()

