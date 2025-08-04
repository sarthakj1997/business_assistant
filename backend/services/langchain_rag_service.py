# backend/services/langchain_rag_service.py - Fixed version
import os
from typing import Dict, List
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from sqlalchemy import text
from database.setup_db import SessionLocal
from .rag_embedding_invoice import search_invoices

class LangChainRAGService:
    def __init__(self):
        self.llm = ChatGroq(
            model="llama3-70b-8192",
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        # Session-based memory storage
        self.memories = {}
        
        # Simple prompt templates without memory integration
        self.direct_answer_prompt = ChatPromptTemplate.from_template(
            """You are a business assistant. Answer questions about invoices directly and concisely.

Rules:
- Provide only the requested information
- Don't mention data sources or analysis process
- Use bullet points for multiple items
- Maximum 3 sentences

Previous conversation:
{chat_history}

Context: {context}

Question: {question}

Answer:"""
        )
        
        self.sql_response_prompt = ChatPromptTemplate.from_template(
            """Format the database results into a clear, direct answer. Don't mention it's from a database.

Previous conversation:
{chat_history}

Data: {data}

Question: {question}

Answer:"""
        )
    
    def get_memory(self, session_id: str) -> ConversationBufferWindowMemory:
        """Get or create memory for session"""
        if session_id not in self.memories:
            self.memories[session_id] = ConversationBufferWindowMemory(
                k=5,  # Keep last 5 exchanges
                return_messages=True
            )
        return self.memories[session_id]
    
    def answer_question(self, question: str, user_id: int = None, session_id: str = "default") -> Dict:
        """Main method with LangChain memory and prompts"""
        
        memory = self.get_memory(session_id)
        thinking_steps = []
        sources = {"vector_search": [], "database_query": None, "sql_results": []}
        
        # Get chat history
        chat_history = self._format_chat_history(memory)
        
        # Determine strategy
        strategy = self._determine_strategy(question)
        thinking_steps.append(f"🎯 Strategy: {strategy}")
        
        if strategy == "direct_sql":
            answer = self._handle_direct_sql(question, user_id, chat_history, thinking_steps, sources)
        else:
            answer = self._handle_vector_search(question, user_id, chat_history, thinking_steps, sources)
        
        # Store in memory manually
        memory.save_context(
            {"input": question},
            {"output": answer}
        )
        
        return {
            "thinking": thinking_steps,
            "answer": answer,
            "sources": sources,
            "session_id": session_id
        }
    
    def _format_chat_history(self, memory: ConversationBufferWindowMemory) -> str:
        """Format chat history as string"""
        messages = memory.chat_memory.messages
        if not messages:
            return "No previous conversation."
        
        history_parts = []
        for msg in messages[-6:]:  # Last 3 exchanges
            if isinstance(msg, HumanMessage):
                history_parts.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                history_parts.append(f"Assistant: {msg.content}")
        
        return "\n".join(history_parts)
    
    def _determine_strategy(self, question: str) -> str:
        question_lower = question.lower()
        
        # More precise patterns
        if any(pattern in question_lower for pattern in [
            r"order id \d+", r"invoice \d+", r"order \d+",
            r"customer \w+", r"total price \d+", r"dated \d+"
        ]):
            return "direct_sql"
        
        # Add more vector patterns
        if any(pattern in question_lower for pattern in [
            "product", "item", "contains", "includes", "what products"
        ]):
            return "vector_search"
        
        return "vector_search"  # Default
    
    def _handle_direct_sql(self, question: str, user_id: int, chat_history: str,
                          thinking_steps: List, sources: Dict) -> str:
        """Handle with direct SQL + LangChain prompt"""
        
        thinking_steps.append("💾 Using direct SQL query")
        
        # Build and execute SQL
        sql_query = self._build_sql_query(question, user_id)
        sources["database_query"] = sql_query
        
        sql_results = self._execute_sql_query(sql_query)
        sources["sql_results"] = sql_results
        
        # Use LangChain prompt template
        prompt = self.sql_response_prompt.format(
            chat_history=chat_history,
            data=sql_results,
            question=question
        )
        
        response = self.llm.invoke(prompt)
        return response.content
    
    def _handle_vector_search(self, question: str, user_id: int, chat_history: str,
                             thinking_steps: List, sources: Dict) -> str:
        """Handle with vector search + LangChain prompt"""
        
        thinking_steps.append("🔍 Using vector search")
        
        # Vector search
        vector_results = search_invoices(question, user_id, top_k=10)
        sources["vector_search"] = self._format_vector_sources(vector_results)
        
        # Format context
        context = self._format_vector_context(vector_results)
        
        # Use LangChain prompt template
        prompt = self.direct_answer_prompt.format(
            chat_history=chat_history,
            context=context,
            question=question
        )
        
        response = self.llm.invoke(prompt)
        return response.content
    
    def _build_sql_query(self, question: str, user_id: int) -> str:
        import re
        
        # More precise order ID extraction
        order_match = re.search(r'\b(?:order|invoice)(?:\s*id)?\s*(\d{5})\b', question.lower())
        if order_match:
            order_id = order_match.group(1)
            return f"""
                SELECT i.order_id, i.contact_name, i.invoice_date, i.total_price,
                    li.product_name, li.quantity, li.unit_price
                FROM invoices i
                LEFT JOIN line_items li ON i.id = li.invoice_id
                WHERE i.order_id = '{order_id}'
                AND ({user_id} IS NULL OR i.user_id = {user_id})
                ORDER BY i.invoice_date DESC
            """
        
        # Customer name extraction
        customer_match = re.search(r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b', question)
        if customer_match:
            customer = customer_match.group(1)
            return f"""
                SELECT i.order_id, i.contact_name, i.invoice_date, i.total_price
                FROM invoices i
                WHERE i.contact_name ILIKE '%{customer}%'
                AND ({user_id} IS NULL OR i.user_id = {user_id})
                ORDER BY i.invoice_date DESC
            """
        
        # Amount-based queries
        amount_match = re.search(r'\$?(\d+\.?\d*)', question)
        if amount_match and any(word in question.lower() for word in ['total', 'price', 'amount']):
            amount = amount_match.group(1)
            return f"""
                SELECT i.order_id, i.contact_name, i.invoice_date, i.total_price
                FROM invoices i
                WHERE i.total_price = {amount}
                AND ({user_id} IS NULL OR i.user_id = {user_id})
            """
        
        return f"""
            SELECT i.order_id, i.contact_name, i.invoice_date, i.total_price
            FROM invoices i
            WHERE ({user_id} IS NULL OR i.user_id = {user_id})
            ORDER BY i.invoice_date DESC
            LIMIT 5
        """

    
    def _execute_sql_query(self, sql_query: str) -> List[Dict]:
        """Execute SQL query"""
        db = SessionLocal()
        try:
            result = db.execute(text(sql_query))
            return [dict(row._mapping) for row in result]
        except Exception as e:
            return [{"error": str(e)}]
        finally:
            db.close()
    
    def _format_vector_sources(self, results: List[Dict]) -> List[Dict]:
        """Format vector sources"""
        sources = []
        for i, result in enumerate(results[:5]):
            metadata = result['metadata']
            sources.append({
                "rank": i + 1,
                "score": round(result.get('score', 0), 3),
                "type": metadata.get('type'),
                "order_id": metadata.get('order_id'),
                "customer": metadata.get('contact_name')
            })
        return sources
    
    def _format_vector_context(self, results: List[Dict]) -> str:
        # Only use top 3 most relevant results
        context_parts = []
        for result in results[:3]:  # Reduced from 5 to 3
            metadata = result['metadata']
            score = result.get('score', 0)
            
            # Only include high-confidence results
            if score > 0.7:  # Threshold for relevance
                if metadata['type'] == 'invoice':
                    context_parts.append(
                        f"Invoice {metadata.get('order_id')}: {metadata.get('contact_name')}, "
                        f"Date: {metadata.get('invoice_date')}, Total: ${metadata.get('total_price')}"
                    )
        
        return "\n".join(context_parts) if context_parts else "No highly relevant results found."

    
    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Get conversation history for a session"""
        if session_id not in self.memories:
            return []
        
        memory = self.memories[session_id]
        messages = memory.chat_memory.messages
        
        history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append({"role": "human", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})
        
        return history
