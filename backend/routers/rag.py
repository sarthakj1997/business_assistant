# # backend/routers/rag.py
# from fastapi import APIRouter
# from backend.services.rag_embedding_invoice import search_invoices
# from backend.services.agent_rag_service import AgentRAGService

# router = APIRouter()
# agent_service = AgentRAGService()

# @router.get("/search")
# async def search_invoice_data(query: str, user_id: int = None):
#     results = search_invoices(query, user_id)
#     return {"results": results}

# @router.post("/ask")
# async def ask_question(question: str, user_id: int = None):
#     response = agent_service.answer_question(question, user_id)
#     return {
#         "question": question,
#         "thinking": response["thinking"],
#         "answer": response["answer"],
#         "sources": response["sources"]
#     }

# backend/routers/rag.py
from fastapi import APIRouter
from backend.services.langchain_rag_service import LangChainRAGService

router = APIRouter()
langchain_service = LangChainRAGService()

@router.post("/ask")
async def ask_question(question: str, user_id: int = None, session_id: str = "default"):
    response = langchain_service.answer_question(question, user_id, session_id)
    return response

@router.get("/history/{session_id}")
async def get_history(session_id: str):
    history = langchain_service.get_conversation_history(session_id)
    return {"session_id": session_id, "history": history}
