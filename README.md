# Business Assistant - AI-Powered Invoice Processing & Query System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116+-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.47+-red.svg)](https://streamlit.io)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-blue.svg)](https://postgresql.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://docker.com)

##  Overview

**Business Assistant** is an intelligent invoice processing and querying system that transforms PDF invoices into a queryable knowledge base. Users can upload invoices and ask natural language questions to get instant insights about their financial data.

### Key Features
- 📄 **PDF Invoice Processing**: Automated extraction and parsing of invoice data
- 🤖 **Natural Language Queries**: Ask questions in plain English
- 🔍 **Hybrid Search**: Combines vector similarity and SQL queries for optimal results
- 💬 **Conversation Memory**: Maintains context across multiple questions
- 📊 **Source Attribution**: Shows which invoices support each answer
- ⚡ **Real-time Processing**: Instant responses with thinking process visualization

## 🏗️ Architecture

### System Components
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │    FastAPI      │    │   PostgreSQL    │
│   Frontend      │◄──►│    Backend      │◄──►│   Database      │
│   (Port 8501)   │    │   (Port 8000)   │    │   (Port 5432)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Pinecone      │
                       │ Vector Database │
                       └─────────────────┘
```

### Technology Stack
- **Backend**: FastAPI, SQLAlchemy, Alembic
- **Frontend**: Streamlit with interactive chat interface
- **Database**: PostgreSQL for structured data
- **Vector Store**: Pinecone for semantic search
- **AI/ML**: LangChain, Groq/OpenAI APIs
- **Document Processing**: PyMuPDF for PDF parsing
- **Deployment**: Docker & Docker Compose

## 📁 Project Structure

```
business_assistant/
├── backend/
│   ├── main.py                     # FastAPI application
│   ├── routers/
│   │   ├── upload_pdf.py          # PDF upload endpoints
│   │   └── rag.py                 # Query processing endpoints
│   └── services/
│       ├── parse_invoice.py       # PDF text extraction
│       ├── process_invoice.py     # Data parsing & validation
│       ├── rag_embedding_invoice.py # Vector search
│       ├── agent_rag_service.py   # RAG implementation
│       └── langchain_rag_service.py # LangChain service
├── database/
│   ├── models/                    # SQLAlchemy models
│   ├── migrations/                # Alembic migrations
│   └── setup_db.py               # Database configuration
├── frontend/
│   └── streamlit_app.py          # Web interface
├── Docker/
│   ├── docker-compose.yml        # Multi-service deployment
│   ├── Dockerfile.backend        # Backend container
│   └── Dockerfile.frontend       # Frontend container
├── evaluation/
│   └── comprehensive_evaluation.py # Performance testing
└── test_notebooks/
    └── batch_invoices/           # Sample invoice PDFs
```

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- API Keys for:
  - Groq API (for LLM)
  - Pinecone (for vector storage)
  - OpenAI (optional backup)

### 1. Clone Repository
```bash
git clone https://github.com/sarthakj1997/business_assistant.git
cd business_assistant
```

### 2. Environment Setup
Create `.env` file:
```env
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
OPENAI_API_KEY=your_openai_api_key  # Optional
```

### 3. Launch with Docker
```bash
cd Docker
docker-compose up --build
```

### 4. Access Application
- **Frontend**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 💡 How It Works

### Invoice Processing Pipeline
1. **Upload**: User uploads PDF invoice via Streamlit interface
2. **Extract**: PyMuPDF extracts text with layout preservation
3. **Parse**: LLM extracts structured data (customer, items, amounts)
4. **Store**: Data saved to PostgreSQL with confidence scores
5. **Embed**: Vector embeddings created and stored in Pinecone

### Query Processing
1. **Input**: User asks natural language question
2. **Strategy**: System determines optimal approach:
   - **Direct SQL**: For exact lookups ("Show order 10250")
   - **Vector Search**: For semantic queries ("Products from Mario")
   - **Hybrid**: Combines both for complex questions
3. **Execute**: Runs appropriate queries
4. **Generate**: LLM creates natural language response
5. **Display**: Shows answer with sources and thinking process

## 🔍 Query Examples

```
# Exact Order Lookup
"Show me invoice with Order ID 10250"

# Customer Queries
"What invoices are for Mario Pontes?"
"How much has VICTE customer spent?"

# Product Searches
"Which orders contain Chai products?"
"Show me all dairy products ordered"

# Analytics
"What's the total revenue from Brazil?"
"Which customer has the highest order value?"

# Date-based
"Show invoices from July 2016"
"Recent orders from this month"
```

## 🗄️ Database Schema

### Invoices Table
```sql
CREATE TABLE invoices (
    id SERIAL PRIMARY KEY,
    user_id INTEGER,
    order_id VARCHAR UNIQUE,
    customer_id VARCHAR,
    invoice_date DATE,
    contact_name VARCHAR,
    address VARCHAR,
    city VARCHAR,
    country VARCHAR,
    total_price FLOAT,
    confidence_score FLOAT,
    raw_text TEXT
);
```

### Line Items Table
```sql
CREATE TABLE line_items (
    id SERIAL PRIMARY KEY,
    invoice_id INTEGER REFERENCES invoices(id),
    product_name TEXT,
    quantity INTEGER,
    unit_price FLOAT,
    line_total FLOAT,
    confidence_score FLOAT
);
```

## 🤖 AI Components

### Vector Embeddings
Three types of embeddings stored in Pinecone:
- **Invoice-level**: Customer, date, total, location metadata
- **Product-level**: Aggregated products per invoice
- **Line-item level**: Individual product details

### LLM Integration
- **Primary**: Groq API with Llama3-70B
- **Backup**: OpenAI GPT models
- **Framework**: LangChain for memory and prompts
- **Temperature**: 0 for consistent, factual responses

### Query Strategy Routing
```python
def determine_strategy(question):
    if "order id" in question.lower():
        return "direct_sql"  # Exact lookup
    elif "product" in question.lower():
        return "vector_search"  # Semantic search
    else:
        return "hybrid"  # Combined approach
```

## 📊 Performance & Evaluation

### Metrics Tracked
- **Retrieval Quality**: Precision@5, Recall@10, MRR
- **Answer Accuracy**: Entity extraction, factual correctness
- **Performance**: Response times, strategy routing accuracy
- **User Experience**: Source attribution, thinking transparency

### Run Evaluation
```bash
python evaluation/comprehensive_evaluation.py
```

## 🔧 Development

### Local Setup (without Docker)
```bash
# Install dependencies
pip install -r requirements.txt

# Setup database
python database/setup_db.py

# Run migrations
alembic upgrade head

# Start backend
cd backend && uvicorn main:app --reload --port 8000

# Start frontend (new terminal)
streamlit run frontend/streamlit_app.py --server.port 8501
```

### Adding New Features
1. **New Endpoints**: Add to `backend/routers/`
2. **Database Changes**: Create Alembic migration
3. **AI Services**: Extend `backend/services/`
4. **Frontend**: Modify `frontend/streamlit_app.py`



### API Testing
```bash
# Test upload
curl -X POST "http://localhost:8000/upload/pdf" \
     -F "file=@invoice.pdf" \
     -F "user_id=1"

# Test query
curl -X POST "http://localhost:8000/rag/ask" \
     -G -d "question=Show me order 10250" \
     -d "user_id=1"
```

## 🚀 Deployment

### Production Considerations
- Use environment-specific `.env` files
- Configure proper PostgreSQL credentials
- Set up SSL/TLS for HTTPS
- Implement proper logging and monitoring
- Scale with Kubernetes or cloud services



## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain** for RAG framework
- **Groq** for fast LLM inference
- **Pinecone** for vector database
- **FastAPI** for modern Python web framework
- **Streamlit** for rapid UI development

---

**Built with ❤️ for intelligent business automation**