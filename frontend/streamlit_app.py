import streamlit as st
import requests
import json
import pandas as pd
import os

# FastAPI backend URL
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.title("Business Assistant - Invoice RAG")

# Sidebar for file upload
st.sidebar.header("Upload Invoice")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    if st.sidebar.button("Upload"):
        files = {"file": uploaded_file.getvalue()}
        data = {"user_id": 1}  # Default user ID
        
        try:
            response = requests.post(f"{API_BASE_URL}/upload/pdf", files={"file": uploaded_file}, data=data)
            if response.status_code == 200:
                st.sidebar.success("File uploaded successfully!")
            else:
                st.sidebar.error(f"Upload failed: {response.text}")
        except requests.exceptions.RequestException as e:
            st.sidebar.error(f"Connection error: {e}")

# Main chat interface
st.header("Ask Questions About Your Invoices")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and isinstance(message["content"], dict):
            # Format structured response
            st.markdown(message["content"]["answer"])
        else:
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about your invoices..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response from FastAPI
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/rag/ask",
                    params={"question": prompt, "user_id": 1}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display thinking process
                    with st.expander("🧠 Thinking Process", expanded=False):
                        for i, step in enumerate(result.get("thinking", []), 1):
                            st.markdown(f"**Step {i}:** {step}")
                    
                    # Display main answer with better formatting
                    answer = result.get("answer", "No answer received")
                    
                    # Clean up answer formatting
                    formatted_answer = answer.replace("\\n", "\n").strip()
                    st.markdown(formatted_answer)
                    
                    # Display sources with better formatting
                    with st.expander("📚 Sources Used", expanded=False):
                        sources = result.get("sources", {})
                        
                        if sources.get("vector_search"):
                            st.subheader("🔍 Vector Search Results")
                            
                            # Create a formatted table for vector results
                            vector_data = []
                            for source in sources["vector_search"][:5]:
                                if source.get("type") == "invoice":
                                    vector_data.append({
                                        "Type": "Invoice",
                                        "Score": f"{source.get('score', 0):.3f}",
                                        "Order ID": source.get('order_id', 'N/A'),
                                        "Customer": source.get('customer', 'N/A'),
                                        "Date": source.get('date', 'N/A'),
                                        "Total": f"${source.get('total', 0)}"
                                    })
                                else:
                                    vector_data.append({
                                        "Type": "Product",
                                        "Score": f"{source.get('score', 0):.3f}",
                                        "Product": source.get('product', 'N/A'),
                                        "Quantity": source.get('quantity', 'N/A'),
                                        "Price": f"${source.get('price', 0)}"
                                    })
                            
                            if vector_data:
                                df = pd.DataFrame(vector_data)
                                st.dataframe(df, use_container_width=True)
                        
                        if sources.get("database_query"):
                            st.subheader("💾 SQL Query")
                            st.code(sources["database_query"], language="sql")
                        
                        if sources.get("sql_results"):
                            st.subheader("📊 Database Results")
                            sql_results = sources["sql_results"][:5]  # Show top 5
                            
                            if sql_results and isinstance(sql_results[0], dict):
                                # Convert to DataFrame for better display
                                df_sql = pd.DataFrame(sql_results)
                                st.dataframe(df_sql, use_container_width=True)
                            else:
                                st.json(sql_results)
                    
                    # Store structured response in chat history
                    structured_response = {
                        "answer": formatted_answer,
                        "thinking": result.get("thinking", []),
                        "sources": sources
                    }
                    st.session_state.messages.append({"role": "assistant", "content": structured_response})
                    
                else:
                    error_msg = f"❌ **Error {response.status_code}**\n\n{response.text}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    
            except requests.exceptions.RequestException as e:
                error_msg = f"🔌 **Connection Error**\n\n{str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Clear chat button
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
