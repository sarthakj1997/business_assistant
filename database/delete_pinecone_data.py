# delete_pinecone_data.py
import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = "invoice-rag"

def delete_all_vectors():
    index = pc.Index(INDEX_NAME)
    
    # Delete all vectors in the index
    index.delete(delete_all=True)
    print("All vectors deleted from Pinecone index")

if __name__ == "__main__":
    delete_all_vectors()
