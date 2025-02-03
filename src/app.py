import streamlit as st
from langchain_ollama import OllamaLLM
import chromadb
from chromadb.utils import embedding_functions
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Initialize global variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files_content" not in st.session_state:
    st.session_state.uploaded_files_content = []

# Initialize ChromaDB Client and Collection
def initialize_chromadb():
    try:
        client = chromadb.Client()
        ef = embedding_functions.OpenAIEmbeddingFunction(api_key="YOUR_API_KEY")  # Replace with your OpenAI API key
        collection = client.get_or_create_collection(name="chatbot_memory", embedding_function=ef)
        return collection
    except Exception as e:
        st.error(f"Error initializing ChromaDB: {e}")
        return None

# Multi-Query Generation
def generate_multi_queries(user_input, llm):
    prompt_template = """
    Generate multiple reworded versions of the following user query to improve retrieval:
    Query: {query}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["query"])
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(query=user_input).split("\n")

# Retrieve documents using Multi-Query

def retrieve_multi_query_docs(collection, queries, top_k=3):
    results = []
    for query in queries:
        res = collection.query(query_texts=[query], n_results=top_k)
        if res and res.get("documents"):
            results.extend(res["documents"])
    return list(set(results))  # Remove duplicates

# RAG Fusion: Merging results from different queries
def rag_fusion(retrieved_docs):
    return "\n".join(retrieved_docs) if retrieved_docs else ""

# Function to generate a response
def generate_response(model_name, prompt):
    try:
        llm = OllamaLLM(model=model_name)
        response = llm.invoke(prompt)
        return response
    except Exception as e:
        return f"Error: {e}"

# Main function for Streamlit app
def main():
    st.title("Ollama Chatbot with Multi-Query & RAG Fusion")

    # Initialize ChromaDB
    collection = initialize_chromadb()
    if collection is None:
        st.warning("ChromaDB could not be initialized. Contextual memory is disabled.")

    # Sidebar for model selection
    st.sidebar.header("Settings")
    model_choice = st.sidebar.selectbox("Choose a model:", ("llama3.2:1b", "llama3.1:8b"))

    # Input box for user prompt
    user_input = st.text_input("Enter your message:", key="user_input")

    if user_input:
        llm = OllamaLLM(model=model_choice)

        # Generate multiple queries
        queries = generate_multi_queries(user_input, llm)
        st.write("### Generated Queries for Retrieval")
        st.write(queries)

        # Retrieve documents
        retrieved_docs = retrieve_multi_query_docs(collection, queries) if collection else []
        fused_context = rag_fusion(retrieved_docs)

        # Construct final prompt
        final_prompt = f"Context:\n{fused_context}\nUser: {user_input}" if fused_context else user_input

        # Generate response
        response = generate_response(model_choice, final_prompt)
        st.session_state.chat_history.append(f"You: {user_input}")
        st.session_state.chat_history.append(f"Bot: {response}")

        # Display response
        st.write("### Bot Response")
        st.write(response)

    # Display chat history
    st.write("### Chat History")
    if st.session_state.chat_history:
        st.text_area("Conversation:", value="\n".join(st.session_state.chat_history), height=300, key="chat_history_display")

# Run the app
if __name__ == "__main__":
    main()
