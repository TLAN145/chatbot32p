# Ollama Chatbot with ChromaDB Memory
This project demonstrates how to build a chatbot using Ollama's LLM (Language Model) and ChromaDB for maintaining contextual memory. The chatbot remembers previous interactions and leverages past conversation data to provide relevant and accurate responses. The app is built using Streamlit for the frontend interface. This updated version allows for reading multiple `.txt` files for additional context, offering more adaptability and responsiveness in the conversations. Bot can answer to the questions about Constitution of Kazakhstan. 

## Features
- **Contextual Memory**: Uses ChromaDB to store and retrieve conversation history and context.
- **Language Model Interaction**: Interacts with the `OllamaLLM` for generating responses.
- **File Uploads**: Upload and process multiple `.txt` files to provide additional context during interactions.
- **Streamlit Interface**: A user-friendly interface for easy interaction and viewing conversation history.
- **Model Selection**: Choose between different Ollama models via the sidebar.
- **Chat History**: The chatbot remembers previous exchanges for more coherent and contextual conversations.
- **Clear Chat History**: Clears both the session's chat history and ChromaDB's stored data, allowing for fresh interactions.

## Requirements
- Python 3.7 or higher
- Streamlit
- Langchain (Ollama integration)
- ChromaDB
- OpenAI API Key (for embeddings in ChromaDB)

## Usage
- **Interact with the chatbot** via the text input box. The chatbot retains context from past conversations and can process multiple `.txt` files to enhance its responses.
- **Upload `.txt` files** to provide additional context for your queries, making the chatbot's responses more informed.

### Examples

**File Upload and Contextual Responses**:
- Upload a document on a specific topic (e.g., "Climate Change").
- Ask the chatbot questions based on the document: 
  - "What are the main causes of climate change?"

**Contextual Memory with ChromaDB**:
The chatbot saves and uses past conversations for better responses:
- **User**: "Tell me about AI."
- **Bot**: "AI stands for Artificial Intelligence..."
- **User**: "And how does it relate to ChatGPT?"
- **Bot**: "Building on our earlier discussion about AI..."

**Model Examples**:

- **llama3.2:1b**:
  - **You**: Can you help me with programming?
  - **Bot**: I'd be happy to help you with programming. What kind of programming are you working on, or what problem do you need help with? Let me know how I can assist you.

- **llama3.1:8b**:
  - **You**: Can you help me with programming?
  - **Bot**: I'd be happy to help with your programming-related questions. What type of programming are you working on (e.g., web development, mobile app development, machine learning, etc.) and what specific issue or task do you need assistance with?
 
  - **You**: Tell me about Article 1 of Constitution of Kazakhstan
  - **Bot(llama3.2:1b)**: The Article 1 of the Constitution of Kazakhstan reads: "Kazakhstan is an independent, democratic state which is a sovereign nation-state. The fundamental right and duty of every citizen of Kazakhstan are to respect and defend this State, its territorial integrity and sovereignty." This article sets out the basic principles and goals of the state in Kazakhstan, emphasizing its independence, democracy, and sovereignty. It also recognizes the rights and duties of citizens, including their responsibility to respect and defend the state. In essence, Article 1 of the Constitution of Kazakhstan establishes that Kazakhstan is a sovereign nation-state with democratic values and principles at its core.

## Installation Guide

Follow these steps to set up and run the project on your local machine:

Prerequisites:
- Python 3.8 or later installed on your system.
- A Git client (optional, for cloning the repository).

Steps:

1. Clone the Repository
Clone the project repository to your local machine:

git clone https://github.com/your-repository-url.git
cd your-repository-folder

2. Create a Virtual Environment
Create and activate a Python virtual environment to isolate dependencies:

- For Linux/Mac:
  python3 -m venv venv
  source venv/bin/activate

- For Windows:
  python -m venv venv
  venv\Scripts\activate

3. Install Dependencies
Use pip to install the required Python packages:

pip install -r requirements.txt


### Prerequisites
- Python 3.8 or later installed on your system.
- A Git client (optional, for cloning the repository).


1. **Clone the Repository**:
[https://github.com/your-repo/ollama-chatbot.git](https://github.com/TLAN145/chatbotv3.git)

