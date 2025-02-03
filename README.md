# AI Assistant for the Constitution of Kazakhstan

This project is an **AI-powered chatbot** designed to answer questions specifically related to the **Constitution of the Republic of Kazakhstan**. The assistant provides **accurate interpretations** and **citations of relevant articles and sections** from the official English version of the Constitution. 

The chatbot is built using **Streamlit**, **Ollama**, and **ChromaDB**, incorporating **Multi-Query Retrieval (MQR)** and **RAG Fusion** to enhance response quality.

## Features
- **Legal Document Interpretation**: Provides precise answers based on the Constitution of Kazakhstan.
- **Multi-Query Retrieval (MQR)**: Generates multiple variations of a user query to retrieve the most relevant information.
- **RAG Fusion**: Combines retrieved data from multiple queries for a more comprehensive response.
- **Contextual Memory**: Uses **ChromaDB** to store and retrieve conversation history.
- **Multiple Model Support**: Users can select between **llama3.1:8b** and **llama3.2:1b** models.
- **User-Friendly Interface**: Built with **Streamlit** for an intuitive chat experience.

## Installation Guide

### Prerequisites
- **Python 3.8 or later**
- **Git** (optional, for cloning the repository)
- **Ollama** installed ([Installation Guide](https://ollama.ai/))
- **ChromaDB**
- **Langchain**
- **Streamlit**

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

