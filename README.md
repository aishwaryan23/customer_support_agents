# 🤖 Customer Support Agent with LangGraph, Streamlit & Groq

This project is an AI-powered customer support assistant that:
- Classifies queries into **Technical**, **Billing**, or **General** categories.
- Analyzes **sentiment** (positive, neutral, negative).
- Routes negative queries to **escalation**.
- Responds appropriately using a selected Groq LLM (e.g., LLaMA 3.3 70B).
- Built with **LangGraph**, **LangChain**, **Streamlit**, and **Groq**.
-
## 🚀 Features

- Query classification using LLM
- Sentiment analysis
- Conditional routing with LangGraph
- Streamlit web interface for input/output
- Scalable with modular node structure

## 📦 Installation

### 1. Clone the repository
git clone https://github.com/your-username/customer-support-agent.git
cd customer-support-agent

### 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

### 3. Install dependencies
pip install -r requirements.txt

### 4. Set up .env file
Create a .env file in the root directory:
GROQ_API_KEY=your_groq_api_key_here
