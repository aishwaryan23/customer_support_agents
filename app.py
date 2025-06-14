
import streamlit as st
from typing import TypedDict, Dict
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Define the state
class State(TypedDict):
    query: str
    category: str
    sentiment: str
    response: str

# Set up LLM
import os
llm = ChatGroq(
    temperature=0,
    export groq_api_key= os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

# Define nodes
def categorize(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Categorize the following customer query into one of these categories: Technical, Billing, General. Query: {query}"
    )
    chain = prompt | llm
    category = chain.invoke({"query": state["query"]}).content.strip()
    return {"category": category}

def analyze_sentiment(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Analyze the sentiment of the following customer query. Respond with a single word: positive, neutral, or negative. Query: {query}"
    )
    chain = prompt | llm
    sentiment = chain.invoke({"query": state["query"]}).content.strip()
    return {"sentiment": sentiment}

def handle_technical(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Provide a technical support response to the following query: {query}"
    )
    chain = prompt | llm
    response = chain.invoke({"query": state["query"]}).content
    return {"response": response}

def handle_billing(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Provide a billing support response to the following query: {query}"
    )
    chain = prompt | llm
    response = chain.invoke({"query": state["query"]}).content
    return {"response": response}

def handle_general(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Provide a general support response to the following query: {query}"
    )
    chain = prompt | llm
    response = chain.invoke({"query": state["query"]}).content
    return {"response": response}

def escalate(state: State) -> State:
    return {"response": "This query has been escalated to a human agent due to its negative sentiment."}

def route_query(state: State) -> str:
    if state["sentiment"].lower() == "negative":
        return "escalate"
    elif state["category"] == "Technical":
        return "handle_technical"
    elif state["category"] == "Billing":
        return "handle_billing"
    else:
        return "handle_general"

# Streamlit UI
st.set_page_config(page_title="Customer Support Assistant", layout="centered")
st.title("🤖 Customer Support Assistant")

user_query = st.text_area("Enter your support query:", height=100)

if st.button("Submit"):
    if user_query.strip() == "":
        st.warning("Please enter a query.")
    else:
        with st.spinner("Processing..."):

            # Build and compile the workflow INSIDE the button callback
            workflow = StateGraph(State)
            workflow.add_node("categorize", categorize)
            workflow.add_node("analyze_sentiment", analyze_sentiment)
            workflow.add_node("handle_technical", handle_technical)
            workflow.add_node("handle_billing", handle_billing)
            workflow.add_node("handle_general", handle_general)
            workflow.add_node("escalate", escalate)

            workflow.add_edge("categorize", "analyze_sentiment")
            workflow.add_conditional_edges("analyze_sentiment", route_query, {
                "handle_technical": "handle_technical",
                "handle_billing": "handle_billing",
                "handle_general": "handle_general",
                "escalate": "escalate"
            })

            workflow.add_edge("handle_technical", END)
            workflow.add_edge("handle_billing", END)
            workflow.add_edge("handle_general", END)
            workflow.add_edge("escalate", END)
            workflow.set_entry_point("categorize")

            app = workflow.compile()
            result = app.invoke({"query": user_query})

            st.success("Response generated!")

            st.markdown("### 🔍 Analysis")
            st.write(f"**Category:** {result['category']}")
            st.write(f"**Sentiment:** {result['sentiment']}")

            st.markdown("### 💬 Response")
            st.write(result["response"])
