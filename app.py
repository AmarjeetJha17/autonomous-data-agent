# app.py
import streamlit as st
import requests
import os

API_URL = "http://localhost:8000/chat"

st.set_page_config(page_title="Data Analyst Agent", layout="centered")
st.title("📊 Autonomous Data Analyst")
st.markdown("Ask questions about your e-commerce dataset in plain English.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("E.g., What is the shape of the orders table?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Send the query to our FastAPI backend
    with st.spinner("Analyzing data..."):
        try:
            response = requests.post(API_URL, json={"query": prompt})
            if response.status_code == 200:
                ai_response = response.json()["answer"]
            else:
                ai_response = f"Error: API returned status code {response.status_code}"
        except requests.exceptions.ConnectionError:
            ai_response = "Error: Could not connect to the API. Is FastAPI running?"

    # Display AI response
    with st.chat_message("assistant"):
        # Check if the AI generated a plot
        if "[PLOT_GENERATED]" in ai_response:
            # Remove the tag from the text the user sees
            clean_text = ai_response.replace("[PLOT_GENERATED]", "").strip()
            st.markdown(clean_text)
            
            # Render the image
            image_path = "outputs/current_plot.png"
            if os.path.exists(image_path):
                st.image(image_path)
        else:
            st.markdown(ai_response)
            
    # Save to history (store the raw response so the check works on reload)
    st.session_state.messages.append({"role": "assistant", "content": ai_response})