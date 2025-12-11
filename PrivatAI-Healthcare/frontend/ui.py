import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="PrivatAI: Hospital", layout="wide")

st.title("🏥 PrivatAI: Hospital Knowledge Base")

# --- Sidebar: User Simulation ---
with st.sidebar:
    st.header("🔐 Secure Login Simulation")
    # In a real app, this would be a password field
    user_role = st.selectbox(
        "Select Your Department (Role)",
        ["Cardiology", "Neurology", "Administration", "Oncology"]
    )
    st.info(f"You are now logged in as: **{user_role}**")
    st.markdown("---")
    st.subheader("Upload Knowledge (Admin Only)")
    
    uploaded_file = st.file_uploader("Upload Department Guidelines (PDF/TXT)")
    if st.button("Ingest Document") and uploaded_file:
        with st.spinner("Encrypting and Vectorizing..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            data = {"department": user_role}
            try:
                res = requests.post(f"{API_URL}/upload/", files=files, data=data)
                if res.status_code == 200:
                    st.success(res.json().get("message"))
                else:
                    st.error("Upload failed")
            except Exception as e:
                st.error(f"Connection Error: {e}")

# --- Main Chat Area ---
st.subheader(f"💬 Chat with {user_role} Data")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input(f"Ask about {user_role} protocols..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consulting secure internal records..."):
            try:
                # Send query to backend with the User's Department
                payload = {"query": prompt, "department": user_role}
                response = requests.post(f"{API_URL}/chat/", data=payload)
                if response.status_code == 200:
                    bot_reply = response.json().get("answer")
                else:
                    bot_reply = "Error reaching the secure server."
            except:
                bot_reply = "Server connection failed. Is the Backend running?"
            
            st.markdown(bot_reply)
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})
