import streamlit as st
import streamlit_functions as sf


st.set_page_config(
    page_title="Chat",
    page_icon="🗨️",
)


st.title("Chat")
sf.initialize_app()
sf.make_sidebar()


def clear_chat():
    st.session_state.messages = []

def run_query(query):
    results = st.session_state["VD"].query(query)
    return results


prompt = st.chat_input(
    "Ask me anything", key="h_prompt"
)  # cannot be in a tab to force positioning to bottom


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt is not None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = run_query(prompt).content
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("user").markdown(prompt)
    st.chat_message("assistant").markdown(response)
st.button("clear chat", on_click=clear_chat)
