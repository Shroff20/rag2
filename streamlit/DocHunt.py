import sys
sys.path.append("../source/")
import streamlit as st
import database
import streamlit_functions as sf


st.set_page_config(
    page_title="Hello",
    page_icon="🐦",
)






st.title("Main Page")


sf.initialize_app()
sf.make_sidebar()





