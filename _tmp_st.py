import streamlit as st


h = st.file_uploader('test1', accept_multiple_files=True)

print(type(h))
print(h)

print('deleting')
print(len(h))
h = []

print(h)
print(len(h))


with st.form("my-form", clear_on_submit=True):
        file = st.file_uploader("FILE UPLOADER")
        submitted = st.form_submit_button("process")