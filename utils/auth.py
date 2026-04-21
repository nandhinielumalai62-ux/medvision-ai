import streamlit as st

def login():
    if "login" not in st.session_state:
        st.session_state.login = False

    if not st.session_state.login:
        st.markdown("""
            <style>
            .stButton>button { width: 100%; border-radius: 5px; background-color: #007bff; color: white; }
            </style>
        """, unsafe_allow_html=True)
        
        st.title("🏥 MedVision Hospital Login")
        u = st.text_input("Staff ID")
        p = st.text_input("Security PIN", type="password")
        
        if st.button("Authorize Access"):
            if u == "admin" and p == "1234":
                st.session_state.login = True
                st.rerun()
            else:
                st.error("Invalid Credentials")
        st.stop()