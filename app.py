import streamlit as st

tabs = {
    "Desciption": [
        st.Page("pages/description/description.py", title="Desciption"),
    ],
    "Demo": [
        st.Page("pages/demo.py", title="Helmet Detector Demo"),
    ],
}

def aSideBar():
    st.navigation(tabs).run()
    
st.set_page_config(page_title="Helmet Detection Project", page_icon="🏍️", layout="wide")
aSideBar()