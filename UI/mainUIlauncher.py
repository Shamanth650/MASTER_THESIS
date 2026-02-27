# mainUIlauncher.py - Main entry point for ADASynAI
"""
ADASynAI - ADAS Scenario Generator
Run with: streamlit run mainUIlauncher.py
"""

import streamlit as st
from ui_utils import init_session_state, apply_custom_css, get_current_page

import screen1_standards
import screen2_upload
import screen3_info
import screen4_features
import screen5_generate
import screen5a_generate_xosc
import screen5b_generate_python
import screen6_carla_launcher

st.set_page_config(
    page_title="ADASynAI - Scenario Generator",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

apply_custom_css()
init_session_state()

def main():
    page = get_current_page()

    if page == 'standards':
        screen1_standards.show()
    elif page == 'upload':
        screen2_upload.show()
    elif page == 'info':
        screen3_info.show()
    elif page == 'features':
        screen4_features.show()
    elif page == 'generate':
        screen5_generate.show()
    elif page == 'generate_xosc':
        screen5a_generate_xosc.show()
    elif page == 'generate_python':
        screen5b_generate_python.show()
    elif page == 'carla_launcher':
        screen6_carla_launcher.show()
    else:
        st.error(f"❌ Unknown page: {page}")
        st.session_state.page = 'standards'
        st.rerun()

if __name__ == "__main__":
    main()
