# ui_utils.py - Shared utilities for all UI screens
"""
Common utilities, session state management, and helper functions
shared across all UI screens.
"""

from __future__ import annotations
import sys
from pathlib import Path
import base64


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import json
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# -------------------------
# Session State Management
# -------------------------
def init_session_state():
    """Initialize all session state variables"""
    
    # Page navigation
    if 'page' not in st.session_state:
        st.session_state.page = 'standards'
    
    # Standard selection
    if 'selected_standard' not in st.session_state:
        st.session_state.selected_standard = None
    
    # PDF data
    if 'pdf_file' not in st.session_state:
        st.session_state.pdf_file = None
    if 'pdf_bytes' not in st.session_state:
        st.session_state.pdf_bytes = None
    if 'pdf_name' not in st.session_state:
        st.session_state.pdf_name = None
    
    # Parsed scenarios
    if 'parsed_scenarios' not in st.session_state:
        st.session_state.parsed_scenarios = None
    if 'enriched_scenarios' not in st.session_state:
        st.session_state.enriched_scenarios = None
    
    # Selected scenarios (checkboxes)
    if 'selected_scenarios' not in st.session_state:
        st.session_state.selected_scenarios = []
    
    # Generated code
    if 'xosc_code' not in st.session_state:
        st.session_state.xosc_code = {}  # scenario_name -> code
    if 'python_code' not in st.session_state:
        st.session_state.python_code = {}  # scenario_name -> code
    
    # Provider selection
    if 'provider' not in st.session_state:
        st.session_state.provider = 'claude'
    
    # Current scenario for generation
    if 'current_scenario_idx' not in st.session_state:
        st.session_state.current_scenario_idx = 0

# -------------------------
# Navigation Functions
# -------------------------
def navigate_to(page: str):
    """Navigate to a specific page"""
    st.session_state.page = page
    st.rerun()

def get_current_page() -> str:
    """Get current page name"""
    return st.session_state.page

# -------------------------
# Progress Indicator
# -------------------------
def show_progress():
    """Show which step the user is on (with CSS hooks)"""
    pages = ['standards', 'upload', 'info', 'features', 'generate']
    page_names = ['Standards', 'Upload', 'Info', 'Features', 'Generate']

    if st.session_state.page not in pages:
        return

    current_idx = pages.index(st.session_state.page)

    st.markdown("<div class='adasyn-progress'>", unsafe_allow_html=True)

    cols = st.columns(5)
    for i, (col, page_key, name) in enumerate(zip(cols, pages, page_names)):
        with col:
            if i < current_idx:
                st.markdown(
                    f"<div class='step done' data-step='{page_key}'>⚪<span>{name}</span></div>",
                    unsafe_allow_html=True
                )
            elif i == current_idx:
                st.markdown(
                    f"<div class='step current' data-step='{page_key}'>⚪ <span>{name}</span></div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='step todo' data-step='{page_key}'>⚪ <span>{name}</span></div>",
                    unsafe_allow_html=True
                )

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")


# -------------------------
# Helper Functions
# -------------------------
def get_scenario_name(scenario: Dict[str, Any], idx: int = 0) -> str:
    """Get scenario name from scenario dict"""
    return (
        scenario.get('name') or 
        scenario.get('scenario_name') or 
        f'Scenario {idx + 1}'
    )

def get_scenario_family(scenario: Dict[str, Any]) -> str:
    """Get scenario family (AEB, LSS, VRU)"""
    classification = scenario.get('classification', {})
    return classification.get('family', 'UNKNOWN')

def get_scenario_variant(scenario: Dict[str, Any]) -> str:
    """Get scenario variant"""
    classification = scenario.get('classification', {})
    return classification.get('variant', 'unknown')

# -------------------------
# Styling
# -------------------------
def apply_custom_css():
    bg_path = Path("/home/shamanth/clean_euro/UI/assets/bg.jpg")
    top_logo = Path("/home/shamanth/clean_euro/UI/assets/log.png")
    bottom_logo = Path("/home/shamanth/clean_euro/UI/assets/logo2.png")

    def b64(p): 
        return base64.b64encode(p.read_bytes()).decode()

    bg = b64(bg_path)
    top = b64(top_logo)
    bottom = b64(bottom_logo)

    st.markdown(f"""
    <style>
    /* Hide Streamlit branding */
    header {{ visibility: hidden; height: 0px; }}
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    
    /* Background with logos INSIDE */
    .stApp {{
        background-image:
            url("data:image/png;base64,{top}"),
            url("data:image/png;base64,{bottom}"),
            url("data:image/png;base64,{bg}");
        background-repeat: no-repeat, no-repeat, no-repeat;
        background-position: 24px 18px, center calc(100% - 18px), center;
        background-size: 200px auto, 220px auto, cover;
        background-attachment: fixed, fixed, fixed;
    }}

    .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;
        background: rgba(0,0,0,0.22);
        z-index: 0;
        pointer-events: none;
    }}

    section.main {{
        position: relative;
        z-index: 1;
    }}
    </style>
    """, unsafe_allow_html=True)

# -------------------------
# RAG Generation Functions
# -------------------------
def get_rag_functions():
    """Get RAG generation functions"""
    try:
        from RAG2.generators.orchestrator import generate_scenario_artifacts

        def generate_python(scenario, *, k=None, provider="claude"):
            return generate_scenario_artifacts(scenario, k=k, provider=provider)["carla_py"]

        def generate_xosc(scenario, *, k=None, provider="claude"):
            return generate_scenario_artifacts(scenario, k=k, provider=provider)["xosc"]
        
        return generate_python, generate_xosc

    except Exception as e:
        st.error(f"Failed to import RAG modules: {e}")
        return None, None

# -------------------------
# Parser Functions
# -------------------------
def get_parser_function():
    """Get PDF parser function"""
    try:
        from PARSER.main_parser import run_full_pipeline_for_pdf
        return run_full_pipeline_for_pdf
    except Exception as e:
        st.error(f"Failed to import Parser: {e}")
        return None