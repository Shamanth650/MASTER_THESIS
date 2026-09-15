# screen1_standards.py
# ------------------------------------------------------------
# Responsible for: Rendering the first screen of the wizard, where the
# engineer picks which safety-testing standard (Euro NCAP, China NCAP,
# Japan NCAP, California NCAP) to generate scenarios for, and advances to
# the upload screen once Euro NCAP (the only currently supported standard) is chosen.
# Maintainer: shamanth.adiga@ltts.com
# ------------------------------------------------------------

import streamlit as st
from ui_utils import navigate_to

# Renders the standards-selection screen: styled title, a 2x2 grid of
# standard buttons, and a success message confirming the current selection.
def show():
    """Display standards selection screen"""

    # ---------- PAGE-SPECIFIC STYLING ----------
    st.markdown("""
    <style>
        /* Title */
        .standards-title {
            text-align: center;
            font-size: 56px;
            font-weight: 800;
            letter-spacing: 1px;
            margin-bottom: 40px;
            color: rgba(255,255,255,0.95);
        }

        /* Center container */
        .standards-container {
            max-width: 1000px;
            margin: 0 auto;
        }

        /* Button styling */
        div.stButton > button {
            height: 80px;
            font-size: 58px !important;
            font-weight: 700 !important;
            border-radius: 16px;
        }
    </style>
    """, unsafe_allow_html=True)

    # ---------- TITLE ----------
    st.markdown("<div class='standards-title'>STANDARDS</div>", unsafe_allow_html=True)

    # ---------- CENTERED BUTTON GRID ----------
    st.markdown("<div class='standards-container'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    col3, col4 = st.columns(2, gap="large")

    with col1:
        if st.button("EURO NCAP", use_container_width=True, key="euro"):
            st.session_state.selected_standard = "EURO NCAP"
            navigate_to("upload")

    with col2:
        if st.button("CHINA NCAP", use_container_width=True, key="china"):
            st.session_state.selected_standard = "CHINA NCAP"
            st.info("China NCAP support coming soon!")

    with col3:
        if st.button("JAPAN NCAP", use_container_width=True, key="japan"):
            st.session_state.selected_standard = "JAPAN NCAP"
            st.info("Japan NCAP support coming soon!")

    with col4:
        if st.button("CALIFORNIA NCAP", use_container_width=True, key="california"):
            st.session_state.selected_standard = "CALIFORNIA NCAP"
            st.info("California NCAP support coming soon!")

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- SELECTION FEEDBACK ----------
    if st.session_state.get("selected_standard"):
        st.markdown("<br>", unsafe_allow_html=True)
        st.success(f"**Selected:** {st.session_state.selected_standard}")
