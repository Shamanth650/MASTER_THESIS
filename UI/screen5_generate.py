# screen5_generate.py

import streamlit as st
from ui_utils import navigate_to, show_progress, get_scenario_name

def show():
    show_progress()

    st.markdown("<h1>CODE GENERATION</h1>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if not st.session_state.selected_scenarios:
        st.error("❌ No scenarios selected!")
        if st.button("← Back to Features"):
            navigate_to('features')
        return

    # Scenario selector (keep as-is)
    if len(st.session_state.selected_scenarios) > 1:
        st.markdown("### 📋 Select Scenario")
        scenario_names = [
            get_scenario_name(s, i)
            for i, s in enumerate(st.session_state.selected_scenarios)
        ]
        selected_idx = st.selectbox(
            "Choose which scenario to generate code for:",
            range(len(scenario_names)),
            format_func=lambda i: f"{i+1}. {scenario_names[i]}"
        )
        current_scenario = st.session_state.selected_scenarios[selected_idx]
    else:
        current_scenario = st.session_state.selected_scenarios[0]

    scenario_name = get_scenario_name(current_scenario)

    # DEMO DEFAULT: force cloud (claude) without showing UI
    st.session_state.provider = "claude"

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"### Selected Scenario: **{scenario_name}**")

    st.markdown("<br>", unsafe_allow_html=True)

    # Two buttons: redirect to parameter review before generation
    col1, col2 = st.columns(2, gap="large")

    with col1:
        if st.button("Generate XOSC →", use_container_width=True, key="go_xosc"):
            st.session_state.current_scenario_for_generation = current_scenario
            st.session_state.current_scenario_name = scenario_name
            st.session_state.pending_generation_target = "xosc"
            navigate_to("parameter_review")

    with col2:
        if st.button("Generate Python →", use_container_width=True, key="go_py"):
            st.session_state.current_scenario_for_generation = current_scenario
            st.session_state.current_scenario_name = scenario_name
            st.session_state.pending_generation_target = "python"
            navigate_to("parameter_review")

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Navigation
    nav1, nav2, nav3 = st.columns([1, 1, 1])
    with nav1:
        if st.button("← Back", use_container_width=True):
            navigate_to('features')
    with nav3:
        if st.button("🔄 Start Over", use_container_width=True):
            navigate_to('standards')
