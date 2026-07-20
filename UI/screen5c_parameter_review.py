# screen5c_parameter_review.py - Parameter Override Screen
"""
SCREEN 5c: Review and Override Extracted Parameters
Shown after the user chooses to generate XOSC or Python for a scenario,
before generation actually runs.

Every field the extraction pipeline populated as non-null for this
scenario is shown here, for full transparency. Fields marked editable
can be changed via dropdown and are applied to the generated XOSC.
Fields not yet verified as safe to apply are shown read-only, so the
engineer can still see and confirm exactly what the protocol specifies,
without the tool pretending it can safely act on every one of them yet.
"""

import streamlit as st
from ui_utils import navigate_to, show_progress
from parameter_overrides import get_available_overrides


def show():
    show_progress()

    current_scenario = st.session_state.get("current_scenario_for_generation")
    scenario_name = st.session_state.get("current_scenario_name", "scenario")
    target = st.session_state.get("pending_generation_target", "xosc")

    st.markdown("<h1>REVIEW EXTRACTED PARAMETERS</h1>", unsafe_allow_html=True)
    st.markdown(f"### Scenario: **{scenario_name}**")
    st.markdown("<br>", unsafe_allow_html=True)

    if current_scenario is None:
        st.error("❌ No scenario selected. Please go back and choose one.")
        if st.button("← Back to Code Generation"):
            navigate_to("generate")
        return

    overrides_available = get_available_overrides(current_scenario)

    if "parameter_overrides" not in st.session_state:
        st.session_state.parameter_overrides = {}
    current_selections = st.session_state.parameter_overrides.get(scenario_name, {})

    editable_fields = {k: v for k, v in overrides_available.items() if v.get("editable")
                        and v["source"] == "protocol"}
    display_only_fields = {k: v for k, v in overrides_available.items()
                            if v["source"] == "protocol" and not v.get("editable")}
    category_fields = {k: v for k, v in overrides_available.items() if v["source"] == "category"}

    new_selections = {}

    # ----- Editable protocol fields -----
    st.markdown("#### ✏️ Adjustable parameters (extracted from the protocol)")
    if editable_fields:
        for field, spec in editable_fields.items():
            options = spec["options"]
            default_val = current_selections.get(field, spec["default"])
            try:
                default_idx = options.index(default_val)
            except ValueError:
                default_idx = 0
            chosen = st.selectbox(
                spec["label"], options, index=default_idx, key=f"override_{scenario_name}_{field}"
            )
            new_selections[field] = chosen
    else:
        st.info("No adjustable numeric fields were extracted as non-null for this scenario.")

    # ----- Display-only protocol fields -----
    if display_only_fields:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📋 Extracted from the protocol")
        for field, spec in display_only_fields.items():
            st.write(f"**{spec['label']}:** {spec['default']}")

    # ----- Actor selection (category-derived, editable) -----
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🚗 Actor selection")
    
    for field, spec in category_fields.items():
        options = spec["options"]
        default_val = current_selections.get(field, spec["default"])
        try:
            default_idx = options.index(default_val)
        except ValueError:
            default_idx = 0
        chosen = st.selectbox(
            spec["label"], options, index=default_idx, key=f"override_{scenario_name}_{field}"
        )
        new_selections[field] = chosen

    st.session_state.parameter_overrides[scenario_name] = new_selections

    st.markdown("<br><br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Back to Code Generation", use_container_width=True):
            navigate_to("generate")
    with col3:
        if st.button("Generate →", type="primary", use_container_width=True):
            st.session_state.auto_generate = target
            navigate_to("generate_xosc" if target == "xosc" else "generate_python")