# screen5a_generate_xosc.py
import streamlit as st
from ui_utils import navigate_to, show_progress, get_rag_functions

def show():
    show_progress()

    provider = "claude"  # DEMO default
    current_scenario = st.session_state.get("current_scenario_for_generation")
    scenario_name = st.session_state.get("current_scenario_name", "scenario")
    filename = f"{scenario_name}.xosc"

    st.markdown(f"<h1>Generating: {filename}</h1>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    generate_python, generate_xosc = get_rag_functions()

    if not generate_xosc:
        st.error("❌ XOSC generation function not available!")
        return

    # Auto-generate once when arriving
    if st.session_state.get("auto_generate") == "xosc" and scenario_name not in st.session_state.xosc_code:
        with st.spinner(f"Generating XOSC"):
            try:
                xosc_code = generate_xosc(current_scenario, provider=provider)
                st.session_state.xosc_code[scenario_name] = xosc_code
                st.success("✅ XOSC generated!")
            except Exception as e:
                st.error(f"❌ Generation failed: {e}")
        st.session_state.auto_generate = None

    # Show code + download
    if scenario_name in st.session_state.xosc_code:
        st.code(st.session_state.xosc_code[scenario_name], language="xml", line_numbers=True)

        st.download_button(
            "📥 DOWNLOAD XOSC",
            st.session_state.xosc_code[scenario_name],
            file_name=filename,
            mime="application/xml",
            use_container_width=True
        )

        # Auto-save xosc to scenario_runner examples folder
        import os
        xosc_save_path = os.path.expanduser(f"~/scenario_runner/srunner/examples/{filename}")
        try:
            with open(xosc_save_path, "w") as f:
                f.write(st.session_state.xosc_code[scenario_name])
            st.session_state["generated_xosc_path"] = xosc_save_path
        except Exception:
            pass

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🚀 Launch in CARLA", type="primary", use_container_width=True):
            navigate_to("carla_launcher")

    else:
        if st.button("Generate XOSC", use_container_width=True):
            st.session_state.auto_generate = "xosc"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Navigation row 1: back options
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("← Back to Code Generation", use_container_width=True):
            navigate_to("generate")
    with c2:
        if st.button("Generate Python Instead →", use_container_width=True):
            st.session_state.auto_generate = "python"
            navigate_to("generate_python")

    st.markdown("<br>", unsafe_allow_html=True)

    # Navigation row 2: scenario selection shortcut
    c3, c4 = st.columns([1, 1])
    with c3:
        if st.button("← Back to Scenario Selection", use_container_width=True):
            navigate_to("features")
    with c4:
        if st.button("🔄 Start Over", use_container_width=True):
            navigate_to("standards")
