# screen5b_generate_python.py
import streamlit as st
from ui_utils import navigate_to, show_progress, get_rag_functions

def show():
    show_progress()

    provider = "claude"  # DEMO default

    current_scenario = st.session_state.get("current_scenario_for_generation")
    scenario_name = st.session_state.get("current_scenario_name", "scenario")

    filename = f"{scenario_name}.py"

    st.markdown(f"<h1>Generating: {filename}</h1>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    generate_python, generate_xosc = get_rag_functions()
    if not generate_python:
        st.error("❌ Python generation function not available!")
        return

    # Auto-generate once when arriving
    if st.session_state.get("auto_generate") == "python" and scenario_name not in st.session_state.python_code:
        with st.spinner(f"Generating Python"):
            try:
                py_code = generate_python(current_scenario, provider=provider)
                st.session_state.python_code[scenario_name] = py_code
                st.success("✅ Python generated!")
            except Exception as e:
                st.error(f"❌ Generation failed: {e}")
        st.session_state.auto_generate = None

    # Show code + download
    if scenario_name in st.session_state.python_code:
        st.code(st.session_state.python_code[scenario_name], language="python", line_numbers=True)
        st.download_button(
            "📥 DOWNLOAD PYTHON",
            st.session_state.python_code[scenario_name],
            file_name=filename,
            mime="text/x-python",
            use_container_width=True
        )
    else:
        if st.button("Generate Python", use_container_width=True):
            st.session_state.auto_generate = "python"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("← Back to Code Generation", use_container_width=True):
            navigate_to("generate")
    with c2:
        if st.button("Generate XOSC Instead →", use_container_width=True):
            st.session_state.auto_generate = "xosc"
            navigate_to("generate_xosc")
