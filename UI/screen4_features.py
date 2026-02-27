# screen4_features.py - Features and Report Screen (UPDATED WITH REPORT DOWNLOAD!)
"""
SCREEN 4: Features Selection and Report Generation
User selects which scenarios to generate code for
Can download parsing accuracy report (automatically generated during parsing)
"""

import streamlit as st
from pathlib import Path
from ui_utils import navigate_to, show_progress, get_scenario_name

def show():
    """Display features selection and report generation screen"""
    
    show_progress()
    
    # Title
    st.markdown("<h1>FEATURES IN THIS PDF</h1>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ===== REPORT GENERATION =====
    st.markdown("### 📊 Parsing Accuracy Report")
    
    st.write("Download a detailed analysis of the extraction quality:")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Look for the report in common locations
    possible_report_paths = [
        Path("/home/shamanth/clean_euro/PARSER/Euro_NCAP_Scenario_Analysis_Report.pdf"),
        Path("Parsed_Data/Euro_NCAP_Scenario_Analysis_Report.pdf"),
        Path("Euro_NCAP_Scenario_Analysis_Report.pdf"),
        Path("PARSER/Parsed_Data/Euro_NCAP_Scenario_Analysis_Report.pdf"),
        Path("../Parsed_Data/Euro_NCAP_Scenario_Analysis_Report.pdf"),
    ]
    
    report_path = None
    for path in possible_report_paths:
        if path.exists():
            report_path = path
            break
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if report_path and report_path.exists():
            # Report exists - show download button
            with open(report_path, "rb") as f:
                report_bytes = f.read()
            
            st.download_button(
                label="📥 Download the parsing accuracy report",
                data=report_bytes,
                file_name="Euro_NCAP_Scenario_Analysis_Report.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="download_report"
            )
            
            st.success(f"✅ Report ready! ({len(report_bytes) / 1024:.1f} KB)")
            
        else:
            # Report doesn't exist - show placeholder
            if st.button(
                "📥 Download the parsing accuracy report",
                use_container_width=True,
                disabled=True,
                key="report_placeholder"
            ):
                pass
            
            st.warning("⚠️ Report not found. It will be generated automatically during PDF parsing.")
            
            # Show where we looked
            #with st.expander("🔍 Debug: Report search paths"):
             #   st.write("Searched for report in:")
              #  for path in possible_report_paths:
               #     exists_icon = "✅" if path.exists() else "❌"
                #    st.write(f"{exists_icon} {path}")
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Info box about the report
    with st.expander("ℹ️ About the Parsing Accuracy Report"):
        st.markdown("""
        The **Parsing Accuracy Report** is a professional PDF document that contains:
        
        -  **Overall extraction quality grade** (A-F scale)
        -  **Extraction accuracy percentage** (0-100%)
        -  **Detailed scenario-by-scenario analysis**
        -  **Parameter correctness verification** (speeds, distances, overlaps)
        
        """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ===== SCENARIO SELECTION =====
    st.markdown("### ✅ Select Scenarios")
    st.write("Choose which scenarios you want to generate code for:")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.session_state.parsed_scenarios:
        # Select all / Deselect all
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            if st.button("✅ Select All"):
                st.session_state.selected_scenarios = st.session_state.parsed_scenarios.copy()
                st.rerun()
        
        with col2:
            if st.button("❌ Deselect All"):
                st.session_state.selected_scenarios = []
                st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Show scenarios in a nice table format
        st.markdown("**Available Scenarios:**")
        
        # Checkboxes for each scenario
        for i, scenario in enumerate(st.session_state.parsed_scenarios):
            name = get_scenario_name(scenario, i)
            
            # Get variant if available
            variant = scenario.get('classification', {}).get('variant', '')
            family = scenario.get('classification', {}).get('family', 'Unknown')
            
            # Build display name
            if variant:
                display_name = f"{i+1}. {name} ({variant} - {family})"
            else:
                display_name = f"{i+1}. {name} ({family})"
            
            is_selected = scenario in st.session_state.selected_scenarios
            
            if st.checkbox(
                display_name,
                value=is_selected,
                key=f"cb_{i}"
            ):
                if scenario not in st.session_state.selected_scenarios:
                    st.session_state.selected_scenarios.append(scenario)
            else:
                if scenario in st.session_state.selected_scenarios:
                    st.session_state.selected_scenarios.remove(scenario)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Selection count with better formatting
        selected_count = len(st.session_state.selected_scenarios)
        total_count = len(st.session_state.parsed_scenarios)
        
        if selected_count == 0:
            st.error(f"❌ No scenarios selected (0 / {total_count})")
        elif selected_count == total_count:
            st.success(f"✅ All scenarios selected ({selected_count} / {total_count})")
        else:
            st.info(f"✅ Selected: {selected_count} / {total_count} scenarios")
    
    else:
        st.error("❌ No scenarios available")
        st.info("Please go back and upload a PDF to parse scenarios.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("← Back", use_container_width=True):
            navigate_to('info')
    
    with col3:
        next_disabled = len(st.session_state.selected_scenarios) == 0
        
        if st.button(
            "Next →",
            use_container_width=True,
            disabled=next_disabled,
            help="Select at least one scenario to continue" if next_disabled else "Proceed to code generation"
        ):
            navigate_to('generate')