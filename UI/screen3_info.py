# screen3_info.py - Standard Information Screen
"""
SCREEN 3: Standard Information
Shows brief info about the standard and PDF preview
"""

import streamlit as st
from ui_utils import navigate_to, show_progress, get_scenario_name

def show():
    """Display PDF upload screen"""

    # Upload-screen-only progress styling
    st.markdown("""
    <style>
      /* Push progress bar down so it doesn't collide with the top-left logo */
      .adasyn-progress{
        margin-top: 64px;
      }

      /* Base size for all steps */
      .adasyn-progress .step span{
        font-size: 14px;
        font-weight: 600;
        opacity: 0.85;
      }

      /* Make the CURRENT step bigger (in general) */
      .adasyn-progress .step.current span{
        font-size: 16px;
        font-weight: 800;
        opacity: 1;
      }

      /* For THIS screen: make UPLOAD bigger than the rest */
      .adasyn-progress .step[data-step="upload"] span{
        font-size: 20px !important;
        font-weight: 900 !important;
        opacity: 1 !important;
      }
    </style>
    """, unsafe_allow_html=True)

    show_progress()
    
    # Title
    st.markdown("<h1 style='text-align: center;'>STANDARD INFORMATION</h1>", unsafe_allow_html=True)
    
    col1, spacer ,col2 = st.columns([3,1,2])
    
    # ===== LEFT COLUMN: Brief Info =====
    with col1:
        st.markdown("### BRIEF INFO ABOUT THE STANDARD")
        
        if st.session_state.parsed_scenarios:
            st.write(f"""
            **Standard:** {st.session_state.selected_standard}
            
            **Document:** {st.session_state.pdf_name}
            
            **Total Scenarios:** {len(st.session_state.parsed_scenarios)}
            
            **File Size:** {len(st.session_state.pdf_bytes) / 1024:.1f} KB
            
            """)
            
            # Show scenario list
            st.markdown("**Detected Scenarios:**")
            for i, scenario in enumerate(st.session_state.parsed_scenarios[:10]):
                name = get_scenario_name(scenario, i)
                st.write(f"{i+1}. {name}")
            
            if len(st.session_state.parsed_scenarios) > 10:
                st.write(f"... and {len(st.session_state.parsed_scenarios) - 10} more")
    
    # ===== RIGHT COLUMN: PDF Preview =====
    with col2:
        st.markdown("### PROTOCAL SNAPSHOT")
        
        try:
            from pdf2image import convert_from_bytes
            
            with st.spinner("Rendering PDF preview..."):
                images = convert_from_bytes(
                    st.session_state.pdf_bytes,
                    first_page=1,
                    last_page=1,
                    dpi=150
                )
                
                if images:
                    st.image(images[0], width=400, caption="First page")
                else:
                    st.warning("Could not render PDF preview")
                    
        except ImportError:
            st.warning("⚠️ PDF preview not available. Install pdf2image: `pip install pdf2image`")
            st.info("💡 You can still proceed without preview")
        except Exception as e:
            st.error(f"Error rendering PDF: {e}")
            st.info("💡 You can still proceed")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("← Back", use_container_width=True):
            navigate_to('upload')
    
    with col3:
        if st.button("Next →", use_container_width=True):
            navigate_to('features')