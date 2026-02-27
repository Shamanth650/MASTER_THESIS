# screen2_upload.py - PDF Upload Screen (CORRECTED FOR YOUR PARSER)
"""
SCREEN 2: PDF Upload
User uploads the test protocol PDF document
"""

import streamlit as st
from ui_utils import navigate_to, show_progress, get_parser_function

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
    st.markdown("<h1 style='text-align: center;'>UPLOAD YOUR DOCUMENT HERE</h1>", unsafe_allow_html=True)
    
    st.markdown(
        f"<h3 style='text-align: center;'>Selected Standard: {st.session_state.selected_standard}</h3>", 
        unsafe_allow_html=True
    )
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Center the file uploader
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        uploaded_file = st.file_uploader(
            "Browse for PDF files",
            type=["pdf"],
            help="Upload Euro NCAP test protocol PDF",
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            st.session_state.pdf_file = uploaded_file
            st.session_state.pdf_bytes = uploaded_file.read()
            st.session_state.pdf_name = uploaded_file.name
            
            # Show file info
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            st.info(f"📄 Size: {len(st.session_state.pdf_bytes) / 1024:.1f} KB")
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("← Back", use_container_width=True):
            navigate_to('standards')
    
    with col3:
        if st.button("Next →", use_container_width=True, disabled=not st.session_state.pdf_file):
            if st.session_state.pdf_file:
                # Parse PDF
                with st.spinner("🔄 Parsing PDF document... This may take 4-5 minutes."):
                    try:
                        # Get parser function
                        parser = get_parser_function()
                        if not parser:
                            st.error("Parser not available!")
                            return
                        
                        # Run parser with YOUR signature (pdf_bytes, filename, use_llm_parser)
                        result = parser(
                            pdf_bytes=st.session_state.pdf_bytes,  # ← Bytes
                            filename=st.session_state.pdf_name,    # ← Filename
                            use_llm_parser=True                     # ← Enable LLM enrichment
                        )
                        
                        st.session_state.parsed_scenarios = result
                        
                        st.success(f"✅ Successfully parsed {len(result)} scenarios!")
                        st.balloons()
                        navigate_to('info')
                        
                    except Exception as e:
                        st.error(f"❌ Failed to parse PDF: {e}")
                        st.error("Please check the PDF format and try again.")
                        
                        # Show detailed error for debugging
                        with st.expander("🔍 Debug Info"):
                            import traceback
                            st.code(traceback.format_exc())