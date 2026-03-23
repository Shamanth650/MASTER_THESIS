# ADASynAI – ADAS Scenario Generator for CARLA

ADASynAI is an AI-based tool that automatically extracts ADAS validation scenarios from Euro NCAP protocol PDFs and generates executable scenarios for CARLA ScenarioRunner.

---

##  Overview

The system performs an end-to-end pipeline:

-  Parse Euro NCAP protocol PDFs  
-  Extract structured scenario data  
-  Enrich missing parameters using LLM + RAG  
-  Generate OpenSCENARIO (`.xosc`) files  
-  Execute scenarios in CARLA  

---
##  Project Structure

```bash
clean_euro/
│
├── EuroNcap/                      # Input PDFs
│
├── PARSER/                       # PDF parsing + scenario extraction
│   ├── Parsed_Data/              # Extracted images/text
│   ├── main_parser.py            # Main parsing pipeline
│   ├── stage2_build_structured_and_evidence.py
│   ├── scenario_extractor.py
│   ├── llm_enricher.py
│   ├── report_generator.py
│   
├── RAG2/                         # LLM + RAG pipeline
│   ├── chroma_ncap_1536/         # Vector DB (Chroma)
│   ├── generators/
│   ├── knowledge_base/
│   ├── prompts/
│   ├── llm_client.py
│   ├── embeddings.py
│   ├── scenario_utils.py
│   └── xosc_builder.py
│
├── UI/                           # Streamlit frontend
│   ├── assets/
│   ├── mainUIlauncher.py         # Entry point
│   ├── screen1_standards.py
│   ├── screen2_upload.py
│   ├── screen3_info.py
│   ├── screen4_features.py
│   ├── screen5_generate.py
│   ├── screen5a_generate_xosc.py
│   ├── screen5b_generate_python.py
│   ├── screen6_carla_launcher.py
│   └── ui_utils.py
---
```
## Setup (Linux)

### 1. Clone the Repository

git clone <your-repo-link>
cd <your-repo-folder>

### 2. Create Virtual Environment
python3 -m venv venv
source venv/bin/activate

### 3. Install ScenarioRunner

Clone the ScenarioRunner repository:
git clone https://github.com/carla-simulator/scenario_runner.git

### 4. Dependency libraries
pip install -r requirements.txt

### 5. Run the tool
streamlit run UI/mainUIlauncher.py
