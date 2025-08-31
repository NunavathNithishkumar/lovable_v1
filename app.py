import streamlit as st
import requests
import json
from datetime import datetime, timedelta
import google.generativeai as genai
import tempfile
import os
from typing import Dict, List
import pandas as pd
import re

# Configure page
st.set_page_config(
    page_title="AI Prompt Evolution Suite",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin: 1rem 0;
        padding: 0.5rem;
        background: linear-gradient(90deg, #ff7f0e20, transparent);
        border-left: 4px solid #ff7f0e;
        border-radius: 5px;
    }
    .phase-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2ca02c;
        margin: 1.5rem 0;
        padding: 1rem;
        background: linear-gradient(90deg, #2ca02c20, transparent);
        border-left: 6px solid #2ca02c;
        border-radius: 8px;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #f0fff0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2ca02c;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff8dc;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffa500;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #fff0f0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #d62728;
        margin: 1rem 0;
    }
    .workflow-step {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #dee2e6;
        margin: 1rem 0;
    }
    .completed-step {
        background-color: #d4edda;
        border-color: #2ca02c;
    }
    .current-step {
        background-color: #fff3cd;
        border-color: #ffa500;
    }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .refinement-section {
        background-color: #fff9c4;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #fbc02d;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'primary_prompt' not in st.session_state:
    st.session_state.primary_prompt = None
if 'master_prompt' not in st.session_state:
    st.session_state.master_prompt = None
if 'call_insights' not in st.session_state:
    st.session_state.call_insights = []
if 'transcriptions' not in st.session_state:
    st.session_state.transcriptions = []
if 'api_configured' not in st.session_state:
    st.session_state.api_configured = False
if 'current_phase' not in st.session_state:
    st.session_state.current_phase = 1
if 'agent_details' not in st.session_state:
    st.session_state.agent_details = {}
if 'refinement_history' not in st.session_state:
    st.session_state.refinement_history = []
if 'refinement_chat' not in st.session_state:
    st.session_state.refinement_chat = []

def configure_gemini(api_key):
    """Configure Gemini AI with the provided API key"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        st.session_state.api_configured = True
        return model
    except Exception as e:
        st.error(f"Failed to configure Gemini AI: {str(e)}")
        st.session_state.api_configured = False
        return None

def seconds_to_hms(seconds):
    """Convert seconds (float) to [HH:MM:SS] format."""
    td = str(timedelta(seconds=int(seconds)))
    if len(td.split(':')) == 2:
        td = "00:" + td
    return f"[{td}]"

def transcribe_audio(audio_file, deepgram_api_key: str, language: str = "hi") -> Dict:
    """Transcribe audio using Deepgram API with diarization."""
    try:
        response = requests.post(
            "https://api.deepgram.com/v1/listen",
            headers={
                "Authorization": f"Token {deepgram_api_key}",
                "Content-Type": "audio/mpeg"
            },
            params={
                "punctuate": "true",
                "diarize": "true",
                "language": language
            },
            data=audio_file
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Transcription failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error during transcription: {str(e)}")
        return None

def format_transcript(result: Dict) -> str:
    """Format transcript with speaker diarization and timestamps."""
    if not result or "results" not in result:
        return "No transcription available"
    
    words = result["results"]["channels"][0]["alternatives"][0]["words"]
    
    # Group words by speaker
    current_speaker = None
    current_start = None
    current_text = []
    formatted_transcript = []
    
    for word in words:
        speaker = f"Speaker {word['speaker']+1}"
        
        if speaker != current_speaker:
            # Add the last speaker block if exists
            if current_speaker is not None:
                formatted_transcript.append(
                    f"{seconds_to_hms(current_start)} {current_speaker}: \"{' '.join(current_text)}\""
                )
            
            # Start a new speaker block
            current_speaker = speaker
            current_start = word["start"]
            current_text = [word["word"]]
        else:
            current_text.append(word["word"])
    
    # Add the last block
    if current_text:
        formatted_transcript.append(
            f"{seconds_to_hms(current_start)} {current_speaker}: \"{' '.join(current_text)}\""
        )
    
    return "\n".join(formatted_transcript)

def generate_primary_prompt(script_content, template_content, agent_details, model):
    """Generate the primary prompt using script, template and agent details"""
    try:
        instructions = f"""
You are an expert AI prompt engineer specializing in creating CASE-SPECIFIC prompts for AI calling agents. 
Your task is to analyze a provided script and fill a universal template to create a CUSTOMIZED, ready-to-use PRIMARY prompt for that specific use case.

AGENT DETAILS TO INCORPORATE:
- Agent Name: {agent_details.get('name', 'Agent')}
- Company: {agent_details.get('company', 'Company')}
- Language: {agent_details.get('language', 'Hinglish')}
- Category: {agent_details.get('category', 'General')}

CRITICAL REQUIREMENTS:
1. The final output MUST be a CASE-SPECIFIC prompt, not a generic template
2. ALL placeholders like [abc], [XYZ], [Vehicle Model], etc. must be filled with actual values from the script
3. The final output MUST be in proper Markdown format with the following sections:
   - **Primary Objective**
   - **Objective** (numbered list)
   - **Strict Rules**
   - **User Details** (if applicable)
   - **AI Agent Identity**
   - **Name Usage Guideline**
   - **Call scheduling rules**
   - **Call Script** (main flow)
   - **Strict Interaction Rules (English)**
   - **Handling Short Responses & Maintaining Conversation Flow [Interruption Rules]**
   - **Standard Objection Handling**
   - **Fundamental Guidelines for Responses**
   - **Numeric & Language Best Practices**
   - **Guidelines for Conversation in {agent_details.get('language', 'Hinglish')}**
   - **Strict Guidelines**

4. Create a COMPLETE, WORKING prompt that can be used immediately for that specific calling scenario
5. Incorporate the agent name "{agent_details.get('name', 'Agent')}" throughout the prompt
6. Set the language appropriately for "{agent_details.get('language', 'Hinglish')}"
7. Customize for the "{agent_details.get('category', 'General')}" use case

ANALYSIS AND EXTRACTION PROCESS:
1. **Extract Key Information** from the script:
   - Company name, product/service details
   - Agent role, customer details, identifiers
   - Call purpose, current state, proposed state
   - Pricing, discounts, offers
   - Objections and responses
   - Call flow steps
   - Specific dialogue examples

2. **Fill ALL Template Variables** with extracted information:
   - Replace [abc] with actual agent name or role from script
   - Replace [XYZ] with customer name/type from script
   - Replace [Vehicle Model] with actual product identifier from script
   - Replace [Amount], [Discount Amount] with actual values from script
   - Fill in ALL bracketed placeholders with specific information

3. **Customize Content Sections**:
   - Update Knowledge Base with script-specific information
   - Modify objection handling with script's actual objections and responses
   - Adapt call flow to match script's specific process
   - Include script-specific dialogue examples

The output should be a COMPLETE, CASE-SPECIFIC PRIMARY prompt that an AI agent can use immediately.
"""
        
        full_prompt = f"""
{instructions}

SCRIPT TO ANALYZE:
{script_content}

UNIVERSAL TEMPLATE TO CUSTOMIZE:
{template_content}

TASK: Create a PRIMARY AI calling agent prompt by filling the template with script information and agent details. This will be improved later based on real call insights.

Generate a complete, ready-to-use, case-specific PRIMARY AI calling agent prompt in Markdown format.
"""
        
        with st.spinner("🤖 Generating PRIMARY prompt..."):
            response = model.generate_content(full_prompt)
            return response.text
    
    except Exception as e:
        st.error(f"Error generating primary prompt: {str(e)}")
        return None

def extract_call_insights(transcript: str, primary_prompt: str, model):
    """Extract actionable insights from call transcript to improve the prompt"""
    try:
        analysis_prompt = f"""
You are an expert call analysis consultant. Analyze this call transcript against the current AI agent prompt and extract SPECIFIC, ACTIONABLE insights that can be used to improve the prompt.

**CURRENT PRIMARY PROMPT:**
{primary_prompt}

**ACTUAL CALL TRANSCRIPT:**
{transcript}

Please provide insights in the following JSON-like format for easy integration:

## 🔍 CALL ANALYSIS INSIGHTS

### ✅ What Worked Well:
[List specific techniques, phrases, or approaches from the call that were effective]

### ❌ What Didn't Work:
[List specific issues, missed opportunities, or ineffective approaches]

### 🎯 Missing from Current Prompt:
[List specific elements present in the successful call but missing from current prompt]

### 🔄 Prompt Improvements Needed:
[Specific, actionable recommendations for updating the prompt]

### 💬 Effective Dialogue Examples:
[Extract specific dialogue examples that should be added to the prompt]

### 🚨 New Objections & Responses:
[List any new objections encountered and how they were handled]

### 📊 Performance Scores:
- Script Adherence: X/10
- Persuasiveness: X/10
- Professionalism: X/10
- Information Gathering: X/10

### 🎯 Key Learnings for AI Agent:
[Specific behavioral patterns, timing, or techniques that the AI should learn]

Focus on SPECIFIC, IMPLEMENTABLE insights that can directly improve the AI agent prompt.
"""
        
        with st.spinner("🔍 Extracting insights from call..."):
            response = model.generate_content(analysis_prompt)
            return response.text
    
    except Exception as e:
        st.error(f"Error extracting insights: {str(e)}")
        return None

def generate_master_prompt(primary_prompt: str, all_insights: List[str], agent_details, model):
    """Generate the final master prompt using primary prompt and all collected insights"""
    try:
        combined_insights = "\n\n---\n\n".join(all_insights)
        
        master_prompt_instructions = f"""
You are an expert AI prompt engineer. Your task is to create a MASTER AI calling agent prompt by improving the PRIMARY prompt using insights from multiple real call recordings.

AGENT DETAILS:
- Agent Name: {agent_details.get('name', 'Agent')}
- Company: {agent_details.get('company', 'Company')}
- Language: {agent_details.get('language', 'Hinglish')}
- Category: {agent_details.get('category', 'General')}

**PRIMARY PROMPT (BASELINE):**
{primary_prompt}

**INSIGHTS FROM REAL CALLS:**
{combined_insights}

**YOUR TASK:**
Create a MASTER prompt that:
1. **Keeps the core structure** of the primary prompt with all required sections
2. **Integrates successful techniques** identified from real calls
3. **Adds missing elements** that were found effective in actual calls
4. **Improves objection handling** based on real scenarios encountered
5. **Enhances dialogue examples** with proven effective phrases
6. **Optimizes call flow** based on what actually works
7. **Addresses weaknesses** identified in the analysis

**MASTER PROMPT REQUIREMENTS:**
- Must be MORE comprehensive than the primary prompt
- Must include REAL, tested dialogue examples
- Must have PROVEN objection handling techniques
- Must incorporate SUCCESSFUL persuasion methods
- Must be IMMEDIATELY usable for AI agents
- Must maintain proper Markdown formatting
- Must include ALL required sections in the exact structure:
  * **Primary Objective**
  * ### **Objective**
  * **Strict Rules**
  * **User Details** (if applicable)
  * **AI Agent Identity**
  * **Name Usage Guideline**
  * **Call scheduling rules**
  * **Call Script** (main flow)
  * ## Strict Interaction Rules (English)
  * **Handling Short Responses & Maintaining Conversation Flow [Interruption Rules]**
  * ### **Standard Objection Handling**
  * ### **Fundamental Guidelines for Responses**
  * ### **Numeric & Language Best Practices**
  * ### **Guidelines for Conversation in {agent_details.get('language', 'Hinglish')}**
  * ### **Strict Guidelines**

Generate the FINAL MASTER AI calling agent prompt that represents the evolution from theory (primary prompt) to practice (real call insights).
"""
        
        with st.spinner("🧠 Creating MASTER prompt from insights..."):
            response = model.generate_content(master_prompt_instructions)
            return response.text
    
    except Exception as e:
        st.error(f"Error generating master prompt: {str(e)}")
        return None

def refine_master_prompt(current_prompt: str, user_feedback: str, model):
    """Refine the master prompt based on user feedback"""
    try:
        refinement_prompt = f"""
You are an expert AI prompt engineer. A user has provided feedback about issues with their current master prompt. Your task is to update the prompt to address their concerns while maintaining the overall structure and quality.

**CURRENT MASTER PROMPT:**
{current_prompt}

**USER FEEDBACK/ISSUE:**
{user_feedback}

**YOUR TASK:**
1. Analyze the user's feedback to understand the specific issue
2. Identify which part of the prompt needs modification
3. Make targeted improvements to address the issue
4. Ensure the updated prompt maintains all required sections
5. Keep the overall structure and flow intact
6. Make minimal changes that specifically address the user's concern

**REQUIREMENTS:**
- Keep the same markdown structure with all sections
- Make surgical changes that directly address the feedback
- Maintain the prompt's effectiveness for other scenarios
- Ensure changes are practical and implementable
- Provide a complete, updated prompt

Generate the REFINED master prompt with the requested improvements.
"""
        
        with st.spinner("🔧 Refining master prompt..."):
            response = model.generate_content(refinement_prompt)
            return response.text
    
    except Exception as e:
        st.error(f"Error refining prompt: {str(e)}")
        return None

def display_workflow_progress():
    """Display the current workflow progress"""
    st.markdown("### 🔄 Workflow Progress")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.session_state.primary_prompt:
            st.markdown('<div class="workflow-step completed-step">', unsafe_allow_html=True)
            st.markdown("**✅ Phase 1: COMPLETED**")
            st.markdown("Primary Prompt Generated")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            step_class = "current-step" if st.session_state.current_phase == 1 else "workflow-step"
            st.markdown(f'<div class="workflow-step {step_class}">', unsafe_allow_html=True)
            st.markdown("**📝 Phase 1: Create Primary**")
            st.markdown("Agent Details + Script + Template")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if len(st.session_state.call_insights) > 0:
            st.markdown('<div class="workflow-step completed-step">', unsafe_allow_html=True)
            st.markdown(f"**✅ Phase 2: COMPLETED**")
            st.markdown(f"Analyzed {len(st.session_state.call_insights)} calls")
            st.markdown('</div>', unsafe_allow_html=True)
        elif st.session_state.primary_prompt:
            step_class = "current-step" if st.session_state.current_phase == 2 else "workflow-step"
            st.markdown(f'<div class="workflow-step {step_class}">', unsafe_allow_html=True)
            st.markdown("**🎵 Phase 2: Extract Insights**")
            st.markdown("Upload Call Recordings")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="workflow-step">', unsafe_allow_html=True)
            st.markdown("**⏳ Phase 2: Waiting**")
            st.markdown("Extract Insights")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        if st.session_state.master_prompt:
            st.markdown('<div class="workflow-step completed-step">', unsafe_allow_html=True)
            st.markdown("**✅ Phase 3: COMPLETED**")
            st.markdown("Master Prompt Ready!")
            st.markdown('</div>', unsafe_allow_html=True)
        elif len(st.session_state.call_insights) > 0:
            step_class = "current-step" if st.session_state.current_phase == 3 else "workflow-step"
            st.markdown(f'<div class="workflow-step {step_class}">', unsafe_allow_html=True)
            st.markdown("**🧠 Phase 3: Create Master**")
            st.markdown("Generate Final Prompt")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="workflow-step">', unsafe_allow_html=True)
            st.markdown("**⏳ Phase 3: Waiting**")
            st.markdown("Create Master Prompt")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        if st.session_state.master_prompt:
            step_class = "current-step" if st.session_state.current_phase == 4 else "workflow-step"
            st.markdown(f'<div class="workflow-step {step_class}">', unsafe_allow_html=True)
            st.markdown("**🔧 Phase 4: Refine**")
            st.markdown("Test & Improve")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="workflow-step">', unsafe_allow_html=True)
            st.markdown("**⏳ Phase 4: Waiting**")
            st.markdown("Test & Refine")
            st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Main header
    st.markdown('<h1 class="main-header">🧠 AI Prompt Evolution Suite</h1>', unsafe_allow_html=True)
    
    # Description
    st.markdown("""
    <div class="info-box">
        <strong>Evolve Your AI Agent Prompts Through Real-World Learning!</strong><br>
        This suite follows a 4-phase approach:
        <br><br>
        <strong>Phase 1:</strong> 📝 Create PRIMARY prompt from agent details + script + template<br>
        <strong>Phase 2:</strong> 🎵 Extract insights from real call recordings<br>
        <strong>Phase 3:</strong> 🧠 Generate MASTER prompt using real-world insights<br>
        <strong>Phase 4:</strong> 🔧 Test & refine prompt based on performance feedback
    </div>
    """, unsafe_allow_html=True)
    
    # Display workflow progress
    display_workflow_progress()
    
    # Sidebar for API configuration
    with st.sidebar:
        st.header("🔧 API Configuration")
        
        # Gemini API Key
        gemini_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            help="Required for AI analysis and prompt generation"
        )
        
        # Deepgram API Key
        deepgram_key = st.text_input(
            "Deepgram API Key", 
            type="password",
            help="Required for audio transcription"
        )
        
        # Language selection for transcription
        language = st.selectbox("Transcription Language", 
                               options=["hi", "en", "es", "fr", "de"],
                               index=0,
                               help="Select the primary language of the call")
        
        # Configure Gemini
        model = None
        if gemini_key:
            model = configure_gemini(gemini_key)
            if st.session_state.api_configured:
                st.success("✅ Gemini AI configured!")
            else:
                st.error("❌ Failed to configure Gemini AI")
        else:
            st.warning("⚠️ Enter Gemini API key")
        
        # Deepgram status
        if deepgram_key:
            st.success("✅ Deepgram API key provided")
        else:
            st.warning("⚠️ Deepgram key needed for Phase 2")
        
        st.markdown("---")
        st.markdown("### 📊 Progress Summary")
        st.write(f"**Agent Details:** {'✅' if st.session_state.agent_details else '❌'}")
        st.write(f"**Primary Prompt:** {'✅' if st.session_state.primary_prompt else '❌'}")
        st.write(f"**Call Insights:** {len(st.session_state.call_insights)} collected")
        st.write(f"**Master Prompt:** {'✅' if st.session_state.master_prompt else '❌'}")
        st.write(f"**Refinements:** {len(st.session_state.refinement_history)} made")
        
        if st.button("🔄 Reset All Progress", type="secondary"):
            for key in ['primary_prompt', 'master_prompt', 'call_insights', 'transcriptions', 'agent_details', 'refinement_history', 'refinement_chat']:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.current_phase = 1
            st.rerun()
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📝 Phase 1: Primary Prompt", 
        "🎵 Phase 2: Call Analysis",
        "📋 Call Insights",
        "🧠 Phase 3: Master Prompt", 
        "🔧 Phase 4: Refine & Test",
        "📊 Final Results"
    ])
    
    # PHASE 1: PRIMARY PROMPT CREATION
    with tab1:
        st.markdown('<div class="phase-header">📝 Phase 1: Create Primary Prompt</div>', unsafe_allow_html=True)
        
        if st.session_state.primary_prompt:
            st.markdown("""
            <div class="success-box">
                <strong>✅ Phase 1 Completed!</strong><br>
                Your primary prompt has been generated with agent details. You can now proceed to Phase 2 to upload call recordings and extract insights.
            </div>
            """, unsafe_allow_html=True)
            
            # Show agent details
            if st.session_state.agent_details:
                with st.expander("👤 Agent Details", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Name:** {st.session_state.agent_details.get('name', 'N/A')}")
                        st.write(f"**Company:** {st.session_state.agent_details.get('company', 'N/A')}")
                    with col2:
                        st.write(f"**Language:** {st.session_state.agent_details.get('language', 'N/A')}")
                        st.write(f"**Category:** {st.session_state.agent_details.get('category', 'N/A')}")
            
            # Display primary prompt
            with st.expander("📖 View Primary Prompt", expanded=False):
                st.markdown(st.session_state.primary_prompt)
            
            # Download option
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                "📥 Download Primary Prompt",
                st.session_state.primary_prompt,
                file_name=f"primary_prompt_{timestamp}.md",
                mime="text/markdown"
            )
            
        else:
            # Agent Details Section
            st.markdown('<div class="step-header">👤 Agent Details</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                agent_name = st.text_input(
                    "Agent Name *",
                    value=st.session_state.agent_details.get('name', ''),
                    placeholder="e.g., Riya, Rahul, Sarah"
                )
                
                company_name = st.text_input(
                    "Company Name *",
                    value=st.session_state.agent_details.get('company', ''),
                    placeholder="e.g., Puravankara, HDFC Bank, Airtel"
                )
            
            with col2:
                agent_language = st.selectbox(
                    "Agent Language *",
                    options=["Hinglish", "English", "Hindi", "Tamil", "Telugu", "Gujarati", "Marathi"],
                    index=0 if not st.session_state.agent_details.get('language') else 
                          ["Hinglish", "English", "Hindi", "Tamil", "Telugu", "Gujarati", "Marathi"].index(st.session_state.agent_details.get('language', 'Hinglish'))
                )
                
                prompt_category = st.selectbox(
                    "Prompt Category *",
                    options=["Lead Qualification", "EMI Reminder", "Property Sales", "Loan Collection", "Insurance Sales", "Customer Support", "Appointment Booking", "Survey & Feedback"],
                    index=0 if not st.session_state.agent_details.get('category') else
                          ["Lead Qualification", "EMI Reminder", "Property Sales", "Loan Collection", "Insurance Sales", "Customer Support", "Appointment Booking", "Survey & Feedback"].index(st.session_state.agent_details.get('category', 'Lead Qualification'))
                )
            
            # Save agent details
            if st.button("💾 Save Agent Details", type="secondary"):
                if agent_name and company_name and agent_language and prompt_category:
                    st.session_state.agent_details = {
                        'name': agent_name,
                        'company': company_name,
                        'language': agent_language,
                        'category': prompt_category
                    }
                    st.success("✅ Agent details saved!")
                else:
                    st.error("❌ Please fill all required fields")
            
            st.markdown("---")
            
            # Script and Template Section
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown('<div class="step-header">📝 Upload Script</div>', unsafe_allow_html=True)
                
                script_input_method = st.radio(
                    "Choose script input method:",
                    ["Upload File", "Paste Text"],
                    key="script_method"
                )
                
                script_content = ""
                
                if script_input_method == "Upload File":
                    uploaded_script = st.file_uploader(
                        "Upload your script file",
                        type=['txt', 'md', 'doc', 'docx'],
                        help="Upload your call script",
                        key="script_upload"
                    )
                    
                    if uploaded_script:
                        try:
                            script_content = uploaded_script.read().decode('utf-8')
                            st.success(f"✅ Script uploaded: {uploaded_script.name}")
                            with st.expander("Preview Script"):
                                st.text_area("", script_content, height=200, disabled=True)
                        except Exception as e:
                            st.error(f"Error reading script: {str(e)}")
                else:
                    script_content = st.text_area(
                        "Paste your script here:",
                        height=300,
                        placeholder="Enter your complete call script with flow, objections, etc."
                    )
            
            with col2:
                st.markdown('<div class="step-header">📋 Upload Template</div>', unsafe_allow_html=True)
                
                template_input_method = st.radio(
                    "Choose template input method:",
                    ["Upload File", "Paste Text"],
                    key="template_method"
                )
                
                template_content = ""
                
                if template_input_method == "Upload File":
                    uploaded_template = st.file_uploader(
                        "Upload your template file",
                        type=['txt', 'md', 'doc', 'docx'],
                        help="Upload your prompt template",
                        key="template_upload"
                    )
                    
                    if uploaded_template:
                        try:
                            template_content = uploaded_template.read().decode('utf-8')
                            st.success(f"✅ Template uploaded: {uploaded_template.name}")
                            with st.expander("Preview Template"):
                                st.text_area("", template_content, height=200, disabled=True)
                        except Exception as e:
                            st.error(f"Error reading template: {str(e)}")
                else:
                    template_content = st.text_area(
                        "Paste your template here:",
                        height=300,
                        placeholder="Enter your prompt template with placeholders [abc], [XYZ], etc."
                    )
            
            # Generate primary prompt
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Generate Primary Prompt", type="primary", use_container_width=True):
                    if not st.session_state.agent_details:
                        st.error("❌ Please save agent details first")
                    elif not script_content.strip():
                        st.error("❌ Please provide a script")
                    elif not template_content.strip():
                        st.error("❌ Please provide a template")
                    elif not st.session_state.api_configured:
                        st.error("❌ Please configure Gemini API key")
                    else:
                        primary_prompt = generate_primary_prompt(script_content, template_content, st.session_state.agent_details, model)
                        if primary_prompt:
                            st.session_state.primary_prompt = primary_prompt
                            st.session_state.current_phase = 2
                            st.success("✅ Primary prompt generated successfully!")
                            st.balloons()
                            st.rerun()
    
    # PHASE 2: CALL ANALYSIS
    with tab2:
        st.markdown('<div class="phase-header">🎵 Phase 2: Extract Insights from Call Recordings</div>', unsafe_allow_html=True)
        
        if not st.session_state.primary_prompt:
            st.markdown("""
            <div class="warning-box">
                <strong>⚠️ Phase 1 Required</strong><br>
                Please complete Phase 1 (Create Primary Prompt) before proceeding to Phase 2.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
                <strong>Upload call recordings to extract insights that will improve your primary prompt.</strong><br>
                Each call will be transcribed and analyzed to identify what works and what can be improved.
            </div>
            """, unsafe_allow_html=True)
            
            # Show current insights count
            if len(st.session_state.call_insights) > 0:
                st.markdown(f"""
                <div class="success-box">
                    <strong>📊 Progress Update</strong><br>
                    You have analyzed <strong>{len(st.session_state.call_insights)} call(s)</strong> so far. 
                    Continue adding more calls for better insights, or proceed to Phase 3 to generate your master prompt.
                </div>
                """, unsafe_allow_html=True)
            
            # Audio upload section
            st.markdown('<div class="step-header">🎵 Upload Call Recordings</div>', unsafe_allow_html=True)
            
            # Multiple file upload
            uploaded_audios = st.file_uploader(
                "Choose audio files",
                type=['mp3', 'wav', 'm4a', 'flac'],
                help="Upload one or multiple call recordings to extract insights",
                accept_multiple_files=True
            )
            
            if uploaded_audios:
                st.markdown(f"**{len(uploaded_audios)} file(s) selected:**")
                
                # Show uploaded files
                cols = st.columns(min(3, len(uploaded_audios)))
                for i, audio_file in enumerate(uploaded_audios):
                    with cols[i % 3]:
                        st.write(f"📁 {audio_file.name}")
                        st.audio(audio_file)
                
                # Batch processing options
                col1, col2 = st.columns(2)
                with col1:
                    process_mode = st.radio(
                        "Processing Mode:",
                        ["Process All at Once", "Process One by One"],
                        help="Choose how to process multiple files"
                    )
                
                with col2:
                    if process_mode == "Process One by One":
                        selected_file_index = st.selectbox(
                            "Select file to process:",
                            range(len(uploaded_audios)),
                            format_func=lambda x: uploaded_audios[x].name
                        )
                
                # Process button
                if process_mode == "Process All at Once":
                    button_text = f"🚀 Analyze All {len(uploaded_audios)} Calls"
                    button_key = "process_all"
                else:
                    button_text = f"🔍 Analyze: {uploaded_audios[selected_file_index].name}"
                    button_key = "process_single"
                
                if st.button(button_text, type="primary", use_container_width=True, key=button_key):
                    if not deepgram_key:
                        st.error("❌ Please provide Deepgram API key in sidebar")
                    elif not st.session_state.api_configured:
                        st.error("❌ Please configure Gemini API key")
                    else:
                        # Determine which files to process
                        if process_mode == "Process All at Once":
                            files_to_process = uploaded_audios
                        else:
                            files_to_process = [uploaded_audios[selected_file_index]]
                        
                        # Process files
                        successful_analyses = 0
                        failed_analyses = 0
                        
                        # Create progress bar for multiple files
                        if len(files_to_process) > 1:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                        
                        for i, audio_file in enumerate(files_to_process):
                            if len(files_to_process) > 1:
                                progress = (i) / len(files_to_process)
                                progress_bar.progress(progress)
                                status_text.text(f"Processing {audio_file.name} ({i+1}/{len(files_to_process)})")
                            
                            with st.spinner(f"Processing {audio_file.name}..."):
                                # Step 1: Transcribe
                                if len(files_to_process) == 1:
                                    st.info(f"Step 1: Transcribing {audio_file.name}...")
                                
                                transcription_result = transcribe_audio(audio_file.getvalue(), deepgram_key, language)
                                
                                if transcription_result:
                                    formatted_transcript = format_transcript(transcription_result)
                                    st.session_state.transcriptions.append({
                                        'filename': audio_file.name,
                                        'transcript': formatted_transcript,
                                        'timestamp': datetime.now()
                                    })
                                    
                                    if len(files_to_process) == 1:
                                        st.success(f"✅ Transcription completed for {audio_file.name}")
                                        st.info(f"Step 2: Extracting insights from {audio_file.name}...")
                                    
                                    # Step 2: Extract insights
                                    insights = extract_call_insights(formatted_transcript, st.session_state.primary_prompt, model)
                                    
                                    if insights:
                                        st.session_state.call_insights.append({
                                            'filename': audio_file.name,
                                            'insights': insights,
                                            'transcript': formatted_transcript,
                                            'timestamp': datetime.now()
                                        })
                                        
                                        successful_analyses += 1
                                        
                                        if len(files_to_process) == 1:
                                            st.success(f"✅ Insights extracted from {audio_file.name}")
                                            # Show quick preview for single file
                                            with st.expander(f"📋 Preview Insights - {audio_file.name}"):
                                                st.markdown(insights)
                                    else:
                                        failed_analyses += 1
                                        if len(files_to_process) == 1:
                                            st.error(f"❌ Failed to extract insights from {audio_file.name}")
                                else:
                                    failed_analyses += 1
                                    if len(files_to_process) == 1:
                                        st.error(f"❌ Transcription failed for {audio_file.name}")
                        
                        # Final progress update for multiple files
                        if len(files_to_process) > 1:
                            progress_bar.progress(1.0)
                            status_text.text("Processing complete!")
                        
                        # Show final results
                        if successful_analyses > 0:
                            st.success(f"✅ Successfully analyzed {successful_analyses} call(s)! Total calls analyzed: {len(st.session_state.call_insights)}")
                            st.balloons()
                            
                            # Show summary for multiple files
                            if len(files_to_process) > 1:
                                with st.expander(f"📊 Batch Processing Summary"):
                                    st.write(f"**Total Files Processed:** {len(files_to_process)}")
                                    st.write(f"**Successful Analyses:** {successful_analyses}")
                                    st.write(f"**Failed Analyses:** {failed_analyses}")
                                    st.write(f"**Success Rate:** {(successful_analyses/len(files_to_process)*100):.1f}%")
                                    
                                    # List processed files
                                    st.markdown("**Successfully Processed Files:**")
                                    recent_insights = st.session_state.call_insights[-successful_analyses:]
                                    for insight in recent_insights:
                                        st.write(f"• {insight['filename']}")
                        
                        if failed_analyses > 0:
                            st.error(f"❌ Failed to analyze {failed_analyses} call(s)")
                
                # Quick stats
                if len(st.session_state.call_insights) > 0:
                    st.markdown("---")
                    st.markdown("### 📊 Current Progress")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Files Selected", len(uploaded_audios))
                    with col2:
                        st.metric("Calls Analyzed", len(st.session_state.call_insights))
                    with col3:
                        st.metric("Ready for Phase 3", "✅" if len(st.session_state.call_insights) > 0 else "❌")
            
            # Quick actions
            if len(st.session_state.call_insights) > 0:
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**{len(st.session_state.call_insights)} calls analyzed**")
                    st.write("Ready to generate master prompt!")
                with col2:
                    if st.button("➡️ Proceed to Phase 3", type="secondary", use_container_width=True):
                        st.session_state.current_phase = 3
                        st.rerun()

    # CALL INSIGHTS VIEWER
    with tab3:
        st.header("📋 Extracted Call Insights")
        
        if len(st.session_state.call_insights) == 0:
            st.info("No call insights available yet. Upload and analyze call recordings in Phase 2.")
        else:
            st.markdown(f"**Total Calls Analyzed:** {len(st.session_state.call_insights)}")
            
            # Display each insight
            for i, call_data in enumerate(st.session_state.call_insights, 1):
                with st.expander(f"📞 Call {i}: {call_data['filename']}", expanded=False):
                    
                    # Tabs for transcript and insights
                    sub_tab1, sub_tab2 = st.tabs(["🔍 Insights", "📝 Transcript"])
                    
                    with sub_tab1:
                        st.markdown(call_data['insights'])
                    
                    with sub_tab2:
                        st.code(call_data['transcript'], language=None)
                    
                    # Download options
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            f"📥 Download Insights {i}",
                            call_data['insights'],
                            file_name=f"call_{i}_insights.md",
                            mime="text/markdown",
                            key=f"insights_download_{i}"
                        )
                    with col2:
                        st.download_button(
                            f"📥 Download Transcript {i}",
                            call_data['transcript'],
                            file_name=f"call_{i}_transcript.txt",
                            mime="text/plain",
                            key=f"transcript_download_{i}"
                        )
            
            # Consolidated insights download
            if len(st.session_state.call_insights) > 1:
                st.markdown("---")
                all_insights = "\n\n" + "="*50 + "\n\n".join([
                    f"# CALL {i+1}: {call['filename']}\n\n{call['insights']}" 
                    for i, call in enumerate(st.session_state.call_insights)
                ])
                
                st.download_button(
                    "📦 Download All Insights Combined",
                    all_insights,
                    file_name=f"all_call_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )

    # PHASE 3: MASTER PROMPT CREATION
    with tab4:
        st.markdown('<div class="phase-header">🧠 Phase 3: Generate Master Prompt</div>', unsafe_allow_html=True)
        
        if not st.session_state.primary_prompt:
            st.markdown("""
            <div class="warning-box">
                <strong>⚠️ Phase 1 Required</strong><br>
                Please complete Phase 1 (Create Primary Prompt) first.
            </div>
            """, unsafe_allow_html=True)
        elif len(st.session_state.call_insights) == 0:
            st.markdown("""
            <div class="warning-box">
                <strong>⚠️ Phase 2 Required</strong><br>
                Please complete Phase 2 (Extract Call Insights) by uploading and analyzing at least one call recording.
            </div>
            """, unsafe_allow_html=True)
        else:
            if st.session_state.master_prompt:
                st.markdown("""
                <div class="success-box">
                    <strong>🎉 Phase 3 Completed!</strong><br>
                    Your MASTER prompt has been generated using insights from real call recordings. 
                    This is your optimized AI agent prompt ready for testing and refinement in Phase 4.
                </div>
                """, unsafe_allow_html=True)
                
                # Display master prompt
                with st.expander("🧠 View Master Prompt", expanded=True):
                    st.markdown(st.session_state.master_prompt)
                
                # Download and actions
                col1, col2 = st.columns(2)
                with col1:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.download_button(
                        "📥 Download Master Prompt",
                        st.session_state.master_prompt,
                        file_name=f"master_prompt_{timestamp}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                with col2:
                    if st.button("🔄 Regenerate Master Prompt", use_container_width=True):
                        st.session_state.master_prompt = None
                        st.rerun()
                
            else:
                # Show summary before generation
                st.markdown("""
                <div class="info-box">
                    <strong>Ready to Generate Master Prompt!</strong><br>
                    This will combine your primary prompt with insights from all analyzed call recordings 
                    to create the final, optimized AI agent prompt with all required sections.
                </div>
                """, unsafe_allow_html=True)
                
                # Summary stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Primary Prompt", "✅ Ready")
                with col2:
                    st.metric("Calls Analyzed", len(st.session_state.call_insights))
                with col3:
                    st.metric("Agent Details", "✅ Configured")
                
                # Preview what will be included
                with st.expander("📋 Preview: What Will Be Included", expanded=False):
                    st.markdown("**👤 Agent Details:**")
                    for key, value in st.session_state.agent_details.items():
                        st.write(f"• **{key.title()}:** {value}")
                    
                    st.markdown("**📝 Primary Prompt:**")
                    st.text_area("", st.session_state.primary_prompt[:500] + "...", height=100, disabled=True)
                    
                    st.markdown(f"**🔍 Insights from {len(st.session_state.call_insights)} Call(s):**")
                    for i, call in enumerate(st.session_state.call_insights, 1):
                        st.write(f"• Call {i}: {call['filename']}")
                
                # Generate master prompt button
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🧠 Generate Master Prompt", type="primary", use_container_width=True):
                        if not st.session_state.api_configured:
                            st.error("❌ Please configure Gemini API key")
                        else:
                            # Extract all insights
                            all_insights = [call['insights'] for call in st.session_state.call_insights]
                            
                            # Generate master prompt
                            master_prompt = generate_master_prompt(
                                st.session_state.primary_prompt,
                                all_insights,
                                st.session_state.agent_details,
                                model
                            )
                            
                            if master_prompt:
                                st.session_state.master_prompt = master_prompt
                                st.session_state.current_phase = 4
                                st.success("🎉 Master prompt generated successfully!")
                                st.balloons()
                                st.rerun()
                            else:
                                st.error("❌ Failed to generate master prompt")

    # PHASE 4: REFINE & TEST
    with tab5:
        st.markdown('<div class="phase-header">🔧 Phase 4: Test & Refine Master Prompt</div>', unsafe_allow_html=True)
        
        if not st.session_state.master_prompt:
            st.markdown("""
            <div class="warning-box">
                <strong>⚠️ Phase 3 Required</strong><br>
                Please complete Phase 3 (Generate Master Prompt) before proceeding to Phase 4.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
                <strong>Test your master prompt and refine it based on real performance!</strong><br>
                Found issues? Describe what's not working and get an updated prompt instantly.
                This works like Lovable - just tell us what needs to be fixed and we'll update the prompt.
            </div>
            """, unsafe_allow_html=True)
            
            # Current Master Prompt Section
            with st.expander("🧠 Current Master Prompt", expanded=False):
                st.markdown(st.session_state.master_prompt)
            
            # Refinement History
            if st.session_state.refinement_history:
                st.markdown(f"### 📈 Refinement History ({len(st.session_state.refinement_history)} changes made)")
                
                for i, refinement in enumerate(reversed(st.session_state.refinement_history), 1):
                    with st.expander(f"🔄 Refinement {len(st.session_state.refinement_history) - i + 1}: {refinement['timestamp'].strftime('%Y-%m-%d %H:%M')}", expanded=False):
                        st.markdown("**Issue Reported:**")
                        st.write(refinement['feedback'])
                        st.markdown("**Changes Made:**")
                        st.code(refinement['summary'], language=None)
            
            # Refinement Interface
            st.markdown('<div class="refinement-section">', unsafe_allow_html=True)
            st.markdown("### 🛠️ Refine Your Prompt")
            st.markdown("Describe any issues you're experiencing with the current prompt:")
            
            # Chat-like interface for refinements
            user_feedback = st.text_area(
                "What's not working? What needs to be improved?",
                placeholder="""Examples:
• "The agent is being too pushy in the opening"
• "Missing objection handling for price concerns"
• "Agent is not following the script flow properly"
• "Need to add more empathy in rejection handling"
• "The closing is too abrupt"
• "Agent is hallucinating information not in the prompt"
""",
                height=150,
                key="refinement_feedback"
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("🔧 Refine Prompt", type="primary", disabled=not user_feedback.strip()):
                    if not st.session_state.api_configured:
                        st.error("❌ Please configure Gemini API key")
                    elif not user_feedback.strip():
                        st.error("❌ Please describe the issue")
                    else:
                        # Refine the prompt
                        refined_prompt = refine_master_prompt(
                            st.session_state.master_prompt,
                            user_feedback,
                            model
                        )
                        
                        if refined_prompt:
                            # Save to history
                            st.session_state.refinement_history.append({
                                'feedback': user_feedback,
                                'old_prompt': st.session_state.master_prompt,
                                'new_prompt': refined_prompt,
                                'summary': f"Updated prompt based on: {user_feedback[:100]}{'...' if len(user_feedback) > 100 else ''}",
                                'timestamp': datetime.now()
                            })
                            
                            # Update current prompt
                            st.session_state.master_prompt = refined_prompt
                            
                            st.success("✅ Prompt refined successfully!")
                            st.balloons()
                            st.rerun()
                        else:
                            st.error("❌ Failed to refine prompt")
            
            with col2:
                if st.button("📥 Download Current Version", use_container_width=True):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    version = len(st.session_state.refinement_history) + 1
                    st.download_button(
                        f"📥 Download Master Prompt v{version}",
                        st.session_state.master_prompt,
                        file_name=f"master_prompt_v{version}_{timestamp}.md",
                        mime="text/markdown",
                        key="download_current_version"
                    )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Quick refinement suggestions
            st.markdown("### 💡 Common Refinement Areas")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🎭 Improve Tone & Personality", use_container_width=True):
                    st.session_state.refinement_feedback = "The agent's tone needs to be more friendly and empathetic. Make the personality more warm and conversational while maintaining professionalism."
                    st.rerun()
            
            with col2:
                if st.button("🛡️ Reduce Hallucinations", use_container_width=True):
                    st.session_state.refinement_feedback = "The agent is providing information that's not in the prompt. Add stronger guidelines to prevent hallucinations and stick only to provided information."
                    st.rerun()
            
            with col3:
                if st.button("🔄 Fix Call Flow Issues", use_container_width=True):
                    st.session_state.refinement_feedback = "The agent is not following the call flow properly. Strengthen the script adherence and step-by-step progression rules."
                    st.rerun()

    # FINAL RESULTS
    with tab6:
        st.header("📊 Final Results & Evolution Tracking")
        
        if not st.session_state.master_prompt:
            st.info("Complete all phases to see the final results and evolution tracking.")
        else:
            st.markdown("""
            <div class="success-box">
                <strong>🎉 Congratulations!</strong><br>
                You have successfully evolved your AI agent prompt from theory to practice using real call insights and iterative refinements.
            </div>
            """, unsafe_allow_html=True)
            
            # Evolution summary
            st.markdown("### 📈 Evolution Summary")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Phases Completed", "4/4")
            with col2:
                st.metric("Calls Analyzed", len(st.session_state.call_insights))
            with col3:
                st.metric("Insights Extracted", len(st.session_state.call_insights))
            with col4:
                st.metric("Refinements Made", len(st.session_state.refinement_history))
            with col5:
                st.metric("Final Version", f"v{len(st.session_state.refinement_history) + 1}")
            
            # Evolution Timeline
            st.markdown("### 🕒 Evolution Timeline")
            
            timeline_data = []
            timeline_data.append({"Phase": "Phase 1", "Action": "Primary Prompt Created", "Details": f"Agent: {st.session_state.agent_details.get('name', 'N/A')}, Category: {st.session_state.agent_details.get('category', 'N/A')}"})
            timeline_data.append({"Phase": "Phase 2", "Action": f"{len(st.session_state.call_insights)} Calls Analyzed", "Details": "Real call insights extracted"})
            timeline_data.append({"Phase": "Phase 3", "Action": "Master Prompt Generated", "Details": "Combined insights with primary prompt"})
            
            for i, refinement in enumerate(st.session_state.refinement_history, 1):
                timeline_data.append({"Phase": f"Phase 4.{i}", "Action": f"Refinement {i}", "Details": refinement['feedback'][:50] + "..."})
            
            df = pd.DataFrame(timeline_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Comparison tabs
            st.markdown("### 🔍 Evolution Comparison")
            
            compare_tab1, compare_tab2, compare_tab3, compare_tab4 = st.tabs([
                "📝 Primary Prompt (v1)", 
                "🧠 Master Prompt (Final)", 
                "📊 Evolution Analysis",
                "🔄 Refinement History"
            ])
            
            with compare_tab1:
                st.markdown("#### Original Primary Prompt (Version 1)")
                st.markdown(st.session_state.primary_prompt)
                
                st.download_button(
                    "📥 Download Primary Prompt",
                    st.session_state.primary_prompt,
                    file_name=f"primary_prompt_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
            
            with compare_tab2:
                st.markdown(f"#### Final Master Prompt (Version {len(st.session_state.refinement_history) + 1})")
                st.markdown(st.session_state.master_prompt)
                
                st.download_button(
                    "📥 Download Final Master Prompt",
                    st.session_state.master_prompt,
                    file_name=f"master_prompt_final_v{len(st.session_state.refinement_history) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
            
            with compare_tab3:
                st.markdown("#### Evolution Analysis")
                
                # Generate evolution analysis
                if st.button("📊 Generate Evolution Analysis", type="secondary"):
                    if st.session_state.api_configured:
                        evolution_prompt = f"""
Compare these AI agent prompts and provide a detailed analysis of how the prompt evolved from theory to practice:

**AGENT DETAILS:**
- Name: {st.session_state.agent_details.get('name', 'N/A')}
- Company: {st.session_state.agent_details.get('company', 'N/A')}
- Language: {st.session_state.agent_details.get('language', 'N/A')}
- Category: {st.session_state.agent_details.get('category', 'N/A')}

**ORIGINAL PRIMARY PROMPT:**
{st.session_state.primary_prompt}

**EVOLVED MASTER PROMPT:**
{st.session_state.master_prompt}

**EVOLUTION PROCESS:**
- Insights used: {len(st.session_state.call_insights)} real call recordings analyzed
- Refinements made: {len(st.session_state.refinement_history)} iterative improvements

Please provide analysis in this format:

## 📊 PROMPT EVOLUTION ANALYSIS

### ✅ Key Improvements Made:
[List specific improvements from primary to master]

### 🎯 New Elements Added:
[List new elements that weren't in the primary prompt]

### 💬 Enhanced Dialogue Examples:
[Compare dialogue examples between versions]

### 🔄 Improved Objection Handling:
[How objection handling was enhanced]

### 📈 Practical Enhancements:
[Real-world improvements based on call insights]

### 🎭 Behavioral Improvements:
[How the AI agent behavior was refined]

### 🔧 Refinement Impact:
[How iterative refinements improved the prompt]

### 📊 Evolution Metrics:
- Comprehensiveness: Estimate % increase
- Practical Examples: Count new examples added
- Objection Coverage: Count new objections covered
- Dialogue Quality: Improvements made
- Structure Completeness: All required sections present

### 💡 Key Learnings:
[What this evolution teaches us about AI prompt development]
"""
                        
                        with st.spinner("Analyzing prompt evolution..."):
                            response = model.generate_content(evolution_prompt)
                            if response:
                                st.markdown(response.text)
                                
                                # Download evolution analysis
                                st.download_button(
                                    "📥 Download Evolution Analysis",
                                    response.text,
                                    file_name=f"evolution_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                                    mime="text/markdown"
                                )
                    else:
                        st.error("❌ Please configure Gemini API key")
            
            with compare_tab4:
                st.markdown("#### Refinement History")
                
                if not st.session_state.refinement_history:
                    st.info("No refinements made yet. All improvements came from call insights analysis.")
                else:
                    for i, refinement in enumerate(st.session_state.refinement_history, 1):
                        with st.expander(f"🔧 Refinement {i}: {refinement['timestamp'].strftime('%Y-%m-%d %H:%M')}", expanded=False):
                            
                            col1, col2 = st.columns([1, 1])
                            with col1:
                                st.markdown("**Issue Reported:**")
                                st.write(refinement['feedback'])
                            
                            with col2:
                                st.markdown("**Changes Summary:**")
                                st.code(refinement['summary'], language=None)
                            
                            # Show before/after comparison for this refinement
                            ref_tab1, ref_tab2 = st.tabs([f"Before v{i}", f"After v{i+1}"])
                            
                            with ref_tab1:
                                st.text_area("", refinement['old_prompt'], height=200, disabled=True, key=f"before_{i}")
                            
                            with ref_tab2:
                                st.text_area("", refinement['new_prompt'], height=200, disabled=True, key=f"after_{i}")
            
            # Final download package
            st.markdown("---")
            st.markdown("### 📦 Complete Package Download")
            
            # Create complete package
            refinement_section = ""
            if st.session_state.refinement_history:
                refinement_section = f"""

## 🔧 Refinement History

{chr(10).join([f"### Refinement {i}: {refinement['timestamp'].strftime('%Y-%m-%d %H:%M')}{chr(10)}**Issue:** {refinement['feedback']}{chr(10)}**Summary:** {refinement['summary']}{chr(10)}" for i, refinement in enumerate(st.session_state.refinement_history, 1)])}
"""
            
            complete_package = f"""# AI Prompt Evolution Complete Package

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Evolution Summary

- **Phases Completed:** 4/4
- **Agent Name:** {st.session_state.agent_details.get('name', 'N/A')}
- **Company:** {st.session_state.agent_details.get('company', 'N/A')}
- **Language:** {st.session_state.agent_details.get('language', 'N/A')}
- **Category:** {st.session_state.agent_details.get('category', 'N/A')}
- **Calls Analyzed:** {len(st.session_state.call_insights)}
- **Refinements Made:** {len(st.session_state.refinement_history)}
- **Final Version:** v{len(st.session_state.refinement_history) + 1}

## 📝 Original Primary Prompt (v1)

{st.session_state.primary_prompt}

---

## 🔍 Call Insights Used

{chr(10).join([f"### Call {i+1}: {call['filename']}{chr(10)}{call['insights']}{chr(10)}" for i, call in enumerate(st.session_state.call_insights)])}

---
{refinement_section}
---

## 🧠 Final Master Prompt (v{len(st.session_state.refinement_history) + 1})

{st.session_state.master_prompt}

---

*Generated by AI Prompt Evolution Suite - Complete 4-Phase Evolution Process*
"""
            
            st.download_button(
                "📦 Download Complete Evolution Package",
                complete_package,
                file_name=f"ai_prompt_evolution_package_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True,
                type="primary",
                key="download_complete_package"
            )
    
    # Help section
    with st.expander("ℹ️ How to Use This Enhanced Suite"):
        st.markdown("""
        ### 🔄 Complete 4-Phase Workflow Guide:

        #### 📝 Phase 1: Create Primary Prompt
        1. **Configure API**: Enter your Gemini API key in the sidebar
        2. **Agent Details**: Fill in agent name, company, language, and category
        3. **Upload Script**: Provide your call script with detailed flow and information
        4. **Upload Template**: Provide a prompt template with placeholders [abc], [XYZ], etc.
        5. **Generate**: Click "Generate Primary Prompt" to create your baseline prompt

        #### 🎵 Phase 2: Extract Insights from Real Calls
        1. **Configure Deepgram**: Enter your Deepgram API key for transcription
        2. **Upload Recordings**: Upload actual call recordings (MP3, WAV, M4A, FLAC)
        3. **Analyze**: Each call will be transcribed and analyzed for insights
        4. **Collect**: Analyze multiple calls to gather comprehensive insights

        #### 🧠 Phase 3: Generate Master Prompt
        1. **Review**: Check your primary prompt and collected insights
        2. **Generate**: Create the final master prompt that combines theory + practice
        3. **Structure**: Ensures all required sections are included in proper markdown format
        4. **Optimize**: Incorporates real-world learnings into structured prompt

        #### 🔧 Phase 4: Test & Refine (NEW!)
        1. **Test**: Use your master prompt in real scenarios
        2. **Report Issues**: Describe any problems or areas for improvement
        3. **Refine**: Get instant updates to fix specific issues
        4. **Iterate**: Continue refining based on performance feedback
        5. **Track**: All changes are versioned and tracked

        ### 💡 Best Practices:

        #### Phase 1: 
        - Choose accurate agent details (name, language, category)
        - Include comprehensive scripts with objection handling
        - Use templates with clear placeholders

        #### Phase 2: 
        - Analyze both successful and challenging calls
        - Use high-quality audio files for better transcription
        - Include diverse call scenarios

        #### Phase 3: 
        - Review the generated structure carefully
        - Ensure all required sections are present
        - Verify agent details are properly incorporated

        #### Phase 4: 
        - Test thoroughly before reporting issues
        - Be specific about what's not working
        - Use iterative refinement for best results

        ### 🎯 Expected Outcomes:

        - **Primary Prompt**: Theoretical baseline with agent details and structure
        - **Call Insights**: Real-world patterns and improvements needed  
        - **Master Prompt**: Production-ready prompt with all required sections
        - **Refined Prompt**: Continuously improved based on real performance

        ### 📋 Required Master Prompt Sections:
        
        The final master prompt includes all these sections:
        - **Primary Objective**
        - **Objective** (numbered list)
        - **Strict Rules**
        - **User Details**
        - **AI Agent Identity**
        - **Name Usage Guideline**
        - **Call scheduling rules**
        - **Call Script** (main flow)
        - **Strict Interaction Rules (English)**
        - **Handling Short Responses & Maintaining Conversation Flow**
        - **Standard Objection Handling**
        - **Fundamental Guidelines for Responses**
        - **Numeric & Language Best Practices**
        - **Guidelines for Conversation** (in selected language)
        - **Strict Guidelines**
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("🧠 **AI Prompt Evolution Suite v2.0**: Complete 4-phase evolution process from theoretical prompts to production-ready, continuously refined AI agent instructions through systematic analysis and iterative improvement.")

if __name__ == "__main__":
    main()