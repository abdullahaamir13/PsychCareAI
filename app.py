import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Add project directories to path
if 'components' not in sys.path:
    sys.path.append('components')
if 'models' not in sys.path:
    sys.path.append('models')
if 'utils' not in sys.path:
    sys.path.append('utils')
if 'database' not in sys.path:
    sys.path.append('database')

# Import custom components
from components.dashboard import show_dashboard
from components.patient_input import show_patient_input
from components.chatbot import show_chatbot
from components.analysis_results import show_analysis_results
from models.ml_models import PsychiatricMLModels
from models.nlp_processor import NLPProcessor
from utils.risk_assessment import RiskAssessment
from database.schema import DatabaseManager

# Configure page
st.set_page_config(
    page_title="PsychCareAI - Mental Health Analysis Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for healthcare styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2E86AB 0%, #A23B72 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'ml_models' not in st.session_state:
        st.session_state.ml_models = PsychiatricMLModels()
    
    if 'nlp_processor' not in st.session_state:
        st.session_state.nlp_processor = NLPProcessor()
        st.session_state.nlp_processor.initialize_models()
    
    if 'risk_assessor' not in st.session_state:
        st.session_state.risk_assessor = RiskAssessment()
    
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    
    if 'current_patient' not in st.session_state:
        st.session_state.current_patient = None
    
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    if 'page_history' not in st.session_state:
        st.session_state.page_history = []

def load_ml_models():
    """Load ML models from saved files"""
    try:
        if not st.session_state.ml_models.is_trained():
            with st.spinner("Loading ML models..."):
                success = st.session_state.ml_models.load_models()
                if success:
                    st.success("✅ ML models loaded successfully!")
                    return True
                else:
                    st.error("❌ Failed to load ML models. Please ensure models are trained and saved.")
                    st.info("💡 Tip: Run the MH.ipynb notebook to train and save models first.")
                    return False
        return True
    except Exception as e:
        st.error(f"❌ Error loading ML models: {str(e)}")
        return False

def sidebar_navigation():
    """Create sidebar navigation"""
    st.sidebar.title("🧠 PsychCareAI")
    st.sidebar.markdown("---")
    
    # Navigation menu
    pages = {
        "Dashboard": "dashboard",
        "Patient Input": "patient_input",
        "AI Chatbot": "chatbot",
        "Analysis Results": "analysis_results"
    }
    
    selected_page = st.sidebar.selectbox(
        "Navigate to:",
        list(pages.keys()),
        index=0
    )
    
    # System status
    st.sidebar.markdown("---")
    st.sidebar.subheader(" System Status")
    
    # Check ML models status
    models_status = "✅ Loaded" if st.session_state.ml_models.is_trained() else "❌ Not Loaded"
    st.sidebar.text(f"ML Models: {models_status}")
    
    # Check available models
    if st.session_state.ml_models.is_trained():
        available_models = st.session_state.ml_models.get_available_models()
        st.sidebar.text(f"Available Models: {len(available_models)}")
        
        if st.sidebar.expander("View Models"):
            for model in available_models:
                st.sidebar.text(f"• {model}")
    
    # NLP processor status
    nlp_status = "✅ Ready" if st.session_state.nlp_processor else "❌ Not Ready"
    st.sidebar.text(f"NLP Processor: {nlp_status}")
    
    # Database status
    db_status = "✅ Connected" if st.session_state.db_manager else "❌ Disconnected"
    st.sidebar.text(f"Database: {db_status}")
    
    # Current patient info
    if st.session_state.current_patient:
        st.sidebar.markdown("---")
        st.sidebar.subheader("👤 Current Patient")
        patient = st.session_state.current_patient
        st.sidebar.text(f"ID: {patient.get('patient_id', 'N/A')}")
        st.sidebar.text(f"Age: {patient.get('OnsetAge', 'N/A')}")
        if patient.get('PresentComplaint'):
            complaint_preview = patient['PresentComplaint'][:50] + "..." if len(patient['PresentComplaint']) > 50 else patient['PresentComplaint']
            st.sidebar.text(f"Complaint: {complaint_preview}")
    
    # Quick actions
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ Quick Actions")
    
    if st.sidebar.button("🔄 Reload Models"):
        st.session_state.ml_models = PsychiatricMLModels()
        load_ml_models()
        st.rerun()
    
    if st.sidebar.button("🗑️ Clear Patient Data"):
        st.session_state.current_patient = None
        st.session_state.analysis_results = None
        st.success("Patient data cleared!")
        st.rerun()
    
    if st.sidebar.button("📋 Export Results") and st.session_state.analysis_results:
        # Create export data
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'patient_data': st.session_state.current_patient,
            'analysis_results': st.session_state.analysis_results
        }
        
        # Convert to JSON for download
        import json
        json_data = json.dumps(export_data, indent=2, default=str)
        st.sidebar.download_button(
            label="📥 Download JSON",
            data=json_data,
            file_name=f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    return pages[selected_page]

def main_header():
    """Display main application header"""
    # Header
    st.markdown("""
        <div class="main-header">
        <h1>🧠 AI-Powered Psychiatric Care Platform</h1>
        <p>Early Detection & Personalized Intervention System</p></div>
        """, unsafe_allow_html=True)

def show_alerts():
    """Show important alerts and notifications"""
    # Check for high-risk patients
    if st.session_state.analysis_results:
        risk_level = st.session_state.analysis_results.get('risk_assessment', {}).get('risk_level', 'Low')
        
        if risk_level == 'High':
            st.error("🚨 **HIGH RISK PATIENT** - Immediate intervention required!")
        elif risk_level == 'Medium':
            st.warning("⚠️ **MEDIUM RISK PATIENT** - Enhanced monitoring recommended")
    
    # Check model status
    if not st.session_state.ml_models.is_trained():
        st.warning("⚠️ ML models not loaded. Some features may be unavailable.")

def footer():
    """Display application footer"""
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**PsychCareAI v1.0**")
        st.markdown("Mental Health Analysis Platform")
    
    with col2:
        st.markdown("**Support**")
        st.markdown("For technical support, contact @ [github.com/abdullahaamir13](https://github.com/abdullahaamir13)")

    with col3:
        st.markdown("**Disclaimer**")
        st.markdown("This tool is for clinical decision support only")

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Load ML models
    models_loaded = load_ml_models()
    
    # Show main header
    main_header()
    
    # Show alerts
    show_alerts()
    
    # Get current page from sidebar
    current_page = sidebar_navigation()
    
    # Route to appropriate page
    if current_page == "dashboard":
        show_dashboard()
    elif current_page == "patient_input":
        if models_loaded:
            show_patient_input()
        else:
            st.error("❌ ML models must be loaded to use patient input features.")
            st.info("Please ensure MH.ipynb has been run to train and save models.")
    elif current_page == "chatbot":
        if models_loaded:
            show_chatbot()
        else:
            st.error("❌ ML models must be loaded to use chatbot features.")
            st.info("Please ensure MH.ipynb has been run to train and save models.")
    elif current_page == "analysis_results":
        try:
            show_analysis_results()
        except Exception as e:
            st.error(f"⚠️ Failed to render analysis results: {e}")
            import traceback
            st.text(traceback.format_exc())
    
    # Show footer
    footer()

if __name__ == "__main__":
    main()