import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
import uuid

def show_patient_input():
    """Display patient input form and process data"""
    st.header("👤 Patient Data Input")
    
    # Check if models are available
    if not st.session_state.ml_models.is_trained():
        st.error("❌ ML models not available. Please ensure models are trained and loaded.")
        return
    
    # Patient input form
    with st.form("patient_input_form"):
        st.subheader("📋 Patient Information")
        
        # Basic information
        col1, col2 = st.columns(2)
        
        with col1:
            patient_id = st.text_input(
                "Patient ID", 
                value=f"P{datetime.now().strftime('%Y%m%d%H%M%S')}",
                help="Unique identifier for the patient"
            )
            
            onset_age = st.number_input(
                "Age of Onset", 
                min_value=5, 
                max_value=100, 
                value=25,
                help="Age when symptoms first appeared"
            )
            
            duration = st.number_input(
                "Duration of Illness (years)", 
                min_value=0.0, 
                max_value=50.0, 
                value=1.0, 
                step=0.1,
                help="How long the patient has been experiencing symptoms"
            )
            
            episode = st.number_input(
                "Episode Number", 
                min_value=1, 
                max_value=20, 
                value=1,
                help="Current episode number (if recurrent)"
            )
        
        with col2:
            # Vitals
            st.markdown("**Vital Signs**")
            
            lower_bp = st.number_input(
                "Lower BP (mmHg)", 
                min_value=40, 
                max_value=150, 
                value=80
            )
            
            upper_bp = st.number_input(
                "Upper BP (mmHg)", 
                min_value=80, 
                max_value=250, 
                value=120
            )
            
            respiratory_rate = st.number_input(
                "Respiratory Rate (breaths/min)", 
                min_value=8, 
                max_value=40, 
                value=18
            )
            
            temperature = st.number_input(
                "Temperature (°F)", 
                min_value=95.0, 
                max_value=110.0, 
                value=98.6, 
                step=0.1
            )
        
        # Physical measurements
        st.markdown("**Physical Measurements**")
        col3, col4 = st.columns(2)
        
        with col3:
            weight = st.number_input(
                "Weight (kg)", 
                min_value=20.0, 
                max_value=300.0, 
                value=70.0, 
                step=0.1
            )
            
            height = st.number_input(
                "Height (m)", 
                min_value=1.0, 
                max_value=2.5, 
                value=1.70, 
                step=0.01
            )
        
        with col4:
            dose_frequency = st.selectbox(
                "Medication Dose Frequency",
                [0, 1, 2, 3, 4, 5],
                index=0,
                help="Number of medication doses per day"
            )
        
        # Substance use
        st.markdown("**Substance Use History**")
        col5, col6 = st.columns(2)
        
        with col5:
            is_substance_reported = st.checkbox("Substance Use Reported")
            is_alcohol = st.checkbox("Alcohol Use")
            is_marijuana = st.checkbox("Marijuana Use")
        
        with col6:
            is_opium = st.checkbox("Opium Use")
            is_heroin = st.checkbox("Heroin Use")
        
        # Behavioral assessment
        st.markdown("**Behavioral Assessment**")
        col7, col8, col9 = st.columns(3)
        
        with col7:
            is_hygiene_appropriate = st.checkbox("Appropriate Hygiene", value=True)
        
        with col8:
            is_posture_appropriate = st.checkbox("Appropriate Posture", value=True)
        
        with col9:
            is_behaviour_cooperative = st.checkbox("Cooperative Behavior", value=True)
        
        # Informant and relation
        col10, col11 = st.columns(2)
        
        with col10:
            informant = st.selectbox(
                "Informant",
                [0, 1, 2, 3, 4],
                index=0,
                help="Source of information (0=Self, 1=Family, 2=Friend, etc.)"
            )
        
        with col11:
            relation = st.selectbox(
                "Relation to Patient",
                [0, 1, 2, 3, 4],
                index=0,
                help="Relationship of informant to patient"
            )
        
        # Text inputs
        st.markdown("**Clinical Description**")
        
        present_complaint = st.text_area(
            "Present Complaint",
            height=120,
            placeholder="Describe the patient's current symptoms and complaints in detail...",
            help="Detailed description of current symptoms and concerns"
        )
        
        illness_history = st.text_area(
            "Illness History",
            height=100,
            placeholder="Describe the history of the patient's mental health condition...",
            help="History of the mental health condition and previous treatments"
        )
        
        # Submit button
        submitted = st.form_submit_button("🔍 Analyze Patient Data", type="primary")
        
        if submitted:
            # Validate required fields
            if not present_complaint.strip():
                st.error("❌ Present Complaint is required for analysis.")
                return
            
            # Create patient data dictionary
            patient_data = {
                'patient_id': patient_id,
                'OnsetAge': onset_age,
                'Duration': duration,
                'Episode': episode,
                'LowerBP': lower_bp,
                'UpperBP': upper_bp,
                'RespiratoryRate': respiratory_rate,
                'Temp': temperature,
                'Weight': weight,
                'Height': height,
                'DoseFrequency': dose_frequency,
                'IsSubstanceReported': 1 if is_substance_reported else 0,
                'IsAlcohol': 1 if is_alcohol else 0,
                'IsMarijuana': 1 if is_marijuana else 0,
                'IsOpium': 1 if is_opium else 0,
                'IsHeroin': 1 if is_heroin else 0,
                'IsHygieneAppropriate': 1 if is_hygiene_appropriate else 0,
                'IsPostureAppropriate': 1 if is_posture_appropriate else 0,
                'IsBehaviourCooperative': 1 if is_behaviour_cooperative else 0,
                'Informant': informant,
                'Relation': relation,
                'PresentComplaint': present_complaint,
                'IllnessHistory': illness_history,
                'timestamp': datetime.now().isoformat()
            }
            
            # Store patient data in session state
            st.session_state.current_patient = patient_data
            
            # Process patient data
            process_patient_data(patient_data)

def process_patient_data(patient_data):
    """Process patient data through ML pipeline"""
    try:
        st.subheader("🔄 Processing Patient Data...")
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: NLP Analysis
        status_text.text("Analyzing text data...")
        progress_bar.progress(20)
        
        nlp_analysis = st.session_state.nlp_processor.extract_symptoms_from_complaint(
            patient_data['PresentComplaint']
        )
        
        # Step 2: Prepare features for ML
        status_text.text("Preparing features for ML models...")
        progress_bar.progress(40)
        
        # Combine all patient data for feature extraction
        ml_features = prepare_ml_features(patient_data, nlp_analysis)
        
        # Step 3: ML Predictions
        status_text.text("Running ML predictions...")
        progress_bar.progress(60)
        
        # Get predictions from all models
        predictions = st.session_state.ml_models.predict_disorder(ml_features)
        
        # Get ensemble prediction
        ensemble_prediction = st.session_state.ml_models.get_ensemble_prediction(ml_features)
        
        # Step 4: Risk Assessment
        status_text.text("Calculating risk assessment...")
        progress_bar.progress(80)
        
        risk_scores = st.session_state.risk_assessor.calculate_risk_score(
            patient_data, ensemble_prediction, nlp_analysis
        )
        risk_level = st.session_state.risk_assessor.get_risk_level(risk_scores['overall_risk'])
        interventions = st.session_state.risk_assessor.get_interventions(risk_level, patient_data, nlp_analysis)
        
        # Step 5: Compile results
        status_text.text("Compiling analysis results...")
        progress_bar.progress(100)
        
        analysis_results = {
            'patient_id': patient_data['patient_id'],
            'timestamp': datetime.now().isoformat(),
            'nlp_analysis': nlp_analysis,
            'ml_predictions': {
                'individual_models': predictions,
                'ensemble': ensemble_prediction
            },
            'risk_assessment': {
                'risk_scores': risk_scores,
                'risk_level': risk_level,
                'interventions': interventions
            },
            'feature_analysis': get_feature_analysis(ml_features)
        }
        
        # Store results in session state
        st.session_state.analysis_results = analysis_results
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Show success message
        st.success("✅ Analysis completed successfully!")
        
        # Show quick results preview
        show_quick_results_preview(analysis_results)
        
        # Navigation prompt
        st.info("📊 View detailed results in the 'Analysis Results' section.")
        
    except Exception as e:
        st.error(f"❌ Error processing patient data: {str(e)}")
        st.exception(e)

def prepare_ml_features(patient_data, nlp_analysis):
    """Prepare features for ML model input"""
    try:
        # Get NLP features from text
        nlp_features = st.session_state.nlp_processor.process_patient_text_data(patient_data)
        
        # Create comprehensive feature dictionary
        ml_features = {}
        
        # Add numerical features
        numerical_features = [
            'OnsetAge', 'Duration', 'Episode', 'LowerBP', 'UpperBP', 
            'RespiratoryRate', 'Temp', 'Weight', 'Height', 'DoseFrequency'
        ]
        
        for feature in numerical_features:
            if feature in patient_data:
                ml_features[feature] = patient_data[feature]
        
        # Add binary features
        binary_features = [
            'IsSubstanceReported', 'IsAlcohol', 'IsMarijuana', 'IsOpium', 
            'IsHeroin', 'IsHygieneAppropriate', 'IsPostureAppropriate', 
            'IsBehaviourCooperative'
        ]
        
        for feature in binary_features:
            if feature in patient_data:
                ml_features[feature] = patient_data[feature]
        
        # Add categorical features
        categorical_features = ['Informant', 'Relation']
        for feature in categorical_features:
            if feature in patient_data:
                ml_features[feature] = patient_data[feature]
        
        # Add derived features
        if 'Weight' in ml_features and 'Height' in ml_features and ml_features['Height'] > 0:
            ml_features['BMI'] = ml_features['Weight'] / (ml_features['Height'] ** 2)
        
        if 'LowerBP' in ml_features and 'UpperBP' in ml_features:
            ml_features['BP_Diff'] = ml_features['UpperBP'] - ml_features['LowerBP']
            ml_features['Mean_BP'] = (ml_features['UpperBP'] + ml_features['LowerBP']) / 2
        
        # Add NLP features
        ml_features.update(nlp_features)
        
        # Add interaction features
        substance_features = ['IsAlcohol', 'IsMarijuana', 'IsOpium', 'IsHeroin']
        ml_features['Total_Substances'] = sum(ml_features.get(f, 0) for f in substance_features)
        ml_features['Poly_Substance_Use'] = 1 if ml_features['Total_Substances'] > 1 else 0
        
        behavior_features = ['IsHygieneAppropriate', 'IsPostureAppropriate', 'IsBehaviourCooperative']
        ml_features['Behavioral_Score'] = sum(ml_features.get(f, 0) for f in behavior_features)
        
        # Age-duration interactions
        if 'OnsetAge' in ml_features and 'Duration' in ml_features:
            ml_features['Age_Duration_Ratio'] = ml_features['OnsetAge'] / (ml_features['Duration'] + 1)
            ml_features['Age_Duration_Product'] = ml_features['OnsetAge'] * ml_features['Duration']
        
        return ml_features
        
    except Exception as e:
        st.error(f"Error preparing ML features: {str(e)}")
        return {}

def get_feature_analysis(ml_features):
    """Get analysis of the prepared features"""
    try:
        # Get risk factors from ML models
        risk_factors = st.session_state.ml_models.get_risk_factors(ml_features)
        
        # Feature summary
        feature_summary = {
            'total_features': len(ml_features),
            'non_zero_features': sum(1 for v in ml_features.values() if v != 0),
            'feature_types': {
                'clinical': len([f for f in ml_features.keys() if any(term in f.lower() for term in ['bp', 'temp', 'weight', 'height'])]),
                'behavioral': len([f for f in ml_features.keys() if any(term in f.lower() for term in ['hygiene', 'behavior', 'substance'])]),
                'nlp': len([f for f in ml_features.keys() if any(term in f.lower() for term in ['sentiment', 'keyword', 'text'])]),
                'demographic': len([f for f in ml_features.keys() if any(term in f.lower() for term in ['age', 'duration', 'episode'])])
            },
            'risk_factors': risk_factors
        }
        
        return feature_summary
        
    except Exception as e:
        st.error(f"Error in feature analysis: {str(e)}")
        return {}

def show_quick_results_preview(analysis_results):
    """Show quick preview of analysis results"""
    st.subheader("📋 Quick Results Preview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Risk level
        risk_level = analysis_results['risk_assessment']['risk_level']
        risk_color = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}.get(risk_level, '⚪')
        st.metric(
            label="Risk Level",
            value=f"{risk_color} {risk_level}",
            delta=f"Score: {analysis_results['risk_assessment']['risk_scores']['overall_risk']:.3f}"
        )
    
    with col2:
        # Top prediction
        ensemble = analysis_results['ml_predictions']['ensemble']
        if ensemble and not ('error' in ensemble):
            top_prediction = max(ensemble.items(), key=lambda x: x[1])
            st.metric(
                label="Top Prediction",
                value=top_prediction[0],
                delta=f"Confidence: {top_prediction[1]:.1%}"
            )
        else:
            st.metric(label="Top Prediction", value="Error", delta="Check models")
    
    with col3:
        # Symptom severity
        severity = analysis_results['nlp_analysis'].get('severity', 'Unknown')
        severity_color = {'severe': '🔴', 'moderate': '🟡', 'mild': '🟢'}.get(severity.lower(), '⚪')
        st.metric(
            label="Symptom Severity",
            value=f"{severity_color} {severity.title()}",
            delta=f"Score: {analysis_results['nlp_analysis'].get('severity_score', 0)}"
        )
    
    # Key findings
    st.markdown("**Key Findings:**")
    
    # NLP insights
    symptoms = analysis_results['nlp_analysis'].get('symptoms', [])
    if symptoms:
        st.markdown(f"• Identified {len(symptoms)} symptom categories")
        for symptom in symptoms[:3]:  # Show top 3
            st.markdown(f"  - {symptom['category'].replace('_', ' ').title()}: {symptom['count']} keywords")
    
    # Risk factors
    risk_factors = analysis_results.get('feature_analysis', {}).get('risk_factors', [])
    if risk_factors:
        st.markdown(f"• Top risk factors identified:")
        for factor in risk_factors[:3]:  # Show top 3
            st.markdown(f"  - {factor['feature']}: {factor['risk_level']} risk")

def show_patient_summary():
    """Show summary of current patient data"""
    if not st.session_state.current_patient:
        st.info("No patient data available")
        return
    
    st.subheader("👤 Current Patient Summary")
    
    patient = st.session_state.current_patient
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Basic Information:**")
        st.text(f"Patient ID: {patient.get('patient_id', 'N/A')}")
        st.text(f"Age of Onset: {patient.get('OnsetAge', 'N/A')}")
        st.text(f"Duration: {patient.get('Duration', 'N/A')} years")
        st.text(f"Episode: {patient.get('Episode', 'N/A')}")
    
    with col2:
        st.markdown("**Vitals:**")
        st.text(f"BP: {patient.get('LowerBP', 'N/A')}/{patient.get('UpperBP', 'N/A')} mmHg")
        st.text(f"Temperature: {patient.get('Temp', 'N/A')} °F")
        st.text(f"Respiratory Rate: {patient.get('RespiratoryRate', 'N/A')} bpm")
        
        # Calculate BMI if available
        if patient.get('Weight') and patient.get('Height'):
            bmi = patient['Weight'] / (patient['Height'] ** 2)
            st.text(f"BMI: {bmi:.1f}")
    
    # Substance use summary
    substance_use = []
    for substance in ['IsAlcohol', 'IsMarijuana', 'IsOpium', 'IsHeroin']:
        if patient.get(substance, 0) == 1:
            substance_use.append(substance.replace('Is', ''))
    
    if substance_use:
        st.markdown(f"**Substance Use:** {', '.join(substance_use)}")
    
    # Present complaint preview
    if patient.get('PresentComplaint'):
        st.markdown("**Present Complaint (Preview):**")
        complaint_preview = patient['PresentComplaint'][:200] + "..." if len(patient['PresentComplaint']) > 200 else patient['PresentComplaint']
        st.text_area("", value=complaint_preview, height=100, disabled=True)

# Add patient summary to the main function
def show_patient_input():
    """Enhanced patient input with summary"""
    # Show current patient summary if available
    if st.session_state.current_patient:
        show_patient_summary()
        st.markdown("---")
        
        if st.button("🆕 New Patient"):
            st.session_state.current_patient = None
            st.session_state.analysis_results = None
            st.rerun()
    
    # Show input form (original function content)
    _show_patient_input_form()

def _show_patient_input_form():
    """Original patient input form function"""
    st.header("👤 Patient Data Input")
    
    # Check if models are available
    if not st.session_state.ml_models.is_trained():
        st.error("❌ ML models not available. Please ensure models are trained and loaded.")
        return
    
    # [Rest of the original show_patient_input function code...]
    # This includes all the form elements and processing logic
    
    # Patient input form
    with st.form("patient_input_form"):
        st.subheader("📋 Patient Information")
        
        # Basic information
        col1, col2 = st.columns(2)
        
        with col1:
            patient_id = st.text_input(
                "Patient ID", 
                value=f"P{datetime.now().strftime('%Y%m%d%H%M%S')}",
                help="Unique identifier for the patient"
            )
            
            onset_age = st.number_input(
                "Age of Onset", 
                min_value=5, 
                max_value=100, 
                value=25,
                help="Age when symptoms first appeared"
            )
            
            duration = st.number_input(
                "Duration of Illness (years)", 
                min_value=0.0, 
                max_value=50.0, 
                value=1.0, 
                step=0.1,
                help="How long the patient has been experiencing symptoms"
            )
            
            episode = st.number_input(
                "Episode Number", 
                min_value=1, 
                max_value=20, 
                value=1,
                help="Current episode number (if recurrent)"
            )
        
        with col2:
            # Vitals
            st.markdown("**Vital Signs**")
            
            lower_bp = st.number_input(
                "Lower BP (mmHg)", 
                min_value=40, 
                max_value=150, 
                value=80
            )
            
            upper_bp = st.number_input(
                "Upper BP (mmHg)", 
                min_value=80, 
                max_value=250, 
                value=120
            )
            
            respiratory_rate = st.number_input(
                "Respiratory Rate (breaths/min)", 
                min_value=8, 
                max_value=40, 
                value=18
            )
            
            temperature = st.number_input(
                "Temperature (°F)", 
                min_value=95.0, 
                max_value=110.0, 
                value=98.6, 
                step=0.1
            )
        
        # Physical measurements
        st.markdown("**Physical Measurements**")
        col3, col4 = st.columns(2)
        
        with col3:
            weight = st.number_input(
                "Weight (kg)", 
                min_value=20.0, 
                max_value=300.0, 
                value=70.0, 
                step=0.1
            )
            
            height = st.number_input(
                "Height (m)", 
                min_value=1.0, 
                max_value=2.5, 
                value=1.70, 
                step=0.01
            )
        
        with col4:
            dose_frequency = st.selectbox(
                "Medication Dose Frequency",
                [0, 1, 2, 3, 4, 5],
                index=0,
                help="Number of medication doses per day"
            )
        
        # Substance use
        st.markdown("**Substance Use History**")
        col5, col6 = st.columns(2)
        
        with col5:
            is_substance_reported = st.checkbox("Substance Use Reported")
            is_alcohol = st.checkbox("Alcohol Use")
            is_marijuana = st.checkbox("Marijuana Use")
        
        with col6:
            is_opium = st.checkbox("Opium Use")
            is_heroin = st.checkbox("Heroin Use")
        
        # Behavioral assessment
        st.markdown("**Behavioral Assessment**")
        col7, col8, col9 = st.columns(3)
        
        with col7:
            is_hygiene_appropriate = st.checkbox("Appropriate Hygiene", value=True)
        
        with col8:
            is_posture_appropriate = st.checkbox("Appropriate Posture", value=True)
        
        with col9:
            is_behaviour_cooperative = st.checkbox("Cooperative Behavior", value=True)
        
        # Informant and relation
        col10, col11 = st.columns(2)
        
        with col10:
            informant = st.selectbox(
                "Informant",
                [0, 1, 2, 3, 4],
                index=0,
                help="Source of information (0=Self, 1=Family, 2=Friend, etc.)"
            )
        
        with col11:
            relation = st.selectbox(
                "Relation to Patient",
                [0, 1, 2, 3, 4],
                index=0,
                help="Relationship of informant to patient"
            )
        
        # Text inputs
        st.markdown("**Clinical Description**")
        
        present_complaint = st.text_area(
            "Present Complaint",
            height=120,
            placeholder="Describe the patient's current symptoms and complaints in detail...",
            help="Detailed description of current symptoms and concerns"
        )
        
        illness_history = st.text_area(
            "Illness History",
            height=100,
            placeholder="Describe the history of the patient's mental health condition...",
            help="History of the mental health condition and previous treatments"
        )
        
        # Submit button
        submitted = st.form_submit_button("🔍 Analyze Patient Data", type="primary")
        
        if submitted:
            # Validate required fields
            if not present_complaint.strip():
                st.error("❌ Present Complaint is required for analysis.")
                return
            
            # Create patient data dictionary
            patient_data = {
                'patient_id': patient_id,
                'OnsetAge': onset_age,
                'Duration': duration,
                'Episode': episode,
                'LowerBP': lower_bp,
                'UpperBP': upper_bp,
                'RespiratoryRate': respiratory_rate,
                'Temp': temperature,
                'Weight': weight,
                'Height': height,
                'DoseFrequency': dose_frequency,
                'IsSubstanceReported': 1 if is_substance_reported else 0,
                'IsAlcohol': 1 if is_alcohol else 0,
                'IsMarijuana': 1 if is_marijuana else 0,
                'IsOpium': 1 if is_opium else 0,
                'IsHeroin': 1 if is_heroin else 0,
                'IsHygieneAppropriate': 1 if is_hygiene_appropriate else 0,
                'IsPostureAppropriate': 1 if is_posture_appropriate else 0,
                'IsBehaviourCooperative': 1 if is_behaviour_cooperative else 0,
                'Informant': informant,
                'Relation': relation,
                'PresentComplaint': present_complaint,
                'IllnessHistory': illness_history,
                'timestamp': datetime.now().isoformat()
            }
            
            # Store patient data in session state
            st.session_state.current_patient = patient_data
            
            # Process patient data
            process_patient_data(patient_data)
