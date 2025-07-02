import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import json

def show_analysis_results():
    """Display comprehensive analysis results for the current patient."""
    st.header("📊 Analysis Results")

    # Safety: check required session variables
    if 'analysis_results' not in st.session_state or not st.session_state.analysis_results:
        st.warning("⚠️ No analysis results available. Please complete patient assessment first.")
        if st.button("➡️ Go to Patient Input"):
            st.session_state.current_page = "patient_input"
            st.rerun()
        return

    if 'current_patient' not in st.session_state or not st.session_state.current_patient:
        st.warning("⚠️ No patient data found.")
        return

    results = st.session_state.analysis_results
    patient_data = st.session_state.current_patient

    # Tabs for detailed breakdown
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Summary", 
        "🤖 ML Predictions", 
        "⚠️ Risk Assessment", 
        "📝 NLP Analysis", 
        "📈 Detailed Report"
    ])

    with tab1:
        show_summary_results(results, patient_data)

    with tab2:
        show_ml_predictions(results)

    with tab3:
        show_risk_assessment(results)

    with tab4:
        show_nlp_analysis(results)

    with tab5:
        show_detailed_report(results, patient_data)

def show_summary_results(results, patient_data):
    """Display summary of analysis results"""
    st.subheader("🎯 Analysis Summary")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Risk level
        risk_level = results['risk_assessment']['risk_level']
        risk_score = results['risk_assessment']['risk_scores']['overall_risk']
        risk_colors = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}
        risk_color = risk_colors.get(risk_level, '⚪')
        
        st.metric(
            label="Risk Level",
            value=f"{risk_color} {risk_level}",
            delta=f"Score: {risk_score:.3f}"
        )
    
    with col2:
        # Top ML prediction
        ensemble = results['ml_predictions']['ensemble']
        if ensemble and 'error' not in ensemble:
            top_prediction = max(ensemble.items(), key=lambda x: x[1])
            st.metric(
                label="Primary Prediction",
                value=top_prediction[0],
                delta=f"{top_prediction[1]:.1%} confidence"
            )
        else:
            st.metric(label="Primary Prediction", value="Error", delta="Check models")
    
    with col3:
        # Symptom severity
        severity = results['nlp_analysis'].get('severity', 'Unknown')
        severity_score = results['nlp_analysis'].get('severity_score', 0)
        severity_colors = {'severe': '🔴', 'moderate': '🟡', 'mild': '🟢'}
        severity_color = severity_colors.get(severity.lower(), '⚪')
        
        st.metric(
            label="Symptom Severity",
            value=f"{severity_color} {severity.title()}",
            delta=f"Score: {severity_score}"
        )
    
    with col4:
        # Urgency level
        urgency = results['nlp_analysis'].get('urgency', 'Low')
        urgency_colors = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}
        urgency_color = urgency_colors.get(urgency, '⚪')
        
        st.metric(
            label="Urgency Level",
            value=f"{urgency_color} {urgency}",
            delta="Based on text analysis"
        )
    
    # Patient information summary
    st.markdown("---")
    st.subheader("👤 Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Basic Information:**")
        st.text(f"Patient ID: {patient_data.get('patient_id', 'N/A')}")
        st.text(f"Age of Onset: {patient_data.get('OnsetAge', 'N/A')} years")
        st.text(f"Duration of Illness: {patient_data.get('Duration', 'N/A')} years")
        st.text(f"Episode Number: {patient_data.get('Episode', 'N/A')}")
        
        # Calculate and display BMI if available
        if patient_data.get('Weight') and patient_data.get('Height'):
            bmi = patient_data['Weight'] / (patient_data['Height'] ** 2)
            bmi_category = get_bmi_category(bmi)
            st.text(f"BMI: {bmi:.1f} ({bmi_category})")
    
    with col2:
        st.markdown("**Clinical Indicators:**")
        
        # Vital signs
        bp_lower = patient_data.get('LowerBP', 'N/A')
        bp_upper = patient_data.get('UpperBP', 'N/A')
        st.text(f"Blood Pressure: {bp_lower}/{bp_upper} mmHg")
        st.text(f"Temperature: {patient_data.get('Temp', 'N/A')} °F")
        st.text(f"Respiratory Rate: {patient_data.get('RespiratoryRate', 'N/A')} bpm")
        
        # Substance use summary
        substances = []
        for substance in ['IsAlcohol', 'IsMarijuana', 'IsOpium', 'IsHeroin']:
            if patient_data.get(substance, 0) == 1:
                substances.append(substance.replace('Is', ''))
        
        if substances:
            st.text(f"Substance Use: {', '.join(substances)}")
        else:
            st.text("Substance Use: None reported")
    
    # Key findings
    st.markdown("---")
    show_key_findings(results)

def show_key_findings(results):
    """Display key findings from the analysis"""
    st.subheader("🔍 Key Findings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Clinical Highlights:**")
        
        # Risk factors
        risk_factors = results.get('feature_analysis', {}).get('risk_factors', [])
        if risk_factors:
            st.markdown("• **Top Risk Factors:**")
            for factor in risk_factors[:3]:
                st.markdown(f"  - {factor['feature']}: {factor['risk_level']} risk")
        
        # ML model agreement
        individual_models = results['ml_predictions']['individual_models']
        if individual_models and 'error' not in individual_models:
            predictions_list = []
            for model_name, predictions in individual_models.items():
                if predictions:
                    top_pred = max(predictions.items(), key=lambda x: x[1])
                    predictions_list.append(top_pred[0])
            
            if predictions_list:
                from collections import Counter
                prediction_counts = Counter(predictions_list)
                most_common = prediction_counts.most_common(1)[0]
                agreement = most_common[1] / len(predictions_list) * 100
                
                st.markdown(f"• **Model Agreement:** {agreement:.0f}% consensus on {most_common[0]}")
    
    with col2:
        st.markdown("**Symptom Analysis:**")
        
        # Symptoms identified
        symptoms = results['nlp_analysis'].get('symptoms', [])
        if symptoms:
            st.markdown(f"• **Symptom Categories:** {len(symptoms)} identified")
            for symptom in symptoms:
                category = symptom['category'].replace('_', ' ').title()
                count = symptom['count']
                st.markdown(f"  - {category}: {count} keywords")
        
        # Sentiment analysis
        sentiment = results['nlp_analysis'].get('sentiment', {})
        if sentiment:
            compound = sentiment.get('compound', 0)
            if compound < -0.3:
                sentiment_desc = "Negative"
            elif compound > 0.3:
                sentiment_desc = "Positive"
            else:
                sentiment_desc = "Neutral"
            
            st.markdown(f"• **Emotional Tone:** {sentiment_desc} (score: {compound:.3f})")

def show_ml_predictions(results):
    """Display ML model predictions and analysis"""
    st.subheader("🤖 Machine Learning Predictions")
    
    # Individual model predictions
    individual_models = results['ml_predictions']['individual_models']
    ensemble = results['ml_predictions']['ensemble']
    
    if 'error' in individual_models:
        st.error(f"❌ ML Prediction Error: {individual_models['error']}")
        return
    
    # Ensemble prediction visualization
    if ensemble and 'error' not in ensemble:
        st.markdown("### 🎯 Ensemble Prediction (Combined Models)")
        
        # Create ensemble prediction chart
        ensemble_df = pd.DataFrame([
            {'Condition': condition, 'Probability': probability}
            for condition, probability in ensemble.items()
        ]).sort_values('Probability', ascending=True)
        
        fig = px.bar(
            ensemble_df,
            x='Probability',
            y='Condition',
            orientation='h',
            title="Ensemble Model Predictions",
            color='Probability',
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Top predictions table
        top_predictions = ensemble_df.tail(5)
        st.markdown("**Top 5 Predictions:**")
        for _, row in top_predictions.iterrows():
            st.text(f"• {row['Condition']}: {row['Probability']:.1%}")
    
    # Individual model comparison
    st.markdown("---")
    st.markdown("### 🔍 Individual Model Analysis")
    
    if individual_models:
        # Create comparison visualization
        model_data = []
        all_conditions = set()
        
        for model_name, predictions in individual_models.items():
            if predictions:
                all_conditions.update(predictions.keys())
                for condition, probability in predictions.items():
                    model_data.append({
                        'Model': model_name,
                        'Condition': condition,
                        'Probability': probability
                    })
        
        if model_data:
            model_df = pd.DataFrame(model_data)
            
            # Get top conditions for visualization
            avg_probs = model_df.groupby('Condition')['Probability'].mean().sort_values(ascending=False)
            top_conditions = avg_probs.head(8).index.tolist()
            
            # Filter data for top conditions
            filtered_df = model_df[model_df['Condition'].isin(top_conditions)]
            
            # Create heatmap
            pivot_df = filtered_df.pivot(index='Condition', columns='Model', values='Probability')
            
            fig = px.imshow(
                pivot_df.values,
                x=pivot_df.columns,
                y=pivot_df.index,
                title="Model Predictions Heatmap (Top Conditions)",
                color_continuous_scale='RdYlGn_r',
                aspect="auto"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Model performance comparison
            st.markdown("### 📊 Model Comparison")
            
            # Calculate model statistics
            model_stats = []
            for model_name in individual_models.keys():
                model_preds = model_df[model_df['Model'] == model_name]
                if not model_preds.empty:
                    max_prob = model_preds['Probability'].max()
                    top_prediction = model_preds.loc[model_preds['Probability'].idxmax(), 'Condition']
                    confidence_variance = model_preds['Probability'].var()
                    
                    model_stats.append({
                        'Model': model_name,
                        'Top Prediction': top_prediction,
                        'Max Confidence': f"{max_prob:.1%}",
                        'Prediction Variance': f"{confidence_variance:.4f}"
                    })
            
            if model_stats:
                stats_df = pd.DataFrame(model_stats)
                st.dataframe(stats_df, use_container_width=True)

def show_risk_assessment(results):
    """Display comprehensive risk assessment"""
    st.subheader("⚠️ Risk Assessment")
    
    risk_info = results['risk_assessment']
    risk_scores = risk_info['risk_scores']
    risk_level = risk_info['risk_level']
    interventions = risk_info['interventions']
    
    # Overall risk visualization
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Risk gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_scores['overall_risk'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Overall Risk Score"},
            delta={'reference': 0.5},
            gauge={
                'axis': {'range': [None, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.4], 'color': "lightgreen"},
                    {'range': [0.4, 0.7], 'color': "yellow"},
                    {'range': [0.7, 1], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.7
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk breakdown
        categories = ['Clinical', 'Psychological', 'Behavioral', 'Demographic']
        scores = [
            risk_scores.get('clinical', 0),
            risk_scores.get('psychological', 0),
            risk_scores.get('behavioral', 0),
            risk_scores.get('demographic', 0)
        ]
        
        fig = go.Figure([go.Bar(
            x=categories,
            y=scores,
            text=[f"{score:.3f}" for score in scores],
            textposition='auto',
            marker_color=['lightcoral', 'lightsalmon', 'lightblue', 'lightgreen']
        )])
        
        fig.update_layout(
            title="Risk Score Breakdown by Category",
            yaxis_title="Risk Score",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk level alert
    if risk_level == 'High':
        st.error(f"🚨 **HIGH RISK PATIENT** - Immediate intervention required!")
    elif risk_level == 'Medium':
        st.warning(f"⚠️ **MEDIUM RISK PATIENT** - Enhanced monitoring recommended")
    else:
        st.success(f"✅ **LOW RISK PATIENT** - Continue routine care")
    
    # Interventions and recommendations
    st.markdown("---")
    st.markdown("### 💊 Recommended Interventions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Immediate Actions:**")
        immediate_actions = interventions.get('immediate_actions', [])
        for action in immediate_actions:
            st.markdown(f"• {action}")
    
    with col2:
        st.markdown("**Treatment Options:**")
        treatment_options = interventions.get('treatment_options', [])
        for treatment in treatment_options:
            st.markdown(f"• {treatment}")
    
    with col3:
        st.markdown("**Monitoring Plan:**")
        monitoring = interventions.get('monitoring', [])
        for monitor in monitoring:
            st.markdown(f"• {monitor}")
    
    # Risk factors analysis
    if 'feature_analysis' in results and results['feature_analysis'].get('risk_factors'):
        st.markdown("---")
        st.markdown("### 🔍 Risk Factors Analysis")
        
        risk_factors = results['feature_analysis']['risk_factors']
        
        # Create risk factors visualization
        if risk_factors:
            risk_df = pd.DataFrame(risk_factors)
            
            fig = px.scatter(
                risk_df,
                x='value',
                y='importance',
                size='importance',
                color='risk_level',
                hover_data=['feature'],
                title="Risk Factors: Importance vs Value",
                color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Top risk factors table
            st.markdown("**Top Risk Factors:**")
            risk_table = risk_df[['feature', 'importance', 'risk_level', 'value']].round(4)
            st.dataframe(risk_table, use_container_width=True)

def show_nlp_analysis(results):
    """Display Natural Language Processing analysis results"""
    st.subheader("📝 NLP Analysis")
    
    nlp_results = results['nlp_analysis']
    
    # Sentiment analysis
    st.markdown("### 😊 Sentiment Analysis")
    
    sentiment = nlp_results.get('sentiment', {})
    if sentiment:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Compound", f"{sentiment.get('compound', 0):.3f}")
        with col2:
            st.metric("Positive", f"{sentiment.get('pos', 0):.3f}")
        with col3:
            st.metric("Neutral", f"{sentiment.get('neu', 0):.3f}")
        with col4:
            st.metric("Negative", f"{sentiment.get('neg', 0):.3f}")
        
        # Sentiment visualization
        sentiment_data = {
            'Aspect': ['Positive', 'Neutral', 'Negative'],
            'Score': [sentiment.get('pos', 0), sentiment.get('neu', 0), sentiment.get('neg', 0)]
        }
        
        fig = px.bar(
            sentiment_data,
            x='Aspect',
            y='Score',
            title="Sentiment Breakdown",
            color='Aspect',
            color_discrete_map={'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Symptom categories
    st.markdown("---")
    st.markdown("### 🏥 Symptom Categories Identified")
    
    symptoms = nlp_results.get('symptoms', [])
    symptom_categories = nlp_results.get('symptom_categories', {})
    
    if symptoms:
        # Create symptom overview
        symptom_data = []
        for symptom in symptoms:
            category = symptom['category'].replace('_', ' ').title()
            count = symptom['count']
            
            # Get severity if available
            severity = 1
            if symptom['category'] in symptom_categories:
                severity = symptom_categories[symptom['category']].get('severity', 1)
            
            symptom_data.append({
                'Category': category,
                'Keyword Count': count,
                'Severity': severity,
                'Keywords': ', '.join(symptom.get('keywords', []))
            })
        
        symptom_df = pd.DataFrame(symptom_data)
        
        # Symptom visualization
        fig = px.scatter(
            symptom_df,
            x='Keyword Count',
            y='Severity',
            size='Keyword Count',
            color='Category',
            hover_data=['Keywords'],
            title="Symptom Categories: Count vs Severity"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed symptom table
        st.markdown("**Detailed Symptom Analysis:**")
        st.dataframe(symptom_df, use_container_width=True)
    else:
        st.info("No specific symptom categories identified in the text analysis.")
    
    # Text characteristics
    st.markdown("---")
    st.markdown("### 📊 Text Characteristics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        severity = nlp_results.get('severity', 'Unknown')
        severity_score = nlp_results.get('severity_score', 0)
        urgency = nlp_results.get('urgency', 'Low')
        
        st.markdown(f"**Overall Severity:** {severity.title()}")
        st.markdown(f"**Severity Score:** {severity_score}")
        st.markdown(f"**Urgency Level:** {urgency}")
    
    with col2:
        # Text statistics
        if st.session_state.current_patient:
            complaint_text = st.session_state.current_patient.get('PresentComplaint', '')
            if complaint_text:
                word_count = len(complaint_text.split())
                char_count = len(complaint_text)
                sentence_count = len(complaint_text.split('.'))
                
                st.markdown(f"**Word Count:** {word_count}")
                st.markdown(f"**Character Count:** {char_count}")
                st.markdown(f"**Sentences:** {sentence_count}")

def show_detailed_report(results, patient_data):
    """Display comprehensive detailed report"""
    st.subheader("📈 Detailed Analysis Report")
    
    # Report header
    st.markdown(f"**Analysis Report for Patient {patient_data.get('patient_id', 'Unknown')}**")
    st.markdown(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.markdown("---")
    
    # Executive summary
    st.markdown("### 📋 Executive Summary")
    
    risk_level = results['risk_assessment']['risk_level']
    risk_score = results['risk_assessment']['risk_scores']['overall_risk']
    ensemble = results['ml_predictions']['ensemble']
    top_prediction, top_confidence = "Unknown", 0
    
    if ensemble and 'error' not in ensemble:
        top_pred_item = max(ensemble.items(), key=lambda x: x[1])
        top_prediction = top_pred_item[0]
        top_confidence = top_pred_item[1]
    
    severity = results['nlp_analysis'].get('severity', 'Unknown')
    urgency = results['nlp_analysis'].get('urgency', 'Low')
    
    summary_text = f"""
    **Patient Profile:**
    - Age of onset: {patient_data.get('OnsetAge', 'N/A')} years
    - Duration of illness: {patient_data.get('Duration', 'N/A')} years
    - Current episode: {patient_data.get('Episode', 'N/A')}

    **Clinical Assessment:**
    - Risk Level: {risk_level} (Score: {risk_score:.3f})
    - Primary AI Prediction: {top_prediction} ({top_confidence:.1%} confidence)
    - Symptom Severity: {severity.title()}
    - Urgency Level: {urgency}

    **Key Recommendations:**
    Based on the comprehensive analysis, this patient presents as {risk_level.lower()} risk and requires 
    {'immediate intervention' if risk_level == 'High' else 'enhanced monitoring' if risk_level == 'Medium' else 'routine follow-up'}.
    """
    st.markdown(summary_text)

    # Clinical indicators
    st.markdown("---")
    st.markdown("### 🩺 Clinical Indicators")

    # Compute BMI safely once
    bmi_val = get_bmi(patient_data)
    bmi_cat = get_bmi_category(bmi_val) if bmi_val is not None else "Unknown"
    bmi_str = f"{bmi_val:.1f} ({bmi_cat})" if bmi_val is not None else "N/A"

    clinical_data = {
        'Vital Signs': [
            f"Blood Pressure: {patient_data.get('LowerBP', 'N/A')}/{patient_data.get('UpperBP', 'N/A')} mmHg",
            f"Temperature: {patient_data.get('Temp', 'N/A')} °F",
            f"Respiratory Rate: {patient_data.get('RespiratoryRate', 'N/A')} bpm"
        ],
        'Physical': [
            f"Weight: {patient_data.get('Weight', 'N/A')} kg",
            f"Height: {patient_data.get('Height', 'N/A')} m",
            f"BMI: {bmi_str}"
        ],
        'Behavioral': [
            f"Hygiene: {'Appropriate' if patient_data.get('IsHygieneAppropriate') else 'Inappropriate'}",
            f"Posture: {'Appropriate' if patient_data.get('IsPostureAppropriate') else 'Inappropriate'}",
            f"Cooperation: {'Cooperative' if patient_data.get('IsBehaviourCooperative') else 'Uncooperative'}"
        ]
    }

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Vital Signs:**")
        for item in clinical_data['Vital Signs']:
            st.text(item)
    with col2:
        st.markdown("**Physical Measurements:**")
        for item in clinical_data['Physical']:
            st.text(item)
    with col3:
        st.markdown("**Behavioral Assessment:**")
        for item in clinical_data['Behavioral']:
            st.text(item)

    # AI Analysis Summary
    st.markdown("---")
    st.markdown("### 🤖 AI Analysis Summary")
    individual_models = results['ml_predictions']['individual_models']
    if individual_models and 'error' not in individual_models:
        st.markdown("**Model Predictions:**")
        for model_name, predictions in individual_models.items():
            if predictions:
                top_pred = max(predictions.items(), key=lambda x: x[1])
                st.text(f"• {model_name}: {top_pred[0]} ({top_pred[1]:.1%})")

    st.markdown("**Text Analysis Insights:**")
    sentiment = results['nlp_analysis'].get('sentiment', {})
    st.text(f"• Emotional tone: {get_sentiment_description(sentiment.get('compound', 0))}")
    symptoms = results['nlp_analysis'].get('symptoms', [])
    if symptoms:
        st.text(f"• Symptom categories identified: {len(symptoms)}")
        for symptom in symptoms[:3]:
            category = symptom['category'].replace('_', ' ').title()
            st.text(f"  - {category}: {symptom['count']} keywords")

    # Export options
    st.markdown("---")
    st.markdown("### 📤 Export Options")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📋 Copy Summary"):
            summary_for_copy = create_text_summary(results, patient_data)
            st.text_area("Copy this summary:", value=summary_for_copy, height=200)

    with col2:
        export_data = {
            'patient_data': patient_data,
            'analysis_results': results,
            'export_timestamp': datetime.now().isoformat()
        }
        json_data = json.dumps(export_data, indent=2, default=str)
        st.download_button(
            label="📥 Download JSON",
            data=json_data,
            file_name=f"analysis_{patient_data.get('patient_id', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    with col3:
        text_report = create_detailed_text_report(results, patient_data)
        st.download_button(
            label="📄 Download Report",
            data=text_report,
            file_name=f"report_{patient_data.get('patient_id', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

# Helper functions
def get_bmi(patient_data):
    """Calculate BMI from patient data"""
    weight = patient_data.get('Weight')
    height = patient_data.get('Height')
    
    if weight and height and height > 0:
        return weight / (height ** 2)
    return None

def get_bmi_category(bmi):
    """Get BMI category description"""
    if bmi is None:
        return "Unknown"
    elif bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def get_sentiment_description(compound_score):
    """Get sentiment description from compound score"""
    if compound_score < -0.3:
        return "Negative"
    elif compound_score > 0.3:
        return "Positive"
    else:
        return "Neutral"

def create_text_summary(results, patient_data):
    """Create a text summary for copying"""
    risk_level = results['risk_assessment']['risk_level']
    risk_score = results['risk_assessment']['risk_scores']['overall_risk']
    
    ensemble = results['ml_predictions']['ensemble']
    top_prediction = "Unknown"
    if ensemble and 'error' not in ensemble:
        top_pred_item = max(ensemble.items(), key=lambda x: x[1])
        top_prediction = f"{top_pred_item[0]} ({top_pred_item[1]:.1%})"
    
    summary = f"""
PATIENT ANALYSIS SUMMARY

Patient ID: {patient_data.get('patient_id', 'N/A')}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RISK ASSESSMENT:
- Risk Level: {risk_level}
- Risk Score: {risk_score:.3f}

AI PREDICTION:
- Primary Diagnosis: {top_prediction}

CLINICAL INDICATORS:
- Age of Onset: {patient_data.get('OnsetAge', 'N/A')} years
- Duration: {patient_data.get('Duration', 'N/A')} years
- Blood Pressure: {patient_data.get('LowerBP', 'N/A')}/{patient_data.get('UpperBP', 'N/A')} mmHg

RECOMMENDATIONS:
{'Immediate intervention required' if risk_level == 'High' else 'Enhanced monitoring recommended' if risk_level == 'Medium' else 'Continue routine care'}

This analysis is generated by AI and should be used as clinical decision support only.
    """
    
    return summary.strip()

from datetime import datetime

def create_detailed_text_report(results, patient_data):
    """Create detailed text report for export"""

    # Safely format BMI
    bmi_val = get_bmi(patient_data)
    bmi_str = f"{bmi_val:.1f}" if bmi_val is not None else "N/A"

    report = f"""
PSYCHCAREAI - COMPREHENSIVE ANALYSIS REPORT
{'='*50}

PATIENT INFORMATION:
Patient ID: {patient_data.get('patient_id', 'N/A')}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Age of Onset: {patient_data.get('OnsetAge', 'N/A')} years
Duration of Illness: {patient_data.get('Duration', 'N/A')} years
Current Episode: {patient_data.get('Episode', 'N/A')}

CLINICAL MEASUREMENTS:
Blood Pressure: {patient_data.get('LowerBP', 'N/A')}/{patient_data.get('UpperBP', 'N/A')} mmHg
Temperature: {patient_data.get('Temp', 'N/A')} °F
Respiratory Rate: {patient_data.get('RespiratoryRate', 'N/A')} bpm
Weight: {patient_data.get('Weight', 'N/A')} kg
Height: {patient_data.get('Height', 'N/A')} m
BMI: {bmi_str}

RISK ASSESSMENT:
{'='*20}
Overall Risk Level: {results['risk_assessment']['risk_level']}
Risk Score: {results['risk_assessment']['risk_scores']['overall_risk']:.3f}

Risk Breakdown:
- Clinical Risk: {results['risk_assessment']['risk_scores'].get('clinical', 0):.3f}
- Psychological Risk: {results['risk_assessment']['risk_scores'].get('psychological', 0):.3f}
- Behavioral Risk: {results['risk_assessment']['risk_scores'].get('behavioral', 0):.3f}
- Demographic Risk: {results['risk_assessment']['risk_scores'].get('demographic', 0):.3f}

MACHINE LEARNING PREDICTIONS:
{'='*30}
"""
    # Add ML predictions
    individual_models = results['ml_predictions']['individual_models']
    if individual_models and 'error' not in individual_models:
        for model_name, predictions in individual_models.items():
            if predictions:
                top_pred = max(predictions.items(), key=lambda x: x[1])
                report += f"{model_name}: {top_pred[0]} ({top_pred[1]:.1%})\n"

    # Add ensemble prediction
    ensemble = results['ml_predictions']['ensemble']
    if ensemble and 'error' not in ensemble:
        top_ensemble = max(ensemble.items(), key=lambda x: x[1])
        report += f"\nEnsemble Prediction: {top_ensemble[0]} ({top_ensemble[1]:.1%})\n"

    # NLP analysis
    sentiment_score = results['nlp_analysis'].get('sentiment', {}).get('compound', 0)
    report += f"""
NLP ANALYSIS:
{'='*15}
Symptom Severity: {results['nlp_analysis'].get('severity', 'Unknown')}
Urgency Level: {results['nlp_analysis'].get('urgency', 'Low')}
Sentiment Score: {sentiment_score:.3f}

SYMPTOMS IDENTIFIED:
"""
    symptoms = results['nlp_analysis'].get('symptoms', [])
    for symptom in symptoms:
        category = symptom['category'].replace('_', ' ').title()
        report += f"- {category}: {symptom['count']} keywords\n"

    # Recommendations
    interventions = results['risk_assessment']['interventions']
    report += f"""
RECOMMENDATIONS:
{'='*15}
Immediate Actions:
"""
    for action in interventions.get('immediate_actions', []):
        report += f"- {action}\n"

    report += "\nTreatment Options:\n"
    for treatment in interventions.get('treatment_options', []):
        report += f"- {treatment}\n"

    report += "\nMonitoring Plan:\n"
    for monitor in interventions.get('monitoring', []):
        report += f"- {monitor}\n"

    report += f"""

DISCLAIMER:
{'='*10}
This analysis is generated by AI and is intended for clinical decision support only.
It should not replace professional medical judgment or direct patient care.
If the patient is in immediate danger, contact emergency services immediately.

Report generated by PsychCareAI v1.0
"""
    return report
