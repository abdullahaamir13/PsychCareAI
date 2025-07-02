import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import re

def show_chatbot():
    """Display AI chatbot interface for mental health support"""
    st.header("🤖 AI Mental Health Assistant")
    
    # Check if models are available
    if not st.session_state.ml_models.is_trained():
        st.error("❌ ML models not available. Chatbot functionality is limited.")
        st.info("💡 Load ML models to enable full chatbot capabilities.")
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [
            {
                'role': 'assistant',
                'content': 'Hello! I\'m your AI Mental Health Assistant. I can help you with:\n\n• Mental health symptom analysis\n• Risk assessment insights\n• Treatment recommendations\n• General mental health questions\n\nHow can I assist you today?',
                'timestamp': datetime.now()
            }
        ]
    
    # Chat interface
    display_chat_interface()
    
    # Chat input
    handle_chat_input()
    
    # Quick actions
    show_quick_actions()

def display_chat_interface():
    """Display the chat conversation"""
    st.subheader("💬 Conversation")
    
    # Create a container for chat messages
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                with st.chat_message("user"):
                    st.write(message['content'])
                    st.caption(f"🕐 {message['timestamp'].strftime('%H:%M:%S')}")
            else:
                with st.chat_message("assistant"):
                    st.write(message['content'])
                    st.caption(f"🤖 {message['timestamp'].strftime('%H:%M:%S')}")

def handle_chat_input():
    """Handle user input and generate responses"""
    # Chat input
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now()
        })
        
        # Generate assistant response
        response = generate_assistant_response(user_input)
        
        # Add assistant response to history
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': response,
            'timestamp': datetime.now()
        })
        
        # Rerun to update the display
        st.rerun()

def generate_assistant_response(user_input):
    """Generate intelligent assistant response based on user input"""
    try:
        # Analyze user input
        user_input_lower = user_input.lower()
        
        # Check for specific intents
        response = None
        
        # Greeting detection
        if any(greeting in user_input_lower for greeting in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
            response = generate_greeting_response()
        
        # Symptom analysis request
        elif any(keyword in user_input_lower for keyword in ['symptoms', 'feeling', 'depressed', 'anxious', 'sad', 'worried']):
            response = generate_symptom_analysis_response(user_input)
        
        # Patient analysis request
        elif any(keyword in user_input_lower for keyword in ['analyze', 'patient', 'diagnosis', 'prediction']):
            response = generate_patient_analysis_response()
        
        # Risk assessment request
        elif any(keyword in user_input_lower for keyword in ['risk', 'danger', 'emergency', 'crisis']):
            response = generate_risk_assessment_response(user_input)
        
        # Treatment recommendations
        elif any(keyword in user_input_lower for keyword in ['treatment', 'therapy', 'help', 'intervention']):
            response = generate_treatment_response()
        
        # Model information request
        elif any(keyword in user_input_lower for keyword in ['model', 'accuracy', 'performance', 'how does']):
            response = generate_model_info_response()
        
        # General mental health questions
        elif any(keyword in user_input_lower for keyword in ['mental health', 'depression', 'anxiety', 'bipolar', 'schizophrenia']):
            response = generate_mental_health_info_response(user_input)
        
        # Default response
        if not response:
            response = generate_default_response(user_input)
        
        return response
        
    except Exception as e:
        return f"I apologize, but I encountered an error while processing your request: {str(e)}. Please try rephrasing your question."

def generate_greeting_response():
    """Generate a greeting response"""
    greetings = [
        "Hello! I'm here to help with your mental health questions. What would you like to know?",
        "Hi there! I'm your AI mental health assistant. How can I support you today?",
        "Good to see you! I'm ready to help with mental health analysis and support. What's on your mind?"
    ]
    
    import random
    return random.choice(greetings)

def generate_symptom_analysis_response(user_input):
    """Generate response for symptom analysis"""
    if st.session_state.nlp_processor:
        try:
            # Analyze the user's description
            analysis = st.session_state.nlp_processor.extract_symptoms_from_complaint(user_input)
            
            response = "Based on your description, I've identified the following:\n\n"
            
            # Sentiment analysis
            sentiment = analysis.get('sentiment', {})
            if sentiment.get('compound', 0) < -0.3:
                response += "• **Emotional tone**: Your message suggests you may be experiencing some distress.\n"
            elif sentiment.get('compound', 0) > 0.3:
                response += "• **Emotional tone**: Your message has a more positive tone.\n"
            else:
                response += "• **Emotional tone**: Your message has a neutral emotional tone.\n"
            
            # Symptoms identified
            symptoms = analysis.get('symptoms', [])
            if symptoms:
                response += "• **Symptoms identified**: "
                symptom_categories = [s['category'].replace('_', ' ').title() for s in symptoms]
                response += ", ".join(symptom_categories) + "\n"
            
            # Severity
            severity = analysis.get('severity', 'unknown')
            response += f"• **Severity level**: {severity.title()}\n"
            
            # Urgency
            urgency = analysis.get('urgency', 'Low')
            if urgency == 'High':
                response += "• **⚠️ Urgency**: High - Consider seeking immediate professional help\n"
            elif urgency == 'Medium':
                response += "• **Urgency**: Medium - Consider scheduling an appointment with a mental health professional\n"
            else:
                response += "• **Urgency**: Low - Continue monitoring your symptoms\n"
            
            response += "\n**Please note**: This is an AI analysis and should not replace professional medical advice. If you're experiencing severe symptoms or having thoughts of self-harm, please contact a mental health professional or crisis hotline immediately."
            
            return response
            
        except Exception as e:
            return "I'd like to help analyze your symptoms, but I'm having trouble processing your description right now. Could you try describing your symptoms again, or consider speaking with a mental health professional?"
    
    return "I understand you're describing symptoms. While I can provide general information, I recommend discussing your specific symptoms with a qualified mental health professional who can provide proper assessment and care."

def generate_patient_analysis_response():
    """Generate response for patient analysis requests"""
    if st.session_state.current_patient and st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        response = "Here's a summary of the current patient analysis:\n\n"
        
        # Patient info
        patient_id = results.get('patient_id', 'Unknown')
        response += f"**Patient ID**: {patient_id}\n"
        
        # Risk assessment
        risk_info = results.get('risk_assessment', {})
        risk_level = risk_info.get('risk_level', 'Unknown')
        risk_score = risk_info.get('risk_scores', {}).get('overall_risk', 0)
        
        risk_emoji = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}.get(risk_level, '⚪')
        response += f"**Risk Level**: {risk_emoji} {risk_level} (Score: {risk_score:.3f})\n"
        
        # Top predictions
        ml_predictions = results.get('ml_predictions', {})
        ensemble = ml_predictions.get('ensemble', {})
        
        if ensemble and 'error' not in ensemble:
            top_prediction = max(ensemble.items(), key=lambda x: x[1])
            response += f"**Top Prediction**: {top_prediction[0]} ({top_prediction[1]:.1%} confidence)\n"
        
        # Key symptoms
        nlp_analysis = results.get('nlp_analysis', {})
        symptoms = nlp_analysis.get('symptoms', [])
        if symptoms:
            response += f"**Symptoms Identified**: {len(symptoms)} categories\n"
            for symptom in symptoms[:3]:  # Top 3
                response += f"  • {symptom['category'].replace('_', ' ').title()}\n"
        
        response += "\nWould you like me to explain any specific aspect of this analysis in more detail?"
        
        return response
    
    elif st.session_state.current_patient:
        return "I see there's patient data available, but the analysis hasn't been completed yet. Please run the analysis first in the 'Patient Input' section."
    
    else:
        return "No patient data is currently available for analysis. Please input patient data first in the 'Patient Input' section, then I can help explain the results."

def generate_risk_assessment_response(user_input):
    """Generate response for risk assessment queries"""
    if 'emergency' in user_input.lower() or 'crisis' in user_input.lower() or 'suicide' in user_input.lower():
        return """🚨 **CRISIS RESPONSE**

If you or someone you know is in immediate danger or having thoughts of suicide, please:

• **Call emergency services immediately (911)**
• **Contact the National Suicide Prevention Lifeline: 988**
• **Go to your nearest emergency room**
• **Call the Crisis Text Line: Text HOME to 741741**

**You are not alone, and help is available.**

For ongoing support:
• NAMI Helpline: 1-800-950-NAMI
• SAMHSA Helpline: 1-800-662-4357

This AI assistant is not a replacement for professional crisis intervention. Please reach out to qualified professionals immediately."""
    
    if st.session_state.analysis_results:
        risk_info = st.session_state.analysis_results.get('risk_assessment', {})
        risk_level = risk_info.get('risk_level', 'Unknown')
        risk_scores = risk_info.get('risk_scores', {})
        
        response = f"**Current Risk Assessment**:\n\n"
        response += f"**Overall Risk Level**: {risk_level}\n"
        response += f"**Risk Score**: {risk_scores.get('overall_risk', 0):.3f}/1.0\n\n"
        
        response += "**Risk Breakdown**:\n"
        breakdown = {
            'Clinical': risk_scores.get('clinical', 0),
            'Psychological': risk_scores.get('psychological', 0),
            'Behavioral': risk_scores.get('behavioral', 0),
            'Demographic': risk_scores.get('demographic', 0)
        }
        
        for category, score in breakdown.items():
            response += f"• {category}: {score:.3f}\n"
        
        # Interventions
        interventions = risk_info.get('interventions', {})
        if interventions:
            response += "\n**Recommended Interventions**:\n"
            immediate_actions = interventions.get('immediate_actions', [])
            for action in immediate_actions[:3]:  # Top 3
                response += f"• {action}\n"
        
        return response
    
    return "I can help assess risk factors, but I need patient data and analysis results first. Please complete a patient assessment in the 'Patient Input' section."

def generate_treatment_response():
    """Generate treatment recommendation response"""
    if st.session_state.analysis_results:
        risk_info = st.session_state.analysis_results.get('risk_assessment', {})
        interventions = risk_info.get('interventions', {})
        
        if interventions:
            response = "**Treatment Recommendations**:\n\n"
            
            # Treatment options
            treatment_options = interventions.get('treatment_options', [])
            if treatment_options:
                response += "**Treatment Options**:\n"
                for treatment in treatment_options:
                    response += f"• {treatment}\n"
            
            # Monitoring
            monitoring = interventions.get('monitoring', [])
            if monitoring:
                response += "\n**Monitoring Plan**:\n"
                for monitor in monitoring:
                    response += f"• {monitor}\n"
            
            response += "\n**Important**: These are AI-generated recommendations based on the analysis. Please consult with qualified mental health professionals for personalized treatment planning."
            
            return response
    
    return "I can provide treatment recommendations based on patient analysis. Please complete a patient assessment first, then I can suggest appropriate interventions and monitoring plans."

def generate_model_info_response():
    """Generate response about model information"""
    if st.session_state.ml_models.is_trained():
        available_models = st.session_state.ml_models.get_available_models()
        feature_count = len(st.session_state.ml_models.get_feature_names())
        
        response = "**AI Model Information**:\n\n"
        response += f"**Available Models**: {len(available_models)}\n"
        
        for i, model in enumerate(available_models, 1):
            response += f"{i}. {model}\n"
        
        response += f"\n**Features Used**: {feature_count} different patient characteristics\n"
        response += "\n**Model Capabilities**:\n"
        response += "• Multi-class psychiatric disorder prediction\n"
        response += "• Ensemble predictions from multiple algorithms\n"
        response += "• Natural language processing of symptoms\n"
        response += "• Comprehensive risk assessment\n"
        response += "• Feature importance analysis\n"
        
        # Get model performance if available
        try:
            model_metrics = st.session_state.ml_models.get_model_metrics()
            if model_metrics:
                response += "\n**Model Performance** (when last evaluated):\n"
                for model_name, metrics in list(model_metrics.items())[:2]:  # Top 2 models
                    accuracy = metrics.get('accuracy', 0)
                    f1_score = metrics.get('f1_score', 0)
                    response += f"• {model_name}: {accuracy:.1%} accuracy, {f1_score:.3f} F1-score\n"
        except:
            pass
        
        response += "\n**Note**: Models are trained on clinical data and provide decision support, not final diagnoses."
        
        return response
    
    return "The AI models are currently not loaded. Please ensure the models have been trained and are available for analysis."

def generate_mental_health_info_response(user_input):
    """Generate informational response about mental health topics"""
    user_input_lower = user_input.lower()
    
    if 'depression' in user_input_lower:
        return """**Depression Information**:

Depression is a common but serious mood disorder that affects how you feel, think, and handle daily activities.

**Common Symptoms**:
• Persistent sad, anxious, or empty mood
• Loss of interest in activities once enjoyed
• Fatigue and decreased energy
• Difficulty concentrating or making decisions
• Changes in sleep patterns
• Changes in appetite or weight

**Treatment Options**:
• Psychotherapy (CBT, IPT, etc.)
• Medication (antidepressants)
• Lifestyle changes (exercise, diet, sleep)
• Support groups and peer support

**When to Seek Help**:
• Symptoms persist for more than 2 weeks
• Symptoms interfere with daily functioning
• Thoughts of self-harm or suicide

Remember: Depression is treatable, and seeking help is a sign of strength."""
    
    elif 'anxiety' in user_input_lower:
        return """**Anxiety Disorders Information**:

Anxiety disorders involve excessive fear or anxiety that interferes with daily activities.

**Common Types**:
• Generalized Anxiety Disorder (GAD)
• Panic Disorder
• Social Anxiety Disorder
• Specific Phobias

**Common Symptoms**:
• Excessive worrying
• Restlessness or feeling on edge
• Rapid heartbeat
• Sweating or trembling
• Difficulty concentrating
• Sleep problems

**Treatment Approaches**:
• Cognitive Behavioral Therapy (CBT)
• Exposure therapy
• Anti-anxiety medications
• Relaxation techniques
• Mindfulness and meditation

**Immediate Coping Strategies**:
• Deep breathing exercises
• Progressive muscle relaxation
• Grounding techniques (5-4-3-2-1 method)"""
    
    elif any(word in user_input_lower for word in ['bipolar', 'manic']):
        return """**Bipolar Disorder Information**:

Bipolar disorder involves extreme mood swings including emotional highs (mania/hypomania) and lows (depression).

**Types**:
• Bipolar I: Manic episodes lasting at least 7 days
• Bipolar II: Hypomanic and depressive episodes
• Cyclothymic Disorder: Milder mood swings

**Manic Episode Symptoms**:
• Elevated or irritable mood
• Increased energy and activity
• Decreased need for sleep
• Grandiose thoughts
• Poor judgment

**Treatment**:
• Mood stabilizers
• Antipsychotic medications
• Psychotherapy
• Lifestyle management
• Regular sleep schedule

**Important**: Bipolar disorder requires professional diagnosis and ongoing treatment."""
    
    else:
        return """**General Mental Health Information**:

Mental health includes our emotional, psychological, and social well-being. It affects how we think, feel, and act.

**Signs of Good Mental Health**:
• Ability to cope with life's stresses
• Productive work and activities
• Healthy relationships
• Sense of purpose and fulfillment

**Warning Signs**:
• Persistent sadness or irritability
• Extreme mood changes
• Social withdrawal
• Changes in eating or sleeping
• Difficulty concentrating
• Substance abuse

**Getting Help**:
• Talk to a mental health professional
• Contact your primary care doctor
• Reach out to trusted friends or family
• Use mental health apps or online resources
• Join support groups

Remember: Mental health is just as important as physical health, and seeking help is normal and beneficial."""

def generate_default_response(user_input):
    """Generate default response for unrecognized inputs"""
    responses = [
        "I understand you're asking about mental health. Could you be more specific about what you'd like to know? I can help with symptom analysis, risk assessment, treatment options, or general mental health information.",
        
        "I'm here to help with mental health questions and analysis. You can ask me about:\n• Symptom analysis\n• Risk assessment\n• Treatment recommendations\n• Model information\n• General mental health topics\n\nWhat would you like to explore?",
        
        "I want to make sure I give you the most helpful response. Could you rephrase your question or let me know specifically what aspect of mental health you're interested in?",
        
        "I'm designed to help with mental health analysis and support. If you have a specific question about symptoms, treatments, or need help understanding analysis results, please let me know!"
    ]
    
    import random
    return random.choice(responses)

def show_quick_actions():
    """Show quick action buttons for common requests"""
    st.subheader("⚡ Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔍 Analyze Current Patient"):
            if st.session_state.current_patient:
                response = generate_patient_analysis_response()
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': response,
                    'timestamp': datetime.now()
                })
                st.rerun()
            else:
                st.info("No patient data available. Please input patient data first.")
    
    with col2:
        if st.button("⚠️ Risk Assessment"):
            response = generate_risk_assessment_response("risk assessment")
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response,
                'timestamp': datetime.now()
            })
            st.rerun()
    
    with col3:
        if st.button("💊 Treatment Options"):
            response = generate_treatment_response()
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response,
                'timestamp': datetime.now()
            })
            st.rerun()
    
    with col4:
        if st.button("🤖 Model Info"):
            response = generate_model_info_response()
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response,
                'timestamp': datetime.now()
            })
            st.rerun()
    
    # Emergency button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("🚨 Crisis Resources", type="primary"):
            crisis_response = generate_risk_assessment_response("emergency crisis")
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': crisis_response,
                'timestamp': datetime.now()
            })
            st.rerun()
    
    # Chat management
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = [
                {
                    'role': 'assistant',
                    'content': 'Chat cleared. How can I help you today?',
                    'timestamp': datetime.now()
                }
            ]
            st.rerun()
    
    with col2:
        if st.button("📋 Export Chat"):
            # Create export data
            chat_export = {
                'timestamp': datetime.now().isoformat(),
                'chat_history': st.session_state.chat_history
            }
            
            # Convert to text format
            export_text = f"PsychCareAI Chat Export - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            for message in st.session_state.chat_history:
                role = "👤 User" if message['role'] == 'user' else "🤖 Assistant"
                timestamp = message['timestamp'].strftime('%H:%M:%S')
                export_text += f"{role} [{timestamp}]:\n{message['content']}\n\n"
            
            st.download_button(
                label="📥 Download Chat",
                data=export_text,
                file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
