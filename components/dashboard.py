import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os

def show_dashboard():
    """Display the main dashboard with system overview and statistics"""
    
    st.header("Dashboard")
    
    # System overview
    show_system_overview()
    
    # Model performance metrics
    show_model_performance()
    
    # Recent activity
    show_recent_activity()
    
    # Quick statistics
    show_quick_statistics()

def show_system_overview():
    """Display system overview with key metrics"""
    st.subheader("🏥 System Overview")
    
    # Create metrics columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Models status
        models_count = len(st.session_state.ml_models.get_available_models()) if st.session_state.ml_models.is_trained() else 0
        st.metric(
            label="Available Models",
            value=models_count,
            delta="Ready" if models_count > 0 else "Not Ready"
        )
    
    with col2:
        # Features count
        features_count = len(st.session_state.ml_models.get_feature_names()) if st.session_state.ml_models.is_trained() else 0
        st.metric(
            label="ML Features",
            value=features_count,
            delta="+Advanced NLP" if features_count > 50 else ""
        )
    
    with col3:
        # Current patient status
        patient_status = "Active" if st.session_state.current_patient else "None"
        st.metric(
            label="Current Patient",
            value=patient_status,
            delta="Analysis Ready" if st.session_state.current_patient else ""
        )
    
    with col4:
        # Analysis status
        analysis_status = "Complete" if st.session_state.analysis_results else "Pending"
        st.metric(
            label="Analysis Status",
            value=analysis_status,
            delta="Results Available" if st.session_state.analysis_results else ""
        )

def show_model_performance():
    """Display model performance metrics"""
    st.subheader("🤖 Model Performance")
    
    if not st.session_state.ml_models.is_trained():
        st.warning("⚠️ No models loaded. Please run MH.ipynb to train models.")
        return
    
    try:
        # Get model metrics
        model_metrics = st.session_state.ml_models.get_model_metrics()
        
        if model_metrics:
            # Create performance chart
            models_data = []
            for model_name, metrics in model_metrics.items():
                models_data.append({
                    'Model': model_name,
                    'Accuracy': metrics.get('accuracy', 0),
                    'Precision': metrics.get('precision', 0),
                    'Recall': metrics.get('recall', 0),
                    'F1-Score': metrics.get('f1_score', 0)
                })
            
            if models_data:
                df_metrics = pd.DataFrame(models_data)
                
                # Performance comparison chart
                fig = px.bar(
                    df_metrics.melt(id_vars=['Model'], var_name='Metric', value_name='Score'),
                    x='Model',
                    y='Score',
                    color='Metric',
                    title="Model Performance Comparison",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Performance table
                st.dataframe(
                    df_metrics.set_index('Model').round(4),
                    use_container_width=True
                )
            else:
                st.info("No model performance data available")
        else:
            st.info("Model metrics not available. Please run model evaluation.")
            
            # Show available models
            available_models = st.session_state.ml_models.get_available_models()
            if available_models:
                st.write("**Available Models:**")
                for i, model in enumerate(available_models, 1):
                    st.write(f"{i}. {model}")
                    
    except Exception as e:
        st.error(f"Error loading model performance: {str(e)}")

def show_recent_activity():
    """Display recent system activity"""
    st.subheader("📈 Recent Activity")
    
    # Simulated activity data (in a real system, this would come from logs/database)
    activity_data = []
    
    # Add current patient activity
    if st.session_state.current_patient:
        activity_data.append({
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'Action': 'Patient Data Input',
            'Details': f"Patient ID: {st.session_state.current_patient.get('patient_id', 'Unknown')}",
            'Status': 'Active'
        })
    
    # Add analysis activity
    if st.session_state.analysis_results:
        activity_data.append({
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'Action': 'Analysis Complete',
            'Details': f"Risk Level: {st.session_state.analysis_results.get('risk_assessment', {}).get('risk_level', 'Unknown')}",
            'Status': 'Complete'
        })
    
    # Add model loading activity
    if st.session_state.ml_models.is_trained():
        activity_data.append({
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'Action': 'Models Loaded',
            'Details': f"{len(st.session_state.ml_models.get_available_models())} models available",
            'Status': 'Ready'
        })
    
    # Add default activities if no real activity
    if not activity_data:
        activity_data = [
            {
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'Action': 'System Ready',
                'Details': 'PsychCareAI platform initialized',
                'Status': 'Ready'
            }
        ]
    
    # Display activity table
    if activity_data:
        df_activity = pd.DataFrame(activity_data)
        st.dataframe(df_activity, use_container_width=True)
    else:
        st.info("No recent activity to display")

def show_quick_statistics():
    """Display quick statistics and insights"""
    st.subheader("📊 Quick Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**System Capabilities**")
        
        capabilities = [
            "✅ Multi-model ML predictions",
            "✅ Advanced NLP text analysis",
            "✅ Comprehensive risk assessment",
            "✅ Real-time patient monitoring",
            "✅ Interactive AI chatbot",
            "✅ Detailed analysis reports"
        ]
        
        for capability in capabilities:
            st.markdown(capability)
    
    with col2:
        st.markdown("**Supported Conditions**")
        
        # Get psychiatric keywords from NLP processor
        if st.session_state.nlp_processor and hasattr(st.session_state.nlp_processor, 'psychiatric_keywords'):
            conditions = list(st.session_state.nlp_processor.psychiatric_keywords.keys())
            
            for condition in conditions:
                condition_display = condition.replace('_', ' ').title()
                st.markdown(f"• {condition_display}")
        else:
            default_conditions = [
                "• Depression",
                "• Anxiety Disorders",
                "• Psychotic Disorders",
                "• Bipolar Disorder",
                "• Substance Abuse",
                "• Cognitive Disorders",
                "• Behavioral Issues"
            ]
            
            for condition in default_conditions:
                st.markdown(condition)

def show_feature_importance():
    """Display feature importance analysis"""
    st.subheader("🔍 Feature Importance Analysis")

    try:
        ml_model = st.session_state.ml_models

        # Check if models are loaded
        if not ml_model or not ml_model.is_trained():
            st.warning("No trained models available. Please load or train models first.")
            return

        # Get feature names
        feature_names = ml_model.get_feature_names()
        if not feature_names:
            st.warning("Feature names are missing. Make sure feature_info.pkl is loaded.")
            return

        # Get feature importances from first available model
        available_models = ml_model.get_available_models()
        if not available_models:
            st.warning("No models available for importance analysis.")
            return

        model_name = available_models[0]
        importance_scores = ml_model.get_feature_importance(model_name)

        # Ensure lengths match
        if len(feature_names) != len(importance_scores):
            st.warning(f"Feature names and importances mismatch: {len(feature_names)} vs {len(importance_scores)}")
            min_len = min(len(feature_names), len(importance_scores))
            feature_names = feature_names[:min_len]
            importance_scores = importance_scores[:min_len]

        # Prepare DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_scores
        }).sort_values(by='Importance', ascending=False)

        top_n = min(20, len(importance_df))
        top_features_df = importance_df.head(top_n)

        # Plot
        fig = go.Figure(go.Bar(
            x=top_features_df['Importance'],
            y=top_features_df['Feature'],
            orientation='h',
            marker_color='skyblue'
        ))
        fig.update_layout(
            title="Top Feature Importances",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # Category summary
        feature_categories = {
            'Clinical': [f for f in feature_names if any(term in f.lower() for term in ['bp', 'temp', 'weight', 'height'])],
            'Demographic': [f for f in feature_names if any(term in f.lower() for term in ['age', 'onset', 'duration'])],
            'NLP': [f for f in feature_names if any(term in f.lower() for term in ['sentiment', 'keyword', 'text'])],
            'Behavioral': [f for f in feature_names if any(term in f.lower() for term in ['hygiene', 'behavior', 'substance'])]
        }

        st.markdown("**Feature Categories Summary:**")
        for category, feats in feature_categories.items():
            if feats:
                st.markdown(f"- **{category}**: {len(feats)} features")

    except Exception as e:
        st.error(f"❌ Error displaying feature importance: {str(e)}")

def show_system_health():
    """Display system health indicators"""
    st.subheader("🏥 System Health")
    
    health_indicators = []
    
    # ML Models health
    if st.session_state.ml_models.is_trained():
        health_indicators.append({
            'Component': 'ML Models',
            'Status': '✅ Healthy',
            'Details': f"{len(st.session_state.ml_models.get_available_models())} models loaded"
        })
    else:
        health_indicators.append({
            'Component': 'ML Models',
            'Status': '❌ Unhealthy',
            'Details': 'Models not loaded'
        })
    
    # NLP Processor health
    if st.session_state.nlp_processor:
        health_indicators.append({
            'Component': 'NLP Processor',
            'Status': '✅ Healthy',
            'Details': 'Ready for text analysis'
        })
    else:
        health_indicators.append({
            'Component': 'NLP Processor',
            'Status': '❌ Unhealthy',
            'Details': 'Not initialized'
        })
    
    # Risk Assessment health
    if st.session_state.risk_assessor:
        health_indicators.append({
            'Component': 'Risk Assessment',
            'Status': '✅ Healthy',
            'Details': 'Ready for risk evaluation'
        })
    else:
        health_indicators.append({
            'Component': 'Risk Assessment',
            'Status': '❌ Unhealthy',
            'Details': 'Not available'
        })
    
    # Database health
    if st.session_state.db_manager:
        health_indicators.append({
            'Component': 'Database',
            'Status': '✅ Healthy',
            'Details': 'Connection established'
        })
    else:
        health_indicators.append({
            'Component': 'Database',
            'Status': '⚠️ Warning',
            'Details': 'Limited functionality'
        })
    
    # Display health table
    df_health = pd.DataFrame(health_indicators)
    st.dataframe(df_health, use_container_width=True)
    
    # Overall system status
    healthy_components = sum(1 for indicator in health_indicators if '✅' in indicator['Status'])
    total_components = len(health_indicators)
    health_percentage = (healthy_components / total_components) * 100
    
    st.metric(
        label="Overall System Health",
        value=f"{health_percentage:.0f}%",
        delta=f"{healthy_components}/{total_components} components healthy"
    )

# Add tabs for different dashboard views
def show_dashboard_with_tabs():
    """Show dashboard with multiple tabs"""
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Performance", "Health", "Features"])
    
    with tab1:
        show_system_overview()
        show_recent_activity()
        show_quick_statistics()
    
    with tab2:
        show_model_performance()
    
    with tab3:
        show_system_health()
    
    with tab4:
        show_feature_importance()

# Use the tabbed version as the main function
def show_dashboard():
    """Main dashboard function with tabs"""
    show_dashboard_with_tabs()
