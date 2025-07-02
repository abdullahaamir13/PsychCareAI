import numpy as np
import pandas as pd
from datetime import datetime

class RiskAssessment:
    def __init__(self):
        self.risk_factors = {
            'demographic': {
                'age_young_adult': {'weight': 0.15, 'threshold': (18, 25)},
                'age_elderly': {'weight': 0.10, 'threshold': (65, 100)},
            },
            'clinical': {
                'substance_abuse': {'weight': 0.25, 'indicators': ['IsSubstanceReported', 'IsAlcohol', 'IsMarijuana', 'IsOpium', 'IsHeroin']},
                'previous_episodes': {'weight': 0.20, 'indicator': 'Episode'},
                'early_onset': {'weight': 0.15, 'threshold': 18},
                'chronic_duration': {'weight': 0.15, 'threshold': 2},
            },
            'behavioral': {
                'poor_hygiene': {'weight': 0.10, 'indicator': 'IsHygieneAppropriate'},
                'uncooperative': {'weight': 0.10, 'indicator': 'IsBehaviourCooperative'},
                'aggressive_behavior': {'weight': 0.15, 'keywords': ['aggressive', 'violent', 'destructive']},
            },
            'psychological': {
                'suicidal_ideation': {'weight': 0.30, 'keywords': ['suicide', 'kill', 'death', 'die', 'harm']},
                'psychotic_symptoms': {'weight': 0.25, 'keywords': ['hallucinations', 'voices', 'paranoid', 'delusions']},
                'severe_depression': {'weight': 0.20, 'keywords': ['hopeless', 'worthless', 'empty']},
                'anxiety_panic': {'weight': 0.15, 'keywords': ['panic', 'fear', 'terror', 'overwhelmed']},
            }
        }
        
        self.intervention_recommendations = {
            'high_risk': {
                'immediate_actions': [
                    'Immediate psychiatric evaluation required',
                    'Consider hospitalization or intensive outpatient program',
                    'Implement safety plan and crisis intervention',
                    'Remove potential means of self-harm',
                    'Ensure 24/7 supervision or support system'
                ],
                'treatment_options': [
                    'Intensive psychotherapy (CBT, DBT)',
                    'Medication evaluation and optimization',
                    'Crisis intervention therapy',
                    'Family therapy and support',
                    'Substance abuse treatment if applicable'
                ],
                'monitoring': [
                    'Daily check-ins with mental health professional',
                    'Weekly psychiatric evaluations',
                    'Continuous risk assessment',
                    'Regular medication monitoring'
                ]
            },
            'medium_risk': {
                'immediate_actions': [
                    'Schedule psychiatric evaluation within 1-2 weeks',
                    'Develop safety plan with patient',
                    'Increase support system engagement',
                    'Consider partial hospitalization program'
                ],
                'treatment_options': [
                    'Regular psychotherapy sessions',
                    'Medication consultation',
                    'Group therapy participation',
                    'Lifestyle interventions (exercise, sleep hygiene)',
                    'Stress management techniques'
                ],
                'monitoring': [
                    'Bi-weekly check-ins',
                    'Monthly psychiatric reviews',
                    'Symptom tracking and mood monitoring',
                    'Treatment compliance assessment'
                ]
            },
            'low_risk': {
                'immediate_actions': [
                    'Schedule routine psychiatric follow-up',
                    'Maintain current treatment plan',
                    'Continue supportive therapy',
                    'Monitor for any changes in symptoms'
                ],
                'treatment_options': [
                    'Maintenance therapy sessions',
                    'Preventive care strategies',
                    'Wellness and recovery programs',
                    'Peer support groups',
                    'Self-management techniques'
                ],
                'monitoring': [
                    'Monthly check-ins',
                    'Quarterly psychiatric reviews',
                    'Annual comprehensive assessment',
                    'Self-monitoring tools and apps'
                ]
            }
        }
    
    def calculate_risk_score(self, patient_data, ml_predictions=None, nlp_analysis=None):
        """Calculate comprehensive risk score for a patient"""
        try:
            risk_scores = {
                'demographic': 0,
                'clinical': 0,
                'behavioral': 0,
                'psychological': 0,
                'overall_risk': 0
            }
            
            # Demographic risk factors
            risk_scores['demographic'] = self._assess_demographic_risk(patient_data)
            
            # Clinical risk factors
            risk_scores['clinical'] = self._assess_clinical_risk(patient_data)
            
            # Behavioral risk factors
            risk_scores['behavioral'] = self._assess_behavioral_risk(patient_data, nlp_analysis)
            
            # Psychological risk factors
            risk_scores['psychological'] = self._assess_psychological_risk(patient_data, nlp_analysis)
            
            # ML prediction risk
            ml_risk = self._assess_ml_prediction_risk(ml_predictions) if ml_predictions else 0
            
            # Calculate overall risk (weighted average)
            overall_risk = (
                risk_scores['demographic'] * 0.15 +
                risk_scores['clinical'] * 0.25 +
                risk_scores['behavioral'] * 0.20 +
                risk_scores['psychological'] * 0.30 +
                ml_risk * 0.10
            )
            
            risk_scores['overall_risk'] = min(overall_risk, 1.0)  # Cap at 1.0
            risk_scores['ml_risk'] = ml_risk
            
            return risk_scores
            
        except Exception as e:
            print(f"Error calculating risk score: {str(e)}")
            return {'overall_risk': 0, 'error': str(e)}
    
    def _assess_demographic_risk(self, patient_data):
        """Assess demographic risk factors"""
        risk_score = 0
        
        # Age-based risk
        if 'OnsetAge' in patient_data and patient_data['OnsetAge'] is not None:
            try:
                age = float(patient_data['OnsetAge'])
                if not np.isnan(age) and age > 0:
                    if 18 <= age <= 25:  # Young adult risk
                        risk_score += 0.3 * self.risk_factors['demographic']['age_young_adult']['weight']
                    elif age >= 65:  # Elderly risk
                        risk_score += 0.2 * self.risk_factors['demographic']['age_elderly']['weight']
            except (ValueError, TypeError):
                pass
        
        return min(risk_score, 1.0)
    
    def _assess_clinical_risk(self, patient_data):
        """Assess clinical risk factors"""
        risk_score = 0
        
        # Substance abuse
        substance_indicators = self.risk_factors['clinical']['substance_abuse']['indicators']
        for indicator in substance_indicators:
            if indicator in patient_data and patient_data[indicator] in (1, True):
                risk_score += 0.15 * self.risk_factors['clinical']['substance_abuse']['weight']
        
        # Episode history
        if 'Episode' in patient_data and patient_data['Episode'] is not None:
            try:
                episodes = float(patient_data['Episode'])
                if not np.isnan(episodes) and episodes > 1:
                    risk_score += min(episodes * 0.1, 0.4) * self.risk_factors['clinical']['previous_episodes']['weight']
            except (ValueError, TypeError):
                pass
        
        # Duration of illness
        if 'Duration' in patient_data and patient_data['Duration'] is not None:
            try:
                duration = float(patient_data['Duration'])
                if not np.isnan(duration) and duration > 2:
                    risk_score += min(duration * 0.05, 0.3) * self.risk_factors['clinical']['chronic_duration']['weight']
            except (ValueError, TypeError):
                pass
        
        # Early onset
        if 'OnsetAge' in patient_data and patient_data['OnsetAge'] is not None:
            try:
                age = float(patient_data['OnsetAge'])
                if not np.isnan(age) and age < 18:
                    risk_score += 0.2 * self.risk_factors['clinical']['early_onset']['weight']
            except (ValueError, TypeError):
                pass
        
        return min(risk_score, 1.0)
    
    def _assess_behavioral_risk(self, patient_data, nlp_analysis):
        """Assess behavioral risk factors"""
        risk_score = 0
        
        # Poor hygiene
        if 'IsHygieneAppropriate' in patient_data and patient_data['IsHygieneAppropriate'] in (0, False):
            risk_score += 0.15 * self.risk_factors['behavioral']['poor_hygiene']['weight']
        
        # Uncooperative behavior
        if 'IsBehaviourCooperative' in patient_data and patient_data['IsBehaviourCooperative'] in (0, False):
            risk_score += 0.2 * self.risk_factors['behavioral']['uncooperative']['weight']
        
        # Aggressive behavior from text analysis
        if nlp_analysis and 'symptom_categories' in nlp_analysis and 'behavioral' in nlp_analysis['symptom_categories']:
            behavioral_severity = nlp_analysis['symptom_categories']['behavioral'].get('severity', 0)
            risk_score += min(behavioral_severity * 0.1, 0.15) * self.risk_factors['behavioral']['aggressive_behavior']['weight']
        
        return min(risk_score, 1.0)
    
    def _assess_psychological_risk(self, patient_data, nlp_analysis):
        """Assess psychological risk factors"""
        risk_score = 0
        
        if nlp_analysis:
            # Suicidal ideation (highest weight)
            if nlp_analysis.get('urgency') == 'High':
                risk_score += 0.4 * self.risk_factors['psychological']['suicidal_ideation']['weight']
            
            # Symptom categories
            if 'symptom_categories' in nlp_analysis:
                categories = nlp_analysis['symptom_categories']
                for category in ['psychosis', 'depression', 'anxiety']:
                    if category in categories:
                        severity = categories[category].get('severity', 0)
                        weight_key = f'psychotic_symptoms' if category == 'psychosis' else f'severe_{category}' if category == 'depression' else 'anxiety_panic'
                        risk_score += min(severity * 0.1, 0.25) * self.risk_factors['psychological'][weight_key]['weight']
            
            # Negative sentiment
            sentiment = nlp_analysis.get('sentiment', {})
            if sentiment.get('compound', 0) < -0.5:
                risk_score += 0.2 * self.risk_factors['psychological']['severe_depression']['weight']
        
        return min(risk_score, 1.0)
    
    def _assess_ml_prediction_risk(self, ml_predictions):
        """Assess risk based on ML predictions"""
        if not ml_predictions or not isinstance(ml_predictions, dict):
            return 0
        
        high_risk_conditions = ['psychosis', 'schizophrenia', 'bipolar', 'depression']
        risk_score = 0
        
        for condition, probability in ml_predictions.items():
            if isinstance(probability, (int, float)) and any(risk_cond in condition.lower() for risk_cond in high_risk_conditions):
                risk_score += min(probability, 1.0) * 0.3 * self.risk_factors['psychological']['psychotic_symptoms']['weight']
        
        return min(risk_score, 1.0)
    
    def get_risk_level(self, risk_score):
        """Convert risk score to categorical risk level"""
        if risk_score >= 0.7:
            return 'High'
        elif risk_score >= 0.4:
            return 'Medium'
        else:
            return 'Low'
    
    def get_interventions(self, risk_level, patient_data=None, nlp_analysis=None):
        """Get personalized intervention recommendations"""
        base_interventions = self.intervention_recommendations.get(
            risk_level.lower() + '_risk', 
            self.intervention_recommendations['low_risk']
        )
        
        # Personalize based on specific conditions
        personalized = self._personalize_interventions(base_interventions, patient_data, nlp_analysis)
        
        return personalized
    
    def _personalize_interventions(self, base_interventions, patient_data=None, nlp_analysis=None):
        """Personalize intervention recommendations based on patient specifics"""
        personalized = base_interventions.copy()
        
        if patient_data:
            # Add substance abuse specific interventions
            substance_abuse = any(
                patient_data.get(indicator, 0) == 1 
                for indicator in ['IsAlcohol', 'IsMarijuana', 'IsOpium', 'IsHeroin']
            )
            
            if substance_abuse:
                if 'Dual diagnosis treatment program' not in personalized['treatment_options']:
                    personalized['treatment_options'].append('Dual diagnosis treatment program')
                if 'Addiction counseling services' not in personalized['treatment_options']:
                    personalized['treatment_options'].append('Addiction counseling services')
        
        if nlp_analysis:
            # Add specific interventions based on symptoms
            symptoms = nlp_analysis.get('symptom_categories', {})
            
            if 'psychosis' in symptoms:
                if 'Antipsychotic medication evaluation' not in personalized['treatment_options']:
                    personalized['treatment_options'].append('Antipsychotic medication evaluation')
            
            if 'anxiety' in symptoms:
                if 'Anxiety management techniques' not in personalized['treatment_options']:
                    personalized['treatment_options'].append('Anxiety management techniques')
            
            if nlp_analysis.get('urgency') == 'High':
                if 'Crisis hotline contact information' not in personalized['immediate_actions']:
                    personalized['immediate_actions'].insert(0, 'Crisis hotline contact information')
        
        return personalized
    
    def generate_risk_report(self, patient_data, risk_scores, risk_level, interventions):
        """Generate comprehensive risk assessment report"""
        report = {
            'assessment_date': datetime.now().isoformat(),
            'risk_level': risk_level,
            'overall_risk_score': risk_scores.get('overall_risk', 0),
            'risk_breakdown': {
                'demographic': risk_scores.get('demographic', 0),
                'clinical': risk_scores.get('clinical', 0),
                'behavioral': risk_scores.get('behavioral', 0),
                'psychological': risk_scores.get('psychological', 0),
                'ml_risk': risk_scores.get('ml_risk', 0)
            },
            'interventions': interventions,
            'recommendations': self._generate_recommendations(risk_level, risk_scores),
            'follow_up_timeline': self._get_follow_up_timeline(risk_level)
        }
        
        return report
    
    def _generate_recommendations(self, risk_level, risk_scores):
        """Generate specific recommendations based on risk assessment"""
        recommendations = []
        
        if risk_level == 'High':
            recommendations.append('Immediate psychiatric intervention required')
            recommendations.append('Consider inpatient hospitalization')
        elif risk_level == 'Medium':
            recommendations.append('Schedule urgent psychiatric consultation')
            recommendations.append('Implement enhanced monitoring')
        else:
            recommendations.append('Continue routine mental health care')
            recommendations.append('Regular follow-up appointments')
        
        # Add specific recommendations based on risk breakdown
        if risk_scores.get('clinical', 0) > 0.5:
            recommendations.append('Review medication compliance and effectiveness')
        
        if risk_scores.get('behavioral', 0) > 0.5:
            recommendations.append('Behavioral intervention strategies needed')
        
        if risk_scores.get('psychological', 0) > 0.7:
            recommendations.append('Intensive psychological support required')
        
        return recommendations
    
    def _get_follow_up_timeline(self, risk_level):
        """Get follow-up timeline based on risk level"""
        timelines = {
            'High': {
                'immediate': '24 hours',
                'short_term': '1 week',
                'medium_term': '2 weeks',
                'long_term': '1 month'
            },
            'Medium': {
                'immediate': '1 week',
                'short_term': '2 weeks',
                'medium_term': '1 month',
                'long_term': '3 months'
            },
            'Low': {
                'immediate': '2 weeks',
                'short_term': '1 month',
                'medium_term': '3 months',
                'long_term': '6 months'
            }
        }
        
        return timelines.get(risk_level, timelines['Low'])
