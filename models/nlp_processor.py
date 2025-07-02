import pandas as pd
import numpy as np
import re
from collections import Counter
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

class NLPProcessor:
    def __init__(self):
        self.sentiment_analyzer = None
        self.lemmatizer = None
        self.stop_words = None
        self.psychiatric_keywords = {
            'depression': ['sad', 'depressed', 'hopeless', 'worthless', 'empty', 'tearful', 'crying', 
                          'lonely', 'isolated', 'fatigue', 'tired', 'exhausted', 'sleep', 'insomnia'],
            'anxiety': ['anxious', 'worried', 'nervous', 'panic', 'fear', 'scared', 'restless', 
                       'tension', 'stress', 'overwhelmed', 'racing thoughts', 'palpitation'],
            'psychosis': ['voices', 'hallucinations', 'paranoid', 'suspicious', 'delusions', 
                         'unrealistic thoughts', 'confused', 'disorganized', 'bizarre behavior'],
            'mania': ['elevated', 'high energy', 'hyperactive', 'grandiose', 'racing thoughts', 
                     'decreased sleep', 'impulsive', 'euphoric', 'excessive talking'],
            'substance_abuse': ['alcohol', 'drugs', 'marijuana', 'opium', 'heroin', 'addiction', 
                               'substance', 'withdrawal', 'craving', 'intoxication'],
            'cognitive': ['memory', 'concentration', 'attention', 'confused', 'disoriented', 
                         'forgetful', 'learning difficulty', 'processing'],
            'behavioral': ['aggressive', 'violent', 'irritable', 'impulsive', 'withdrawn', 
                          'isolating', 'antisocial', 'destructive', 'self-harm']
        }
        
    def initialize_models(self):
        """Initialize NLP models and tools"""
        try:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            return True
        except Exception as e:
            print(f"Error initializing NLP models: {str(e)}")
            return False
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_keywords(self, text, category=None):
        """Extract psychiatric keywords from text"""
        if not text:
            return []
        
        text = self.preprocess_text(text)
        tokens = word_tokenize(text)
        
        found_keywords = []
        
        if category and category in self.psychiatric_keywords:
            keywords = self.psychiatric_keywords[category]
        else:
            keywords = []
            for cat_keywords in self.psychiatric_keywords.values():
                keywords.extend(cat_keywords)
        
        for token in tokens:
            if token in keywords:
                found_keywords.append(token)
        
        return list(set(found_keywords))
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text using VADER"""
        if not text or not self.sentiment_analyzer:
            return {'compound': 0, 'pos': 0, 'neu': 0, 'neg': 0}
        
        try:
            scores = self.sentiment_analyzer.polarity_scores(text)
            return scores
        except:
            return {'compound': 0, 'pos': 0, 'neu': 0, 'neg': 0}
    
    def extract_symptoms_from_complaint(self, complaint_text):
        """Extract symptoms and their severity from patient complaint"""
        if not complaint_text:
            return {"symptoms": [], "severity": "mild", "sentiment": {"compound": 0.0}}
        
        # Fallback to basic analysis
        complaint_text = self.preprocess_text(complaint_text)
        sentiment = self.analyze_sentiment(complaint_text)
        
        # Extract keywords by category
        symptom_categories = {}
        for category in self.psychiatric_keywords:
            keywords = self.extract_keywords(complaint_text, category)
            if keywords:
                symptom_categories[category] = {
                    'keywords': keywords,
                    'count': len(keywords),
                    'severity': self._assess_keyword_severity(keywords, complaint_text)
                }
        
        # Convert to expected format
        found_symptoms = []
        severity_score = 0
        for category, details in symptom_categories.items():
            found_symptoms.append({
                'category': category,
                'keywords': details['keywords'],
                'count': details['count']
            })
            severity_score += details['count']
        
        # Determine overall severity
        if severity_score > 5:
            overall_severity = 'severe'
        elif severity_score > 2:
            overall_severity = 'moderate'
        else:
            overall_severity = 'mild'
        
        return {
            "symptoms": found_symptoms,
            "severity": overall_severity,
            "sentiment": sentiment,
            "severity_score": severity_score,
            "urgency": self._assess_urgency(complaint_text, sentiment),
            "symptom_categories": symptom_categories
        }
    
    def _assess_keyword_severity(self, keywords, text):
        """Assess severity based on keywords and context"""
        severity_modifiers = {
            'severe': 3, 'extreme': 3, 'intense': 3, 'overwhelming': 3,
            'moderate': 2, 'significant': 2, 'considerable': 2,
            'mild': 1, 'slight': 1, 'occasional': 1,
            'not': -1, 'no': -1, 'without': -1, 'absence': -1
        }
        
        severity_score = len(keywords)  # Base score
        
        for modifier, weight in severity_modifiers.items():
            if modifier in text:
                severity_score += weight
        
        # Normalize to 1-5 scale
        if severity_score <= 0:
            return 1
        elif severity_score <= 2:
            return 2
        elif severity_score <= 4:
            return 3
        elif severity_score <= 6:
            return 4
        else:
            return 5
    
    def _count_negative_indicators(self, text):
        """Count negative emotional indicators"""
        negative_words = ['not', 'no', 'never', 'nothing', 'nowhere', 'nobody', 'none', 
                         'neither', 'nor', 'cannot', 'can\'t', 'won\'t', 'wouldn\'t', 
                         'shouldn\'t', 'couldn\'t', 'mustn\'t', 'isn\'t', 'aren\'t', 
                         'wasn\'t', 'weren\'t', 'hasn\'t', 'haven\'t', 'hadn\'t', 
                         'doesn\'t', 'don\'t', 'didn\'t']
        
        count = 0
        for word in negative_words:
            count += text.count(word)
        
        return count
    
    def _assess_urgency(self, text, sentiment):
        """Assess urgency level based on text content and sentiment"""
        urgent_keywords = ['emergency', 'crisis', 'suicide', 'kill', 'death', 'die', 
                          'harm', 'hurt', 'dangerous', 'urgent', 'immediate', 'help']
        
        urgency_score = 0
        
        # Check for urgent keywords
        for keyword in urgent_keywords:
            if keyword in text:
                urgency_score += 2
        
        # Factor in sentiment
        if sentiment['compound'] <= -0.5:
            urgency_score += 1
        
        # Categorize urgency
        if urgency_score >= 4:
            return 'High'
        elif urgency_score >= 2:
            return 'Medium'
        else:
            return 'Low'
    
    def extract_features_from_text(self, text):
        """Extract numerical features from text for ML models"""
        if not text:
            return [0] * 19  # Return zero vector if no text
        
        analysis = self.extract_symptoms_from_complaint(text)
        
        features = []
        
        # Sentiment features
        sentiment = analysis.get('sentiment', {})
        features.extend([
            sentiment.get('compound', 0),
            sentiment.get('pos', 0),
            sentiment.get('neu', 0),
            sentiment.get('neg', 0)
        ])
        
        # Symptom category counts and severities
        categories = analysis.get('symptom_categories', {})
        for category in ['depression', 'anxiety', 'psychosis', 'mania', 'substance_abuse']:
            if category in categories:
                features.append(categories[category]['count'])
                features.append(categories[category]['severity'])
            else:
                features.extend([0, 0])
        
        # Text characteristics
        text_length = len(str(text).split()) if text else 0
        negative_indicators = self._count_negative_indicators(str(text).lower())
        urgency_level = analysis.get('urgency', 'Low')
        
        features.extend([
            text_length,
            negative_indicators,
            1 if urgency_level == 'High' else 0,
            1 if urgency_level == 'Medium' else 0,
            len(categories)  # Number of symptom categories present
        ])
        
        return features

    def process_patient_text_data(self, patient_data):
        """Process patient text data for ML model input"""
        features = {}
        
        # Process main complaint
        if 'PresentComplaint' in patient_data:
            complaint_features = self.extract_features_from_text(patient_data['PresentComplaint'])
            feature_names = [
                'presentcomplaint_sentiment_compound', 'presentcomplaint_sentiment_pos', 
                'presentcomplaint_sentiment_neu', 'presentcomplaint_sentiment_neg',
                'presentcomplaint_depression_keywords', 'presentcomplaint_depression_severity',
                'presentcomplaint_anxiety_keywords', 'presentcomplaint_anxiety_severity',
                'presentcomplaint_psychosis_keywords', 'presentcomplaint_psychosis_severity',
                'presentcomplaint_mania_keywords', 'presentcomplaint_mania_severity',
                'presentcomplaint_substance_keywords', 'presentcomplaint_substance_severity',
                'presentcomplaint_text_length', 'presentcomplaint_negative_indicators',
                'presentcomplaint_urgency_high', 'presentcomplaint_urgency_medium',
                'presentcomplaint_symptom_categories_count'
            ]
            
            for name, value in zip(feature_names, complaint_features):
                features[name] = value
        
        # Process illness history
        if 'IllnessHistory' in patient_data:
            history_text = patient_data['IllnessHistory']
            if history_text:
                sentiment = self.analyze_sentiment(str(history_text))
                text_len = len(str(history_text).split())
                features.update({
                    'illnesshistory_sentiment': sentiment.get('compound', 0),
                    'illnesshistory_negative': sentiment.get('neg', 0),
                    'illnesshistory_length': text_len
                })
            else:
                features.update({
                    'illnesshistory_sentiment': 0,
                    'illnesshistory_negative': 0,
                    'illnesshistory_length': 0
                })
        
        return features
