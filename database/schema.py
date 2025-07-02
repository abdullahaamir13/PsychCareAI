import sqlite3
import os
import json
from datetime import datetime
import pandas as pd
from typing import Dict, List, Optional, Any

class DatabaseManager:
    """Database manager for PsychCareAI application"""
    
    def __init__(self, db_path: str = "psychcare.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create patients table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS patients (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        patient_id TEXT UNIQUE NOT NULL,
                        onset_age INTEGER,
                        duration REAL,
                        episode INTEGER,
                        lower_bp INTEGER,
                        upper_bp INTEGER,
                        respiratory_rate INTEGER,
                        temperature REAL,
                        weight REAL,
                        height REAL,
                        dose_frequency INTEGER,
                        is_substance_reported BOOLEAN,
                        is_alcohol BOOLEAN,
                        is_marijuana BOOLEAN,
                        is_opium BOOLEAN,
                        is_heroin BOOLEAN,
                        is_hygiene_appropriate BOOLEAN,
                        is_posture_appropriate BOOLEAN,
                        is_behaviour_cooperative BOOLEAN,
                        informant INTEGER,
                        relation INTEGER,
                        present_complaint TEXT,
                        illness_history TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create analysis_results table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS analysis_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        patient_id TEXT NOT NULL,
                        analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        risk_level TEXT,
                        risk_score REAL,
                        ml_predictions TEXT,  -- JSON
                        nlp_analysis TEXT,    -- JSON
                        risk_assessment TEXT, -- JSON
                        feature_analysis TEXT, -- JSON
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
                    )
                ''')
                
                # Create model_metrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS model_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT NOT NULL,
                        accuracy REAL,
                        precision_score REAL,
                        recall REAL,
                        f1_score REAL,
                        roc_auc REAL,
                        cv_mean REAL,
                        cv_std REAL,
                        confusion_matrix TEXT, -- JSON
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create user_sessions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT UNIQUE NOT NULL,
                        current_patient_id TEXT,
                        analysis_results_id INTEGER,
                        session_data TEXT, -- JSON
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (analysis_results_id) REFERENCES analysis_results (id)
                    )
                ''')
                
                # Create system_logs table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        log_level TEXT,
                        component TEXT,
                        message TEXT,
                        details TEXT, -- JSON
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                print("Database initialized successfully")
                
        except Exception as e:
            print(f"Error initializing database: {str(e)}")
    
    def save_patient(self, patient_data: Dict[str, Any]) -> bool:
        """Save patient data to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if patient exists
                cursor.execute('SELECT id FROM patients WHERE patient_id = ?', 
                             (patient_data['patient_id'],))
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing patient
                    cursor.execute('''
                        UPDATE patients SET
                            onset_age = ?, duration = ?, episode = ?,
                            lower_bp = ?, upper_bp = ?, respiratory_rate = ?, temperature = ?,
                            weight = ?, height = ?, dose_frequency = ?,
                            is_substance_reported = ?, is_alcohol = ?, is_marijuana = ?,
                            is_opium = ?, is_heroin = ?, is_hygiene_appropriate = ?,
                            is_posture_appropriate = ?, is_behaviour_cooperative = ?,
                            informant = ?, relation = ?, present_complaint = ?,
                            illness_history = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE patient_id = ?
                    ''', (
                        patient_data.get('OnsetAge'),
                        patient_data.get('Duration'),
                        patient_data.get('Episode'),
                        patient_data.get('LowerBP'),
                        patient_data.get('UpperBP'),
                        patient_data.get('RespiratoryRate'),
                        patient_data.get('Temp'),
                        patient_data.get('Weight'),
                        patient_data.get('Height'),
                        patient_data.get('DoseFrequency'),
                        patient_data.get('IsSubstanceReported'),
                        patient_data.get('IsAlcohol'),
                        patient_data.get('IsMarijuana'),
                        patient_data.get('IsOpium'),
                        patient_data.get('IsHeroin'),
                        patient_data.get('IsHygieneAppropriate'),
                        patient_data.get('IsPostureAppropriate'),
                        patient_data.get('IsBehaviourCooperative'),
                        patient_data.get('Informant'),
                        patient_data.get('Relation'),
                        patient_data.get('PresentComplaint'),
                        patient_data.get('IllnessHistory'),
                        patient_data['patient_id']
                    ))
                else:
                    # Insert new patient
                    cursor.execute('''
                        INSERT INTO patients (
                            patient_id, onset_age, duration, episode,
                            lower_bp, upper_bp, respiratory_rate, temperature,
                            weight, height, dose_frequency,
                            is_substance_reported, is_alcohol, is_marijuana,
                            is_opium, is_heroin, is_hygiene_appropriate,
                            is_posture_appropriate, is_behaviour_cooperative,
                            informant, relation, present_complaint, illness_history
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        patient_data['patient_id'],
                        patient_data.get('OnsetAge'),
                        patient_data.get('Duration'),
                        patient_data.get('Episode'),
                        patient_data.get('LowerBP'),
                        patient_data.get('UpperBP'),
                        patient_data.get('RespiratoryRate'),
                        patient_data.get('Temp'),
                        patient_data.get('Weight'),
                        patient_data.get('Height'),
                        patient_data.get('DoseFrequency'),
                        patient_data.get('IsSubstanceReported'),
                        patient_data.get('IsAlcohol'),
                        patient_data.get('IsMarijuana'),
                        patient_data.get('IsOpium'),
                        patient_data.get('IsHeroin'),
                        patient_data.get('IsHygieneAppropriate'),
                        patient_data.get('IsPostureAppropriate'),
                        patient_data.get('IsBehaviourCooperative'),
                        patient_data.get('Informant'),
                        patient_data.get('Relation'),
                        patient_data.get('PresentComplaint'),
                        patient_data.get('IllnessHistory')
                    ))
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"Error saving patient: {str(e)}")
            return False
    
    def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve patient data by ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('SELECT * FROM patients WHERE patient_id = ?', (patient_id,))
                row = cursor.fetchone()
                
                if row:
                    return dict(row)
                return None
                
        except Exception as e:
            print(f"Error retrieving patient: {str(e)}")
            return None
    
    def save_analysis_results(self, patient_id: str, results: Dict[str, Any]) -> bool:
        """Save analysis results to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Extract key information
                risk_level = results.get('risk_assessment', {}).get('risk_level', 'Unknown')
                risk_score = results.get('risk_assessment', {}).get('risk_scores', {}).get('overall_risk', 0)
                
                cursor.execute('''
                    INSERT INTO analysis_results (
                        patient_id, risk_level, risk_score,
                        ml_predictions, nlp_analysis, risk_assessment, feature_analysis
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    patient_id,
                    risk_level,
                    risk_score,
                    json.dumps(results.get('ml_predictions', {})),
                    json.dumps(results.get('nlp_analysis', {})),
                    json.dumps(results.get('risk_assessment', {})),
                    json.dumps(results.get('feature_analysis', {}))
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"Error saving analysis results: {str(e)}")
            return False
    
    def get_analysis_results(self, patient_id: str) -> List[Dict[str, Any]]:
        """Get analysis results for a patient"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM analysis_results 
                    WHERE patient_id = ? 
                    ORDER BY analysis_timestamp DESC
                ''', (patient_id,))
                
                rows = cursor.fetchall()
                results = []
                
                for row in rows:
                    result = dict(row)
                    # Parse JSON fields
                    for field in ['ml_predictions', 'nlp_analysis', 'risk_assessment', 'feature_analysis']:
                        if result[field]:
                            result[field] = json.loads(result[field])
                    results.append(result)
                
                return results
                
        except Exception as e:
            print(f"Error retrieving analysis results: {str(e)}")
            return []
    
    def save_model_metrics(self, model_name: str, metrics: Dict[str, Any]) -> bool:
        """Save model performance metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO model_metrics (
                        model_name, accuracy, precision_score, recall, f1_score,
                        roc_auc, cv_mean, cv_std, confusion_matrix
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    model_name,
                    metrics.get('accuracy'),
                    metrics.get('precision'),
                    metrics.get('recall'),
                    metrics.get('f1_score'),
                    metrics.get('roc_auc'),
                    metrics.get('cv_mean'),
                    metrics.get('cv_std'),
                    json.dumps(metrics.get('confusion_matrix', []))
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"Error saving model metrics: {str(e)}")
            return False
    
    def get_model_metrics(self) -> List[Dict[str, Any]]:
        """Get latest model performance metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM model_metrics 
                    ORDER BY created_at DESC 
                    LIMIT 10
                ''')
                
                rows = cursor.fetchall()
                metrics = []
                
                for row in rows:
                    metric = dict(row)
                    if metric['confusion_matrix']:
                        metric['confusion_matrix'] = json.loads(metric['confusion_matrix'])
                    metrics.append(metric)
                
                return metrics
                
        except Exception as e:
            print(f"Error retrieving model metrics: {str(e)}")
            return []
    
    def log_system_event(self, level: str, component: str, message: str, details: Dict[str, Any] = None) -> bool:
        """Log system events"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO system_logs (log_level, component, message, details)
                    VALUES (?, ?, ?, ?)
                ''', (
                    level,
                    component,
                    message,
                    json.dumps(details) if details else None
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"Error logging system event: {str(e)}")
            return False
    
    def get_recent_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent system logs"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM system_logs 
                    ORDER BY created_at DESC 
                    LIMIT ?
                ''', (limit,))
                
                rows = cursor.fetchall()
                logs = []
                
                for row in rows:
                    log = dict(row)
                    if log['details']:
                        log['details'] = json.loads(log['details'])
                    logs.append(log)
                
                return logs
                
        except Exception as e:
            print(f"Error retrieving logs: {str(e)}")
            return []
    
    def get_patient_list(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get list of patients"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT patient_id, onset_age, duration, created_at, updated_at
                    FROM patients 
                    ORDER BY updated_at DESC 
                    LIMIT ?
                ''', (limit,))
                
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
                
        except Exception as e:
            print(f"Error retrieving patient list: {str(e)}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Patient count
                cursor.execute('SELECT COUNT(*) FROM patients')
                stats['total_patients'] = cursor.fetchone()[0]
                
                # Analysis count
                cursor.execute('SELECT COUNT(*) FROM analysis_results')
                stats['total_analyses'] = cursor.fetchone()[0]
                
                # Risk level distribution
                cursor.execute('''
                    SELECT risk_level, COUNT(*) as count 
                    FROM analysis_results 
                    GROUP BY risk_level
                ''')
                stats['risk_distribution'] = dict(cursor.fetchall())
                
                # Recent activity (last 7 days)
                cursor.execute('''
                    SELECT COUNT(*) FROM analysis_results 
                    WHERE analysis_timestamp >= datetime('now', '-7 days')
                ''')
                stats['recent_analyses'] = cursor.fetchone()[0]
                
                return stats
                
        except Exception as e:
            print(f"Error retrieving statistics: {str(e)}")
            return {}
    
    def cleanup_old_data(self, days_old: int = 90) -> bool:
        """Clean up old data from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete old logs
                cursor.execute('''
                    DELETE FROM system_logs 
                    WHERE created_at < datetime('now', '-{} days')
                '''.format(days_old))
                
                # Delete old analysis results (keep latest for each patient)
                cursor.execute('''
                    DELETE FROM analysis_results 
                    WHERE id NOT IN (
                        SELECT MAX(id) FROM analysis_results 
                        GROUP BY patient_id
                    ) AND analysis_timestamp < datetime('now', '-{} days')
                '''.format(days_old))
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"Error cleaning up old data: {str(e)}")
            return False
    
    def backup_database(self, backup_path: str) -> bool:
        """Create database backup"""
        try:
            import shutil
            shutil.copy2(self.db_path, backup_path)
            return True
            
        except Exception as e:
            print(f"Error creating backup: {str(e)}")
            return False
    
    def export_patient_data(self, patient_id: str = None) -> pd.DataFrame:
        """Export patient data as DataFrame"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if patient_id:
                    query = 'SELECT * FROM patients WHERE patient_id = ?'
                    df = pd.read_sql_query(query, conn, params=(patient_id,))
                else:
                    query = 'SELECT * FROM patients'
                    df = pd.read_sql_query(query, conn)
                
                return df
                
        except Exception as e:
            print(f"Error exporting patient data: {str(e)}")
            return pd.DataFrame()
    
    def close(self):
        """Close database connection (cleanup)"""
        # SQLite connections are closed automatically with context managers
        pass

