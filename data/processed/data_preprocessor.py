import pandas as pd 
import re 
from bs4 import BeautifulSoup
from sklearn.preprocessing import LabelEncoder
from .config import Config
import joblib 

class Preprocessor : 
    def __init__(self):
        Config.setup_dirs()
        self.category_mapping = {
            'HR': 0.8,
            'DESIGNER': 0.7,
            'INFORMATION-TECHNOLOGY': 0.85,
            'TEACHER': 0.65,
            'ADVOCATE': 0.75,
            'BUSINESS-DEVELOPMENT': 0.8,
            'HEALTHCARE': 0.9,
            'FITNESS': 0.6,
            'AGRICULTURE': 0.65,
            'BPO': 0.7,
            'SALES': 0.75,
            'CONSULTANT': 0.85,
            'DIGITAL-MEDIA': 0.7,
            'AUTOMOBILE': 0.75,
            'CHEF': 0.6,
            'FINANCE': 0.85,
            'APPAREL': 0.65,
            'ENGINEERING': 0.9,
            'ACCOUNTANT': 0.8,
            'CONSTRUCTION': 0.75,
            'PUBLIC-RELATIONS': 0.7,
            'BANKING': 0.85,
            'ARTS': 0.65,
            'AVIATION': 0.8           
        }
    

    @staticmethod
    def clean_text(text):
        """Clean resume text from your dataset"""
        if pd.isna(text):
            return ""
            
        # Standard cleaning
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[•·■♦]', ' ', text)  # Multiple bullet point types
        
        # Category-specific cleaning
        text = re.sub(r'Company Name', '[COMPANY]', text)
        text = re.sub(r'\.{2,}', ' ', text)  # Remove multiple dots
        return text.strip()
    
    def extract_features(self, df):
        """Comprehensive feature extraction combining geenra and category-specific features""" 
        
        #1. Basic text features (using clean_text)
        df['text_length'] = df['clean_text'].str.len()
        df['num_skills'] = df['clean_text'].str.count(r'Skills?:', flags=re.I)
    
        # 2. Experience Extraction (improved regex)
        df['experience_years'] = (
            df['clean_text']
            .str.extract(r'(\d+)\s*\+?\s*years?[\s\w]*experience', flags=re.I)
            .fillna(0)
            .astype(int)
        )

        # 3. Education Level Detection
        df['education_level'] = (
            df['clean_text']
            .str.extract(r'(PhD|Master|Bachelor|Diploma|High School)', flags=re.I)
            .fillna('Not Specified')
        )

        # 4. Category-Specific Features
        df['features'] = df.apply(
            lambda x: self._extract_category_specific_features(x['clean_text'], x['Category']), 
            axis=1
        )

        # 5. Convert features dict to columns
        features_df = pd.json_normalize(df['features'])
        df = pd.concat([df, features_df], axis=1)
    
        return df

    def _extract_category_specific_features(self, text, category):
        """Extract profession-specific features"""
        features = {}

        # Engineering/IT Roles
        if category in ['ENGINEERING', 'INFORMATION-TECHNOLOGY']:
            features['tech_stack_count'] = len(re.findall(
                r'\b(Python|Java|C\+\+|JavaScript|SQL|React|TensorFlow)\b', 
                text, re.I
            ))
            features['project_mentions'] = text.count('project')
        
        # Healthcare Roles
        elif category == 'HEALTHCARE':
            features['certification_count'] = len(re.findall(
                r'\b(BLS|ACLS|RN|MD|PhD|Certified)\b', 
                text
            ))
            features['medical_terms'] = len(re.findall(
                r'\b(patient|clinical|treatment|diagnos|medicine)\b', 
                text, re.I
            ))
    
        # Finance Roles
        elif category in ['FINANCE', 'ACCOUNTANT', 'BANKING']:
            features['finance_terms'] = len(re.findall(
                r'\b(budget|forecast|ROI|financial|audit|tax|GAAP)\b', 
                text, re.I
            ))
    
    # For all categories
        features['achievement_count'] = len(re.findall(
            r'(achievement|accomplishment|successfully)', 
            text, re.I
        ))
    
        return features   
    
    
    def detect_red_flags(self, text): 
        """Full fraud detection for your data"""
        flags =  {
            'generic_company' :  len(re.findall(r'\[COMPANY\]', text)) > 2,
            'vague_dates' : bool(re.search(r'(Jan|Feb|Mar).*?\d{4}.*?to.*?(Jan|Feb|Mar).*?\d{4}', text)),
            'skill_gaps' : len(re.findall(r'Skills:.+?([A-Z][a-z]+)', text)) < 3
        }
        return sum(flags.values())
    
    def process(self):
        """Full preprocessing pipeline with multi-category support"""
        df = pd.read_csv(Config.RAW_DATA)
        
        # Clean text
        df['clean_text'] = df.apply(lambda x: self.clean_text(x['Resume_str']), axis=1)
        
        # Handle categories
        df['category_encoded'] = LabelEncoder().fit_transform(df['Category'])
        df['base_score'] = df['Category'].map(self.category_mapping).fillna(0.5)
        
        # Extract features
        df['features'] = df.apply(
            lambda x: self.extract_category_features(x['clean_text'], x['Category']), 
            axis=1
        )
        
        # Detect fraud
        df['is_fraud'] = df.apply(
            lambda x: self.detect_red_flags(x['clean_text'], x['Category']) >= 2, 
            axis=1
        ).astype(int)
        
        # Calculate final score
        df['score'] = df.apply(
            lambda x: min(1.0, x['base_score'] + (0.1 if x['features'].get('tech_terms', 0) > 3 else 0)),
            axis=1
        )
        
        # Save processed data
        joblib.dump(df, Config.PROCESSED_DATA)
        return df
      
if __name__ == "__main__":
    preprocessor = Preprocessor()
    processed_data = preprocessor.process()