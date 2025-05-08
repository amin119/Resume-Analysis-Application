from langchain.document_leaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline 
from .config import Config
from .data_preprocessor import Preprocessor
import os

class HRPolicyRetriver : 
    def __init__(self):
        Config.setup_dirs()
        self.preprocessor = Preprocessor()
        self.qa_pipeline = pipeline(
            "quastion-aswering",
            model="deepset/roberta-base-sqaud2"
        )
        self.vectorstore = self._init_vectorstore()
        self.category_policy_map = {
            # Corporate/Office Roles
            'HR': "recruitment_policy",
            'BUSINESS-DEVELOPMENT': "bd_compensation_guidelines",
            'CONSULTANT': "consultant_engagement_rules",
            'FINANCE': "financial_compliance",
            'ACCOUNTANT': "accounting_standards",
            'PUBLIC-RELATIONS': "pr_media_policy",
            'BANKING': "banking_compliance",
    
            # Technical Roles
            'ENGINEERING': "tech_hiring_guidelines",
            'INFORMATION-TECHNOLOGY': "it_security_policies",
            'DIGITAL-MEDIA': "digital_content_standards",
    
            # Healthcare/Services
            'HEALTHCARE': "healthcare_credentialing",
            'FITNESS': "fitness_certification_requirements",
    
            # Creative Roles
            'DESIGNER': "design_submission_guidelines",
            'ARTS': "artistic_rights_policy",
    
            # Education
            'TEACHER': "educator_certification",
    
            # Legal
            'ADVOCATE': "legal_ethics_procedures",
    
            # Industrial/Manual
            'AGRICULTURE': "agricultural_safety",
            'CONSTRUCTION': "construction_safety",
            'AUTOMOBILE': "automotive_standards",
            'AVIATION': "aviation_compliance",
    
            # Retail/Food Service
            'SALES': "sales_commission_policy",
            'CHEF': "food_safety_standards",
            'APPAREL': "apparel_quality_control",
    
            # Customer Support
            'BPO': "call_center_procedures",
    
            # Special Cases
            'GENERAL': "employee_handbook"  # Fallback policy
        }
    def _init_vectorstore(self):
        if os.path.exists(Config.FAISS_INDEX):
            return FAISS.load_local(
                Config.FAISS_INDEX, 
                HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            )
        print("Initializing RAG index...")
        loader = DirectoryLoader(
            Config.KNOWLEDGE_BASE,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        docs = loader.load()
            
        # we will use the same cleaning as preprocessing 
        for doc in docs:
            doc.page_content = self.preprocessor.clean_text(doc.page_content)
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        texts = text_splitter.split_documents(docs)
            
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
            
        vectorstore = FAISS.from_documents(texts, embeddings)
        vectorstore.save_local(Config.FAISS_INDEX)
        return vectorstore
    
    def query(self, question, category=None):
        """Enhanced with category-aware retrieval"""
        if category and category in self.category_policy_map:
            question = f"{self.category_policy_map[category]}: {question}"
            
        docs = self.vectorstore.similarity_search(question, k=2)
        context = " ".join([d.page_content for d in docs])
        
        #clean context using the same preprocessor
        context = self.peprocessor.clean_text(context)
        
        result = self.qa_pipeline(
            question=question,
            context=context,
            max_answer_len=200
        )
        
        return {
            "answer": result["answer"],
            "confidence": round(result["score"], 2),
            "source": list(set([d.metadata["source"] for d in docs])),
            "category_relevance": category if category else "geenral"
        }
        
# singleton instance 
policy_retriever = HRPolicyRetriver()
        
        