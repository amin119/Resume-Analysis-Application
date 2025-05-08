import joblib
from .config import Config 
from torch.utiles.data import Dataset
import warnings 
warnings.filterwarnings('.ignore')

class ResumeDataset(Dataset):
    def __init__(self, encodings, numerical_features, labels):
        self.encodings = encodings 
        self.numerical_features = numerical_features
        self.labels = labels
        
    def __getitem__(self, idx): 
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['numerical_features'] = torch.tensor(self.numerical_features[idx], dtype=torch.float)
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)
    
class ModelTrainer : 
    def __init__(self):
        Config.setup_dirs()
        self.scaler = StandardScaler()
        self.numerical_cols = [
            'text_length',
            'num_skills',
            'experience_years',
            'features.tech_stack_count',
            'features.certification_count',
            'features.finance_terms',
            'features.achievement_count'               
        ]
    def load_and_prepare_data(self):
        # Load the preprocessed data
        df = joblib.load(Config.PROCESSED_DATA)
        
        #prepare numerical features
        X_numerical = df[self.numerical_cols].fillna(0).values
        X_numerical = self.scaler.fit_transform(X_numerical)
        
        #prepare text data
        texts = df['clean_text'].values
        
        #prepare labels
        y = df['score', 'is_fraud'].values
        
        #split data 
        (X_train_text, X_val_text,
         X_train_num, X_val_num,
         y_train, y_val) = train_test_split(
            texts, X_numerical, y, 
            test_size=0.2, 
            random_state=42
         )
         
        return (X_train_text, X_val_text,
                 X_train_num, X_val_num, 
                 y_train, y_val)
        
    def train(self):
        
        #load data
        (train_texts, val_text,
         train_numerical, val_numerical,
         train_labels, val_labels) = self.load_and_prepare_data()
        
        #intialize tokenizer 
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenizer.save_pretrained(Config.TOKENIZER_DIR)
        
        #tokeize text 
        train_encodings = tokenizer(
            list(train_texts),
            truncation=True,
            padding=True,
            max_length=512,
        )
        val_encodings = tokenizer(
            list(val_texts),
            truncation=True,
            padding=True,
            max_length=512,
        )
        
        #create datasets
        train_dataset = ResumeDataset(
            train_encodings, 
            train_numerical, 
            train_labels
        )
        
        val_dataset = ResumeDataset(
            val_encodings, 
            val_numerical, 
            val_labels
        ) 
        # custom BERT model that accepts numerical features
        class BertWithFeatures(BertForSequenceClassification):
            def __init__(self, config):
                super().__init__(config)
                self.num_features = len(self.numerical_cols)
                self.linear = torch.nn.Linear(self.num_features, config.hidden_size)
            
            def forward(self, input_ids=None, attention_mask=None, numerical_features=None, **kwargs):
                #standard BERT processing 
                outputs = super().forward(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    **kwargs        
                )
                #add numericl features
                if numerical_features is not None:
                    features = self.linear(numerical_features)
                    outputs.logits += features.unsqueeze(1)
                    
                return outputs
            #Initialize model
            model = BertWithFeatures.from_pretrained(
                'bert-base-uncased',
                num_labels=2,  
                problem_type="multi_label_classification",
            )
            #training arguments
            training_args = TrainingArguments(
                output_dir=Config.MODEL_DIR,
                num_train_epochs=5,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=16,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                logging_dir='./logs',
                logging_steps=50,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
            )
            #custom trainer to handle numerical features 
        class CustomTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                #extract numerical features
                num_features = inputs.pop('numerical_features')
                    
                #forward pass
                outputs = model(**inputs, numerical_features=num_features)
                loss = outputs.loss
                return(loss, outputs) if return_outputs else loss
        
        #trainnnnnnnnn
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        trainer.train()
        model.save_pretrained(Config.MODEL_DIR)
        joblib.dump(self.scaler, Config.MODEL_DIR / 'scaler.joblib')

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train()
            
