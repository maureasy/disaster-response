import sys
import pandas as pd
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import re
from sklearn.metrics import classification_report
import numpy as np
from sklearn.model_selection import GridSearchCV
import pickle

def load_data(database_filepath):
    engine = create_engine(str('sqlite:///'+database_filepath))
    conn = engine.connect()
    df = pd.read_sql_table('disaster', conn)
    X = df['message'] 
    y = df.iloc[:,3:]
    return X,y,y.columns


def tokenize(text):
    text_no_punct =re.sub(r"[^A-Za-z0-9 _]"," ",text)
    
    text_split = text_no_punct.split()
    text_clean = [x.strip(' ') for x in text_split]
    return text_clean

def build_model():
    pipeline = Pipeline([
        ("vect", CountVectorizer(tokenizer=tokenize)), 
        ("tfidf", TfidfTransformer()), 
        ("clf", MultiOutputClassifier(RandomForestClassifier()))
         ])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = pd.DataFrame(model.predict(X_test),columns=category_names)
    for i in y_pred:
         print("output category:",i)
         print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
         print(classification_report(Y_test[i], y_pred[i]))


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()