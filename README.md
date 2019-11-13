# Disaster Response Pipeline Project

This project trains over 26000 tagged disaster messages into 36 disaster response groups. The process happens in
two steps where the process_data.py reads the csv file, cleans, prepares the data to be processed and then saves the result 
into a database DisasterReponse.db. The trainclassifier.py reads this table from the dataase, builds a pipeline, splits the
data into training and test set and trains the data. The test set evaluated on all 36 classes and printed out. The results
are shown on a website and you see how your sentence is being classified by the trained model.

### 36 Classifation Categories:

['related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products', 'search_and_rescue', 'security', 'military', 'child_alone', 'water', 'food', 'shelter', 'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid', 'infrastructure_related', 'transport', 'buildings', 'electricity', 'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure', 'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather', 'direct_report']


### File structure

app -- templates
    -- run.py
    
data -- csv files
     -- database files
     -- process_data.py

models -- train_classifier.py
       -- stored models


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to host='0.0.0.0', port=3001 on your browser


