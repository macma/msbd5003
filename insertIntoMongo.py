from pymongo import MongoClient
import pandas as pd
import json
#client = MongoClient()
#db = client.test
#db.test.drop()
#db.test.insert({'a':1})

def import_content(filepath):
    mng_client = MongoClient('localhost', 27017)
    mng_db = mng_client['testdb'] #// Replace mongo db name
    collection_name = 'testcol'# // Replace mongo db collection name
    db_cm = mng_db[collection_name]
    #cdir = os.path.dirname(__file__)
    #file_res = os.path.join(cdir, filepath)

    data = pd.read_csv(filepath)
    data_json = json.loads(data.to_json(orient='records'))
    db_cm.remove()
    db_cm.insert(data_json)

import_content('creditcard.csv')
