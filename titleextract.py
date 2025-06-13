# Extract & save as csv file 'id', 'title' from all json files in topics directory
import json
import os
from glob import glob
import pandas as pd
def extract_titles_to_df():
    # List of files to process
    files = glob("topics/*.json")
    data = []
    
    for filename in files:
        with open(filename, "r") as f:
            content = json.load(f)
            # Assuming topics are in content['topic_list']['topics']
            topics = content.get('topics', [])
            for topic in topics:
                data.append({
                    'id': topic.get('id'),
                    'title': topic.get('title')
                })
    
    df = pd.DataFrame(data)
    df.to_csv("titles_extracted.csv", index=False)
    return df

df = extract_titles_to_df()
print(f"Extracted {len(df)} titles to 'titles_extracted.csv'")
print(df.head())
