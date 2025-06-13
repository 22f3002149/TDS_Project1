import json
import pandas as pd
from glob import glob
from datetime import datetime

def extract_topics_to_df():
    # List of files to process
    files = ["01.json", "02.json", "03.json", "04.json"]
    data = []
    for filename in files:
        with open(filename, "r") as f:
            content = json.load(f)
            # Assuming topics are in content['topic_list']['topics']
            topics = content.get('topics', [])
            for topic in topics:
                data.append({
                    'id': topic.get('id'),
                    'slug': topic.get('slug'),
                    'created_at': topic.get('created_at'),
                    'last_posted_at': topic.get('last_posted_at')
                })
    df = pd.DataFrame(data)
    df.to_csv("topics_extracted.csv", index=False)
    return df

#extract_topics_to_df()
df = pd.read_csv("topics_extracted.csv")

# Remove all records where created_at are after 2025 april 14
df['created_at'] = pd.to_datetime(df['created_at'], utc=True)
cutoff_date = pd.Timestamp('2025-04-14', tz='UTC')
df = df[(df['created_at'] < cutoff_date)]
# Remove all records where created_at and lst_posted_at are before 1 Jan 2025
df['last_posted_at'] = pd.to_datetime(df['last_posted_at'], utc=True)
df = df[(df['last_posted_at'] >= pd.Timestamp('2025-01-01', tz='UTC'))]


# Save the filtered DataFrame to a new CSV file
df.to_csv("filtered_topics.csv", index=False)
# Print the number of records in the filtered DataFrame
print(f"Number of records after filtering: {len(df)}")
# Print the first few rows of the filtered DataFrame
print(df.head())
# Print the number of records in the original DataFrame
print(f"Number of records in the original DataFrame: {len(pd.read_csv('topics_extracted.csv'))}")
# Print the first few rows of the original DataFrame

# remove records with id's 154294 & 160097
df = df[~df['id'].isin([154294, 160097])]
# Save the final DataFrame to a new CSV file
df.to_csv("final_filtered_topics.csv", index=False)
# Print the number of records in the final DataFrame
print(f"Number of records in the final DataFrame: {len(df)}")
# Print the first few rows of the final DataFrame
print(df.head())

