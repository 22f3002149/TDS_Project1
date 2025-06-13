# Create a python script that removes all posts from all json files in ext directory where the post 'created_at' and 'updated_at' is after 2025 April 14
import json
import os
from datetime import datetime
from glob import glob
from datetime import timezone

def remove_posts_after_date(directory, cutoff_date):
    # Get all JSON files in the specified directory
    json_files = glob(os.path.join(directory, "*.json"))
    
    for file_path in json_files:
        with open(file_path, "r") as file:
            data = json.load(file)
        # Check if 'post_stream' and 'posts' key exist
        if 'post_stream' in data and 'posts' in data['post_stream']:
            original_count = len(data['post_stream']['posts'])
            # Filter out posts where 'created_at' or 'updated_at' is after the cutoff date
            data['post_stream']['posts'] = [
                post for post in data['post_stream']['posts']
                if datetime.fromisoformat(post['created_at'].replace("Z", "+00:00")) < cutoff_date and
                   datetime.fromisoformat(post['updated_at'].replace("Z", "+00:00")) < cutoff_date
            ]
            filtered_count = len(data['post_stream']['posts'])
            # Save the modified data back to the file
            with open(file_path, "w") as file:
                json.dump(data, file, indent=4)
            print(f"Processed {file_path}: {original_count} -> {filtered_count} posts remaining.")


directory = "ext"
cutoff_date = datetime(2025, 4, 15, tzinfo=timezone.utc)
remove_posts_after_date(directory, cutoff_date)