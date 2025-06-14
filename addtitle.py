# Write a python script that runs through all files in the 'DisPosts' directory and adds a title to each file 'Topics.csv' file based on the number part of file name as 'id' field in the 'Topics.csv' file. The title for each id is given in the column 'title' of the 'Topics.csv' file. The title should be added as the first line of each file in the format: '# Topic: <title>'.
import os
import csv
def add_titles_to_posts(posts_dir, topics_csv):
    # Read the topics from the CSV file
    topics = {}
    with open(topics_csv, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            topics[row['id']] = row['title']
    print(topics)

    # Iterate through all files in the posts directory
    for root, _, files in os.walk(posts_dir):
        for file in files:
            if file.endswith('.txt'):
                file_id = os.path.splitext(file)[0]  # Get the file name without extension
                # split the file_id with '_' and take the second part
                file_id = file_id.split('_')[1] if '_' in file_id else file_id
                print(file_id)
                if file_id in topics:
                    title = topics[file_id]
                    file_path = os.path.join(root, file)

                    # Read the existing content
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Add the title as the first line
                    new_content = f"# Topic: {title}\n\n{content}"

                    # Write the modified content back to the file
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
    print(f"Titles added to posts in '{posts_dir}' based on '{topics_csv}'.")
# Example usage

posts_directory = 'DisPosts'  # Replace with your actual posts directory
topics_csv_file = 'Topics.csv'  # Replace with your actual topics CSV file
add_titles_to_posts(posts_directory, topics_csv_file)

# This script adds titles to text files in the specified posts directory based on the topics defined in a CSV file.
# It reads the 'id' and 'title' from the CSV file and adds the title as the first line of each text file.
# The title is formatted as '# Topic: <title>'.
# The script assumes that the text files are named with their corresponding 'id' from the CSV file.
# It processes all text files in the 'DisPosts' directory and modifies them accordingly.