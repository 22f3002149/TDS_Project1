# create a list from taking values from Topics.csv in below format
# "https://discourse.onlinedegree.iitm.ac.in/t/" + slug + "/" + str(id) + "/n" for each row in Topics.csv
import pandas as pd
def generate_links_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    links = []
    for index, row in df.iterrows():
        slug = row['slug']
        topic_id = row['id']
        link = f"https://discourse.onlinedegree.iitm.ac.in/t/{slug}/{topic_id}"
        links.append(link)
    # save links to a file
    with open("links.txt", "w") as file:
        for link in links:
            file.write(link + "\n")
    return links

print(generate_links_from_csv("Topics.csv"))