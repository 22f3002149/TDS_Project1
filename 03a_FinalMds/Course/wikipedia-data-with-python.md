## Wikipedia Data with Python

[Wikipedia data with Wikimedia Python library: Here\'s a detailed description of the image:\n\n**Overall Impression:** The image is a presentation slide, likely from an online course. It features text-based information on a light gray background with a blue horizontal bar at the bottom.\n\n**Key Elements:**\n\n* **Title:** "Get the data: Wikimedia" is prominently displayed in a large, bold font.\n* **Course Name:** Below the title, "Course: TOOLS IN DATA SCIENCE" is written in a smaller font.\n* **Instructor Information:**\n * "INSTRUCTOR - ANAND S"\n * "Tutorial Instructor - DIBU PHILIP" are displayed below the course name.\n* **Branding:** In the upper right corner, there\'s a logo and the text "IIT Madras ONLINE DEGREE".\n* **Background:** The slide has a light gray background.\n* **Color Scheme:** The primary colors are light gray, blue and dark text.\n\n**Purpose:** The slide is likely an introduction to a lesson or tutorial within a data science course, specifically focused on obtaining data from Wikimedia platforms.](https://youtu.be_b6puvm-QEY0)

You'll learn how to scrape data from Wikipedia using the `wikipedia` Python library, covering:

- **Installing and Importing**: Use pip install to get the Wikipedia library and import it with import wikipedia as wk.
- **Keyword Search**: Use the search function to find Wikipedia pages containing a specific keyword, limiting results with the results argument.
- **Fetching Summaries**: Use the summary function to get a concise summary of a Wikipedia page, limiting sentences with the sentences argument.
- **Retrieving Full Pages**: Use the page function to obtain the full content of a Wikipedia page, including sections and references.
- **Accessing URLs**: Retrieve the URL of a Wikipedia page using the url attribute of the page object.
- **Extracting References**: Use the references attribute to get all reference links from a Wikipedia page.
- **Fetching Images**: Access all images on a Wikipedia page via the images attribute, which returns a list of image URLs.
- **Extracting Tables**: Use the pandas.read_html function to extract tables from the HTML content of a Wikipedia page, being mindful of table indices.

Here are links and references:

- [Wikipedia Library - Notebook](https://colab.research.google.com/drive/1-w8Jo6xcQs2jK0NxNddPW4HVCZhXmTBe)
- Learn about the [`wikipedia` package](https://wikipedia.readthedocs.io/en/latest/)

**NOTE**: Wikipedia is constantly edited. The page may be different now from when the video was recorded. Handle accordingly.
