## BBC Weather location ID with Python

[BBC Weather location API with Python: Here\'s a detailed description of the image:\n\n* **Overall Impression:** The image is a slide from a presentation or online course. It has a light gray background with a darker gray horizontal band at the bottom.\n* **Title:** The main title is "Get the data: BBC Weather location service" in large, bold, sans-serif white font.\n* **Course Information:** Below the title, "Course: TOOLS IN DATA SCIENCE" is displayed in a smaller, bold font.\n* **Instructor Information:** Further down, it indicates the instructor as "INSTRUCTOR - ANAND S" and the tutorial instructor as "Tutorial Instructor - MAHESH BALAN U".\n* **Logo:** In the upper right corner, there is a circular logo with the text "IIT Madras ONLINE DEGREE". \n* **Color Scheme:** The primary colors are white, various shades of gray, and the colors within the IIT Madras logo.\n* **Font:** The fonts used are predominantly sans-serif, contributing to a clean and modern look.\n\n\n\n](https://youtu.be_IafLrvnamAw)

You'll learn how to get the location ID of any city from the BBC Weather API -- as a precursor to scraping weather data -- covering:

- **Understanding API Calls**: Learn how backend API calls work when searching for a city on the BBC weather website.
- **Inspecting Web Interactions**: Use the browser's inspect element feature to track API calls and understand the network activity.
- **Extracting Location IDs**: Identify and extract the location ID from the API response using Python.
- **Using Python Libraries**: Import and use requests, json, and urlencode libraries to make API calls and process responses.
- **Constructing API URLs**: Create structured API URLs dynamically with constant prefixes and query parameters using urlencode.
- **Building Functions**: Develop a Python function that accepts a city name, constructs the API call, and returns the location ID.

To open the browser Developer Tools on Chrome, Edge, or Firefox, you can:

- Right-click on the page and select "Inspect" to open the developer tools
- OR: Press `F12`
- OR: Press `Ctrl+Shift+I` on Windows
- OR: Press `Cmd+Opt+I` on Mac

Here are links and references:

- [BBC Location ID scraping - Notebook](https://colab.research.google.com/drive/1-iV-tbtRicKR_HXWeu4Hi5aXJCV3QdQp)
- [BBC Weather - Palo Alto (location ID: 5380748)](https://www.bbc.com/weather/5380748)
- [BBC Locator Service - Los Angeles](https://locator-service.api.bbci.co.uk/locations?api_key=AGbFAKx58hyjQScCXIYrxuEwJh2W2cmv&stack=aws&locale=en&filter=international&place-types=settlement%2Cairport%2Cdistrict&order=importance&s=los%20angeles&a=true&format=json)
- Learn about the [`requests` package](https://docs.python-requests.org/en/latest/user/quickstart/). Watch [Python Requests Tutorial: Request Web Pages, Download Images, POST Data, Read JSON, and More](https://youtu.be/tb8gHvYlCFs)

## BBC Weather data with Python

[Scrape BBC weather with Python: Here\'s a detailed description of the image:\n\n**Overall Impression:** The image is a slide from an online course or presentation, likely from an educational institution. \n\n**Key Elements:**\n\n* **Title:** "Get the data: Scrape BBC weather with Python" is prominently displayed in a large, bold font. This indicates the content focuses on web scraping using Python to extract weather data from the BBC website.\n* **Course Title:** Below the main title, it reads "Course: TOOLS IN DATA SCIENCE." \n* **Instructor & Tutorial Instructor:** The slide also lists the instructor as "ANAND S" and the tutorial instructor as "DIBU PHILIP."\n* **Logo:** In the upper-right corner, there’s a logo indicating the course is affiliated with “IIT Madras Online Degree.”\n* **Color Scheme:** The slide has a light gray background with dark blue at the bottom. The text is primarily dark gray.\n* **Layout:** It is a simple slide with clear information and a clean presentation.\n\n\n\nIn essence, the image describes a lesson on data scraping using Python within a data science course offered by IIT Madras.](https://youtu.be_Uc4DgQJDRoI)

You'll learn how to scrape the live weather data of a city from the BBC Weather API, covering:

- **Introduction to Web Scraping**: Understand the basics of web scraping and its legality.
- **Libraries Overview**: Learn the importance of [`requests`](https://docs.python-requests.org/en/latest/user/quickstart/) and [`BeautifulSoup`](https://beautiful-soup-4.readthedocs.io/).
- **Fetching HTML**: Use [`requests`](https://docs.python-requests.org/en/latest/user/quickstart/) to fetch HTML content from a web page.
- **Parsing HTML**: Utilize [`BeautifulSoup`](https://beautiful-soup-4.readthedocs.io/) to parse and navigate the HTML content.
- **Identifying Data**: Inspect HTML elements to locate specific data (e.g., high and low temperatures).
- **Extracting Data**: Extract relevant data using [`BeautifulSoup`](https://beautiful-soup-4.readthedocs.io/)'s `find_all()` function.
- **Data Cleanup**: Clean extracted data to remove unwanted elements.
- **Post-Processing**: Use regular expressions to split large strings into meaningful parts.
- **Data Structuring**: Combine extracted data into a structured pandas DataFrame.
- **Handling Special Characters**: Replace unwanted characters for better data manipulation.
- **Saving Data**: Save the cleaned data into CSV and Excel formats.

Here are links and references:

- [BBC Weather scraping - Notebook](https://colab.research.google.com/drive/1-gkMzE-TKe3U_yh1v0NPn4TM687H2Hcf)
- [BBC Locator Service - Mumbai](https://locator-service.api.bbci.co.uk/locations?api_key=AGbFAKx58hyjQScCXIYrxuEwJh2W2cmv&stack=aws&locale=en&filter=international&place-types=settlement%2Cairport%2Cdistrict&order=importance&s=mumbai&a=true&format=json)
- [BBC Weather - Mumbai (location ID: 1275339)](https://www.bbc.com/weather/1275339)
- [BBC Weather API - Mumbai (location ID: 1275339)](https://weather-broker-cdn.api.bbci.co.uk/en/forecast/aggregated/1275339)
- Learn about the [`json` package](https://docs.python.org/3/library/json.html). Watch [Python Tutorial: Working with JSON Data using the json Module](https://youtu.be/9N6a-VLBa2I)
- Learn about the [`BeautifulSoup` package](https://beautiful-soup-4.readthedocs.io/). Watch [Python Tutorial: Web Scraping with BeautifulSoup and Requests](https://youtu.be/ng2o98k983k)
- Learn about the [`pandas` package](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html). Watch
  - [Python Pandas Tutorial (Part 1): Getting Started with Data Analysis - Installation and Loading Data](https://youtu.be/ZyhVh-qRZPA)
  - [Python Pandas Tutorial (Part 2): DataFrame and Series Basics - Selecting Rows and Columns](https://youtu.be/zmdjNSmRXF4)
- Learn about the [`re` package](https://docs.python.org/3/library/re.html). Watch [Python Tutorial: re Module - How to Write and Match Regular Expressions (Regex)](https://youtu.be/K8L6KVGG-7o)
- Learn about the [`datetime` package](https://docs.python.org/3/library/datetime.html). Watch [Python Tutorial: Datetime Module - How to work with Dates, Times, Timedeltas, and Timezones](https://youtu.be/eirjjyP2qcQ)
