## Scraping with Google Sheets

[Scraping with Google Sheets: Here\'s a detailed description of the image:\n\n**Overall Impression:**\n\nThe image is a screenshot of a Google Docs help page detailing the `IMPORTHtml` function. The help page provides syntax and usage examples.\n\n**Key Elements:**\n\n1. **Title:** The document is titled "IMPORTHtml."\n2. **Sample Usage:**\n * The first sample is: `IMPORTHtml("http://en.wikipedia.org/wiki/Demographics_of_India", "table", 4)`\n * The second sample is: `IMPORTHtml(A2, B2, C2)`\n3. **Syntax:**\n * `IMPORTHtml(url, query, index)` is displayed.\n4. **Parameters Explained:**\n * **url:** The URL of the page to examine, including the protocol (e.g., `http://`).\n * **url Value:** The value for URL must be enclosed in quotation marks or be a reference to a cell containing the appropriate text.\n * **query:** Either "list" or "table" depending on what type of structure contains the desired data.\n * **index:** The index, starting at 1, which identifies which table or list as defined in the source should be returned.\n5. **Browser Interface:** The screenshot shows a Chrome browser with multiple tabs visible, including "IMPORTHtml," "Demographics of India - Wikipedia," and "List of highest-growing urban..."\n6. **Video Conference Panel:** A small video conference panel featuring a person’s face is visible in the lower left corner of the image.\n\n**In summary:** This image is a help document on using the `IMPORTHtml` function in Google Docs, with explanations and examples of how to import data from web pages.](https://youtu.be_eYQEk7XJM7s)

You'll learn how to [import tables on the web using Google Sheets's `=IMPORTHTML()` formula](https://support.google.com/docs/answer/3093339?hl=en), covering:

- **Import HTML Formula**: Use =IMPORTHTML(URL, "query", index) to fetch tables or lists from a web page.
- **Granting Access**: Allow access for formulas to fetch data from external sources.
- **Checking Imported Data**: Verify if the imported table matches the data on the web page.
- **Handling Errors**: Understand common issues and how to resolve them.
- **Sorting Data**: Copy imported data as values and sort it within Google Sheets.
- **Freezing Rows**: Use frozen rows to maintain headers while sorting.
- **Live Formulas**: Learn how web data updates automatically when the source changes.
- **Other Import Functions**: IMPORTXML, IMPORTFEED, IMPORTRANGE, and IMPORTDATA for advanced data fetching options.

Here are links used in the video:

- [Google sheet used in the video](https://docs.google.com/spreadsheets/d/1Qp_YTh1-hJHxjMWE_GofkvLIKgEdKxb6NFImpId3z9o/view)
- [`IMPORTHTML()`](https://support.google.com/docs/answer/3093339)
- [`IMPORTXML()`](https://support.google.com/docs/answer/3093342)
- [Demographics of India](https://en.wikipedia.org/wiki/Demographics_of_India)
- [List of highest grossing Indian films](https://en.wikipedia.org/wiki/List_of_highest-grossing_Indian_films)
