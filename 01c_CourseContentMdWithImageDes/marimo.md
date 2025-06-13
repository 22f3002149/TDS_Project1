## Interactive Notebooks: Marimo

[Marimo](https://marimo.app/) is a new take on notebooks that solves some headaches of Jupyter. It runs cells reactively - when you change one cell, all dependent cells update automatically, just like a spreadsheet.

Marimo's cells can't be run out of order. This makes Marimo more reproducible and easier to debug, but requires a mental shift from the Jupyter/Colab way of working.

It also runs Python directly in the browser and is quite interactive. [Browse the gallery of examples](https://marimo.io/gallery). With a wide variety of interactive widgets, It's growing popular as an alternative to Streamlit for building data science web apps.

Common Operations:

```python
# Create new notebook
uvx marimo new

# Run notebook server
uvx marimo edit notebook.py

# Export to HTML
uvx marimo export notebook.py
```

Best Practices:

1. **Cell Dependencies**
   - Keep cells focused and atomic
   - Use clear variable names
   - Document data flow between cells
2. **Interactive Elements**

   ```python
   # Add interactive widgets
   slider = mo.ui.slider(1, 100)
   # Create dynamic Markdown
   mo.md(f"{slider} {"🟢" * slider.value}")
   ```

3. **Version Control**
   - Keep notebooks are Python files
   - Use Git to track changes
   - Publish on [marimo.app](https://marimo.app/) for collaboration

["marimo: an open-source reactive notebook for Python" - Akshay Agrawal (Nbpy2024): Here\'s a detailed description of the image:\n\n**Overall Impression:**\n\nThe image displays a data analysis interface, likely from a machine learning or data science tool. It shows a preview of images (digits) along with the corresponding data table. A video of a speaker is also present in the bottom right corner.\n\n**Detailed Breakdown:**\n\n1. **Image Preview:** At the top, a horizontal row displays grayscale images of the digit "2." There are 9 images visible.\n2. **Data Table:** Below the image preview is a data table with the following columns:\n * **Index:** Numerical identifier for each row.\n * **X:** Numerical values (negative decimals).\n * **Y:** Numerical values (negative decimals).\n * **Digit:** The label for each image (in this case, all are labeled as "2" or "8").\n3. **Data Table Rows:** The table displays several rows of data, each associated with one of the images. The “Digit” column contains both "2" and "8", indicating the model may be trying to distinguish between these two digits.\n4. **Interface Elements:** The left side features interface elements:\n * Icons for various functions (selection, navigation).\n * Dropdown menus for controlling view layout ("Vertical").\n5. **Video Feed:** A video feed is displayed in the bottom right corner. It features a person speaking (likely a presenter), with the logos of companies like Meta, Bloomberg, and Netflix visible in the background.\n6. **Status Indicators**: There are status indicators ("on startup", "on cell change") visible at the bottom.\n\n**In essence, the image captures a data analysis workflow involving image classification or digit recognition, with a video presentation running in parallel.**](https://youtu.be_9R2cQygaoxQ)
