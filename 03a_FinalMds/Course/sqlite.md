## Database: SQLite

Relational databases are used to store data in a structured way. You'll often access databases created by others for analysis.

PostgreSQL, MySQL, MS SQL, Oracle, etc. are popular databases. But the most installed database is [SQLite](https://www.sqlite.org/index.html). It's embedded into many devices and apps (e.g. your phone, browser, etc.). It's lightweight but very scalable and powerful.

Watch these introductory videos to understand SQLite and how it's used in Python (34 min):

[SQLite Introduction - Beginners Guide to SQL and Databases (22 min): Here\'s a detailed description of the image:\n\n**Overall Impression:** The image appears to be a promotional graphic or thumbnail for an introductory tutorial on SQLite. \n\n**Visual Elements:**\n\n* **Text:** The words "SQLite Introduction" are prominently displayed in large, white, bold lettering.\n* **Person:** A man with curly brown hair and a light beard is visible on the left side of the image. He\'s smiling and looking at the camera.\n* **Icons:** Two icons are present below the text:\n * A cylindrical database icon is on the left.\n * An icon of a quill pen is on the right, on top of a blue square.\n* **Background:** The background is a solid, dark teal/blue color.\n\n**Composition:** The image is well-balanced with text taking center stage. The person and icons act as supporting visual elements. It is designed to attract attention and inform viewers about the content of the associated video or tutorial.](https://youtu.be_8Xyn8R9eKB8)

[SQLite Backend for Beginners - Create Quick Databases with Python and SQL (13 min): Here\'s a detailed description of the image:\n\n**Overall Impression:** The image appears to be a thumbnail or promotional graphic related to SQLite databases, featuring a woman and the SQLite logo.\n\n**Visual Elements:**\n\n* **Person:** A woman with long, wavy, light brown/blonde hair is positioned on the right side of the image. She is facing forward with a neutral expression. She is wearing a patterned light grey jacket or blouse.\n* **Logo:** On the left is the SQLite logo. It\'s a blue icon resembling a puzzle piece with a wave design within, positioned above the text "SQLite" in a clean, sans-serif font.\n* **Text:** Beneath the logo is the word "Databases" in white, bold text.\n* **Background:** The background is a dark navy blue with a grid of light blue lines, resembling a network or digital structure.\n\n**Composition:** The image uses a side-by-side composition, with the logo and text on the left and the woman on the right. The visual elements are well-balanced and the overall design is clean and modern.\n\n**Purpose/Context:** The image is likely used to promote or advertise content related to SQLite databases. The woman\'s presence may suggest educational or instructional material.](https://youtu.be/Ohj-CqALrwk)

There are many non-relational databases (NoSQL) like [ElasticSearch](https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html), [MongoDB](https://www.mongodb.com/docs/manual/), [Redis](https://redis.io/docs/latest/), etc. that you should know about and we may cover later.

Core Concepts:

```sql
-- Create a table
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Insert data
INSERT INTO users (name, email) VALUES
    ('Alice', 'alice@example.com'),
    ('Bob', 'bob@example.com');

-- Query data
SELECT name, COUNT(*) as count
FROM users
GROUP BY name
HAVING count > 1;

-- Join tables
SELECT u.name, o.product
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE o.status = 'pending';
```

Python Integration:

```python
import sqlite3
from pathlib import Path
import pandas as pd

async def query_database(db_path: Path, query: str) -> pd.DataFrame:
    """Execute SQL query and return results as DataFrame.

    Args:
        db_path: Path to SQLite database
        query: SQL query to execute

    Returns:
        DataFrame with query results
    """
    try:
        conn = sqlite3.connect(db_path)
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()

# Example usage
db = Path('data.db')
df = await query_database(db, '''
    SELECT date, COUNT(*) as count
    FROM events
    GROUP BY date
''')
```

Common Operations:

1. **Database Management**

   ```sql
   -- Backup database
   .backup 'backup.db'

   -- Import CSV
   .mode csv
   .import data.csv table_name

   -- Export results
   .headers on
   .mode csv
   .output results.csv
   SELECT * FROM table;
   ```

2. **Performance Optimization**

   ```sql
   -- Create index
   CREATE INDEX idx_user_email ON users(email);

   -- Analyze query
   EXPLAIN QUERY PLAN
   SELECT * FROM users WHERE email LIKE '%@example.com';

   -- Show indexes
   SELECT * FROM sqlite_master WHERE type='index';
   ```

3. **Data Analysis**

   ```sql
   -- Time series aggregation
   SELECT
       date(timestamp),
       COUNT(*) as events,
       AVG(duration) as avg_duration
   FROM events
   GROUP BY date(timestamp);

   -- Window functions
   SELECT *,
       AVG(amount) OVER (
           PARTITION BY user_id
           ORDER BY date
           ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
       ) as moving_avg
   FROM transactions;
   ```

Tools to work with SQLite:

- [SQLiteStudio](https://sqlitestudio.pl/): Lightweight GUI
- [DBeaver](https://dbeaver.io/): Full-featured GUI
- [sqlite-utils](https://sqlite-utils.datasette.io/): CLI tool
- [Datasette](https://datasette.io/): Web interface
