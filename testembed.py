import sqlite3

# 1. Connect to your SQLite database file
conn = sqlite3.connect("chroma_mcptt/chroma.sqlite3")  # adjust path as needed

# 2. Create a cursor object to execute SQL
cursor = conn.cursor()

# 3. Run SQL commands
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables:", tables)

# 4. Example: count rows in embeddings table
cursor.execute("SELECT COUNT(*) FROM embeddings;")
count = cursor.fetchone()[0]
print("Embeddings count:", count)

# 5. Close connection when done
conn.close()
