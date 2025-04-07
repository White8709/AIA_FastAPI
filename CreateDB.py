import sqlite3

conn = sqlite3.connect("test.db")
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS fruit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    price REAL NOT NULL,
    on_offer BOOLEAN DEFAULT 0
)
''')

cursor.executemany('''
INSERT INTO fruit (name, description, price, on_offer)
VALUES (?, ?, ?, ?)
''', [
    ('香蕉', '這是香蕉', 41.9, True),
    ('蘋果', '這是蘋果', 36.0, False),
    ('芭樂', '這是芭樂', 39.7, True)
])

cursor.execute("SELECT * FROM fruit")
for row in cursor.fetchall():
    print(row)

conn.commit()
conn.close()
