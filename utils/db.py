import sqlite3
import os

# Database file path
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'records.db')

def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    # Table structure updated to 8 columns to match your app logic
    cursor.execute('''CREATE TABLE IF NOT EXISTS diagnostics 
                      (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                       name TEXT, age INTEGER, gender TEXT, 
                       phone TEXT, pid TEXT, date TEXT, 
                       diag TEXT, conf TEXT)''')
    conn.commit()
    conn.close()

def save_result(name, age, gender, phone, pid, date, diag, conf):
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''INSERT INTO diagnostics (name, age, gender, phone, pid, date, diag, conf) 
                      VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', 
                   (name, age, gender, phone, pid, date, diag, conf))
    conn.commit()
    conn.close()