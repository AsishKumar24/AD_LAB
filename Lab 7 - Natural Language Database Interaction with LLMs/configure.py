import mysql.connector
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key="AIzaSyCfgKWRY9xCV0APhTifg1GnREnW5SLDzvM")

# MySQL Connection
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="Asu@2003",
        database="EMPLOYEE"
    )
