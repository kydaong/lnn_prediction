
from __future__ import annotations
import os
import pyodbc
from dotenv import load_dotenv

load_dotenv() #read key-value pairs from .env file and set them as environment variables
ODBC_DRIVER = "{ODBC Driver 18 for SQL Server}"

def get_connection() -> pyodbc.Connection:
    server = os.environ["DB_SERVER"]
    database = os.environ["DB_NAME"]
    user = os.environ["DB_USER"]
    password = os.environ["DB_PASSWORD"] 

    conn_str = (
        f"DRIVER={ODBC_DRIVER};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={user};"
        f"PWD={password};"
        "Encrypt=yes;"
        "TrustServerCertificate=no;"
        "Connection Timeout=30;"
    )
    return pyodbc.connect(conn_str)


def get_sqlalchemy_engine():
    """Engine for pandas.to_sql — used later for bulk-loading generated data."""
    import urllib.parse
    from sqlalchemy import create_engine

    server = os.environ["DB_SERVER"]
    database = os.environ["DB_NAME"]
    user = os.environ["DB_USER"]
    password = os.environ["DB_PASSWORD"]

    odbc_str = (
        f"DRIVER={ODBC_DRIVER};SERVER={server};DATABASE={database};"
        f"UID={user};PWD={password};Encrypt=yes;TrustServerCertificate=no;"
    )
    params = urllib.parse.quote_plus(odbc_str)
    return create_engine(f"mssql+pyodbc:///?odbc_connect={params}", fast_executemany=True)
