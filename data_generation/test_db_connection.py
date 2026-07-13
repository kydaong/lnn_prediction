#Quick Connectivity test with DB
from db_connection import get_connection

def main() -> None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DB_NAME(), SUSER_SNAME(), @@VERSION;")
    db_name, user_name, version = cursor.fetchone()
    print(f"Connected OK")
    print(f"  Database : {db_name}")
    print(f"  Logged in as: {user_name}")
    print(f"  Server version: {version.splitlines()[0]}")
    conn.close()


if __name__ == "__main__":
    main()
