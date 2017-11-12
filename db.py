import numpy as np
import io
import sqlite3

def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


def connect(sqlite_file):
    # Converts np.array to TEXT when inserting
    print("Connecting to the database ...")
    sqlite3.register_adapter(np.ndarray, adapt_array)

    # Converts TEXT to np.array when selecting
    sqlite3.register_converter("array", convert_array)

    # Connecting to the database file
    conn = sqlite3.connect(sqlite_file, detect_types = sqlite3.PARSE_DECLTYPES)
    # c = conn.cursor()
    print("Connection established!")

    return conn


def create_table(c, table_name_1, col_names):
    # global new_field_1, new_field_2, field_type
    # Creating a new SQLite table with 1 column
    print("Creating Table ...")
    col_names = ','.join(col_names)
    c.execute('CREATE TABLE IF NOT EXISTS {tn} ({cn})'\
            .format(tn = table_name_1, cn = col_names))
    print("Done!")


def disconnect(conn):
    # global conn
    # Committing changes and closing the connection to the database file
    conn.commit()
    print("Commited Changes to DB!")
    conn.close()
    print("Disconnected from db!")