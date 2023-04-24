import pymysql as mysql
import pymysql.err as error


def create_connection(db_name, host, user, password, port=3306, show_info=True):
    """Connects to an existing database, or create it."""
    try:
        # If the database exists at the specified location, then
        # a connection to the database is established. Otherwise,
        # a new database is created at the specified location,
        # and a connection is established.
        connection = mysql.connect(host=host, user=user, password=password, database=db_name, port=port)
        if show_info:
            print(f"Connection to '{db_name}' database successful.")
    except error.OperationalError:
        try:
            # Connect to existing database.
            connection = mysql.connect(host=host, user=user, password=password)
            connection.cursor().execute(f'CREATE DATABASE {db_name}')
            connection = mysql.connect(host=host, user=user, password=password, database=db_name, port=port)
            if show_info:
                print(f"Cannot connect to '{db_name}'. A new database is created instead.")
        except Exception as err:
            # Print error message if something goes wrong.
            raise Exception(f"The error '{err}' occurred when connecting to '{db_name}'.")

    return connection


def execute_query(connection, query):
    """Executes query in SQLite via the execute method.
    A query is passed as a string.

    Args:
        connection: the database connection.
        query (str): the queried change committed to the database.
    """
    cursor = connection.cursor()  # a cursor object to fetch results
    try:
        cursor.execute(query)
        connection.commit()  # a change committed to the database
    except error.OperationalError as e:
        raise Exception(f"The error '{e}' occurred when executing: \n{query}")


def execute_parameterized_query(connection, query, parameters, replace=False):
    """Executes query in SQLite via the execute method.
    A query is passed as a string.

    Args:
        connection: the database connector.
        query (str): the queried change committed to the database.
        parameters (list): values to insert into the database.
    """
    msg = lambda x: f"The error '{x}' occurred when executing: \n{query}"

    cursor = connection.cursor()  # a cursor object to fetch results
    try:
        try:
            cursor.execute(query, parameters)
            connection.commit()  # a change committed to the database
        except error.IntegrityError as err:
            if err.args[0] == 1062:
                if ('INSERT' in query) and replace:
                    # Replace duplicate data.
                    cursor.execute(query.replace('INSERT', 'REPLACE'), parameters)
                    connection.commit()  # a change committed to the database
                else:
                    pass  # Skip update.
            else:
                raise Exception(msg(err))
    except error.OperationalError as err:
        raise Exception(msg(err))


def create_tables(connection):
    """Creates tables in train database."""
    query = f"""
    CREATE TABLE IF NOT EXISTS config_deepmodel_train(
        task_id VARCHAR(124),
        task_name VARCHAR(124),
        model_name VARCHAR(255),
        model_to_restore VARCHAR(255),
        epoch_interval_saving INTEGER,
        load_generator_data BOOLEAN,
        apply_augmentation BOOLEAN,
        date_created DATETIME
    ) ENGINE=InnoDB DEFAULT CHARSET=latin1;
    """
    execute_query(connection, query)

