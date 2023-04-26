import config
from .queries import execute_query, create_connection


class MainDatabase:
    """Class consisting of methods used to insert data
    in the database.

    Attributes:
        connection: the database connector.
    """
    def __init__(self, replace, db_name, host, user, password, port=3306, show_info=True):
        if replace:
            # Delete existing database.
            query = f"DROP DATABASE {db_name};"
            self.connection = create_connection(
                None, host, user, password, port, show_info=show_info)
            self.connection.cursor().execute(query)

        # Connect or create database.
        self.connection = create_connection(
            db_name, host, user, password, port, show_info=show_info)

    def insert_train_config(self, values):
        query = """
        INSERT IGNORE INTO
          config_deepmodel_train (task_id, task_name, model_name, model_to_restore, 
          epoch_interval_saving, load_generator_data, apply_augmentation, date_created)
        VALUES
          ('{}', '{}', '{}', '{}', {}, {}, {}, '{}')
        """.format(*values)
        execute_query(self.connection, query)


def get_database(replace=False, **kwargs):
    """Returns a known or newly created database
    instance."""
    return MainDatabase(
        replace=replace,
        db_name=config.DB_NAME, **kwargs)
