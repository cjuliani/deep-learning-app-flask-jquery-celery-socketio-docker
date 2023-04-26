GRANT ALL PRIVILEGES ON *.* TO batman;
FLUSH PRIVILEGES;

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