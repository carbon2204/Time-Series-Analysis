import psycopg2
import logging

# Конфигурация базы данных
DB_CONFIG = {
    "dbname": "exchange_rates",
    "user": "postgres",           
    "password": "3752",  
    "host": "localhost",
    "port": 5432
}

def connect_db():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        logging.error(f"Ошибка подключения к базе данных: {e}")
        return None

def insert_exchange_rate(currency_code, rate_to_rub, date):
    conn = connect_db()
    if conn is None:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rates (currency_code, rate_to_rub, date)
                VALUES (%s, %s, %s)
                ON CONFLICT (currency_code, date) DO UPDATE 
                SET rate_to_rub = EXCLUDED.rate_to_rub;
                """,
                (currency_code, rate_to_rub, date)
            )
        conn.commit()
        return True
    except Exception as e:
        logging.error(f"Ошибка записи в базу данных: {e}")
        return False
    finally:
        conn.close()


def fetch_exchange_rate(currency_code, date):
    conn = connect_db()
    if conn is None:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT rate_to_rub FROM rates
                WHERE currency_code = %s AND date = %s;
                """,
                (currency_code, date)
            )
            result = cur.fetchone()
            return result[0] if result else None
    except Exception as e:
        logging.error(f"Ошибка извлечения данных из базы: {e}")
        return None
    finally:
        conn.close()
