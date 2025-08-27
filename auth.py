import psycopg2
import logging
import bcrypt
import uuid
from db.database import connect_db

def create_users_table():
    """
    Создаёт таблицу пользователей (если её ещё нет) с полями:
      - username (уникальный логин),
      - password (хэшированный),
      - email (адрес почты),
      - session_token (для функции "Запомнить меня").
    """
    conn = connect_db()
    if conn is None:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(255) UNIQUE NOT NULL,
                    password VARCHAR(255) NOT NULL,
                    email VARCHAR(255) NOT NULL,
                    session_token VARCHAR(255)
                );
            """)
        conn.commit()
    except Exception as e:
        logging.error(f"Ошибка создания таблицы пользователей: {e}")
    finally:
        conn.close()

def register_user(username, password, email):
    """
    Регистрирует нового пользователя, хэшируя пароль с помощью bcrypt.
    """
    conn = connect_db()
    if conn is None:
        return False
    try:
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (username, password, email) VALUES (%s, %s, %s)",
                (username, hashed_password.decode('utf-8'), email)
            )
        conn.commit()
        return True
    except Exception as e:
        logging.error(f"Ошибка регистрации пользователя: {e}")
        return False
    finally:
        conn.close()

def fetch_user(username):
    """
    Извлекает данные пользователя по логину.
    Возвращает кортеж (username, password, email, session_token) или None, если пользователь не найден.
    """
    conn = connect_db()
    if conn is None:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT username, password, email, session_token FROM users WHERE username = %s",
                (username,)
            )
            result = cur.fetchone()
        return result
    except Exception as e:
        logging.error(f"Ошибка получения пользователя: {e}")
        return None
    finally:
        conn.close()

def update_session_token(username, token):
    """
    Обновляет поле session_token для пользователя.
    """
    conn = connect_db()
    if conn is None:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE users SET session_token = %s WHERE username = %s", (token, username))
        conn.commit()
        return True
    except Exception as e:
        logging.error(f"Ошибка обновления session_token: {e}")
        return False
    finally:
        conn.close()

def fetch_user_by_token(token):
    """
    Извлекает пользователя по session_token.
    """
    conn = connect_db()
    if conn is None:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT username, password, email, session_token FROM users WHERE session_token = %s",
                (token,)
            )
            result = cur.fetchone()
        return result
    except Exception as e:
        logging.error(f"Ошибка получения пользователя по token: {e}")
        return None
    finally:
        conn.close()

def update_username(old_username, new_username):
    conn = connect_db()
    if conn is None:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE users SET username = %s WHERE username = %s", (new_username, old_username))
        conn.commit()
        return True
    except Exception as e:
        logging.error(f"Ошибка обновления логина: {e}")
        return False
    finally:
        conn.close()

def update_password(username, new_password):
    conn = connect_db()
    if conn is None:
        return False
    try:
        hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
        with conn.cursor() as cur:
            cur.execute("UPDATE users SET password = %s WHERE username = %s", (hashed_password.decode('utf-8'), username))
        conn.commit()
        return True
    except Exception as e:
        logging.error(f"Ошибка обновления пароля: {e}")
        return False
    finally:
        conn.close()

def delete_user(username):
    """
    Удаляет пользователя из базы.
    """
    conn = connect_db()
    if conn is None:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM users WHERE username = %s", (username,))
        conn.commit()
        return True
    except Exception as e:
        logging.error(f"Ошибка удаления пользователя: {e}")
        return False
    finally:
        conn.close()
        
def fetch_user_by_email(email):
    conn = connect_db()
    if conn is None:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT username FROM users WHERE email = %s", (email,))
            result = cur.fetchone()
        return result
    except Exception as e:
        logging.error(f"Ошибка получения пользователя по email: {e}")
        return None
    finally:
        conn.close()
