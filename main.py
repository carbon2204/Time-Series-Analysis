import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime, timedelta
import requests
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, acf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pmdarima import auto_arima
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
from tkcalendar import DateEntry
import logging
import sys
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import webbrowser
import os
import json
import uuid
import re
import smtplib
import string
import random
from email.mime.text import MIMEText
import bcrypt
import io
import keyring 

from PIL import Image, ImageTk

# Константа для безопасного хранения сессионного токена в keyring
APP_NAME = "exchange_rate_app"

from db.database import fetch_exchange_rate, insert_exchange_rate, connect_db
from auth import (
    create_users_table, register_user, fetch_user, update_session_token,
    fetch_user_by_token, update_username, update_password, delete_user,
    fetch_user_by_email
)

logging.basicConfig(level=logging.ERROR, stream=sys.stdout)
logger = logging.getLogger()

API_URL = "https://www.cbr-xml-daily.ru"
EMAIL_REGEX = re.compile(r"[^@]+@[^@]+\.[^@]+")

def send_login_notification(recipient_email, username, code):
    """Отправляет уведомление с кодом верификации на указанный email пользователя."""
    subject = "Код верификации для входа в систему"
    body = f"Здравствуйте, {username}!\n\nВаш код верификации: {code}\n\nЕсли вы не пытались войти, проигнорируйте это письмо."
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = "dmitry.kuzmin.work@example.com"      
    msg['To'] = recipient_email

    try:
        smtp_server = "smtp.gmail.com"         
        smtp_port = 465                          
        smtp_user = "dmitry.kuzmin.work@gmail.com"     
        smtp_password = "cada rtpj ednk ykzg"    

        server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        server.login(smtp_user, smtp_password)
        server.send_message(msg)
        server.quit()
        print("Уведомление с кодом отправлено на", recipient_email)
    except Exception as e:
        print("Ошибка отправки email:", e)

###############################################################################
# Функции для работы с keyring (безопасное хранение сессионного токена)

def save_session_token(token):
    """Сохраняем сессионный токен в безопасном хранилище через keyring."""
    keyring.set_password(APP_NAME, "session_token", token)

def load_session_token():
    """Возвращаем сохранённый сессионный токен из keyring (или None, если его нет)."""
    return keyring.get_password(APP_NAME, "session_token")

def delete_session_token():
    """Удаляем сессионный токен из keyring."""
    try:
        keyring.delete_password(APP_NAME, "session_token")
    except keyring.errors.PasswordDeleteError:
        pass

###############################################################################
# Функция для получения новостей с NewsAPI
NEWS_API_KEY = "f09ed1d552554911b19cd47e1ad65a69"  # Замените на ваш API-ключ от NewsAPI.org

def fetch_currency_news(query="валюта", language="ru", page_size=10):
    """
    Получает новости по заданному запросу.
    Параметры:
      query     - поисковый запрос (по умолчанию "валюта")
      language  - язык новостей (по умолчанию "ru")
      page_size - количество новостей
    """
    url = (f"https://newsapi.org/v2/everything?q={query}&language={language}"
           f"&pageSize={page_size}&apiKey={NEWS_API_KEY}")
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get('articles', [])
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка получения новостей: {e}")
        messagebox.showerror("Ошибка", "Не удалось получить новости.")
        return []

###############################################################################
# Функция для установки прав доступа для Unix (если потребуется)
def set_file_permissions_unix(file_path):
    os.chmod(file_path, 0o600)

###############################################################################
# Окно авторизации и регистрации

class AuthWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Авторизация / Регистрация")
        self.root.geometry("300x300")
        self.verification_code = None  

        notebook = ttk.Notebook(root)
        notebook.pack(expand=True, fill='both')
        
        self.login_frame = ttk.Frame(notebook)
        notebook.add(self.login_frame, text="Вход")
        self.build_login(self.login_frame)
        
        self.reg_frame = ttk.Frame(notebook)
        notebook.add(self.reg_frame, text="Регистрация")
        self.build_registration(self.reg_frame)
    
    def build_login(self, frame):
        ttk.Label(frame, text="Логин:", font=('Helvetica', 12)).pack(pady=5)
        self.login_username_entry = ttk.Entry(frame, font=('Helvetica', 12))
        self.login_username_entry.pack(pady=5)
        
        ttk.Label(frame, text="Пароль:", font=('Helvetica', 12)).pack(pady=5)
        self.login_password_entry = ttk.Entry(frame, show="*", font=('Helvetica', 12))
        self.login_password_entry.pack(pady=5)
        
        self.remember_var = tk.BooleanVar()
        self.remember_check = ttk.Checkbutton(frame, text="Запомнить меня", variable=self.remember_var)
        self.remember_check.pack(pady=5)
        
        ttk.Button(frame, text="Войти", command=self.login).pack(pady=5)
    
    def build_registration(self, frame):
        ttk.Label(frame, text="Логин:", font=('Helvetica', 12)).pack(pady=5)
        self.reg_username_entry = ttk.Entry(frame, font=('Helvetica', 12))
        self.reg_username_entry.pack(pady=5)
        
        ttk.Label(frame, text="Пароль:", font=('Helvetica', 12)).pack(pady=5)
        self.reg_password_entry = ttk.Entry(frame, show="*", font=('Helvetica', 12))
        self.reg_password_entry.pack(pady=5)
        
        ttk.Label(frame, text="Email:", font=('Helvetica', 12)).pack(pady=5)
        self.reg_email_entry = ttk.Entry(frame, font=('Helvetica', 12))
        self.reg_email_entry.pack(pady=5)
        
        ttk.Button(frame, text="Зарегистрироваться", command=self.register).pack(pady=10)
    
    def login(self):
        username = self.login_username_entry.get().strip()
        password = self.login_password_entry.get().strip()
        if not username or not password:
            messagebox.showwarning("Предупреждение", "Заполните все поля")
            return
        
        user = fetch_user(username)
        if user:
            db_username, db_password, email, _ = user
            if bcrypt.checkpw(password.encode('utf-8'), db_password.encode('utf-8')):
                code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
                self.verification_code = code
                if email and EMAIL_REGEX.match(email):
                    send_login_notification(email, username, code)
                else:
                    messagebox.showerror("Ошибка", "Некорректный email в профиле")
                    return
                self.open_verification_window(username)
            else:
                messagebox.showerror("Ошибка", "Неверный пароль")
        else:
            messagebox.showerror("Ошибка", "Пользователь не найден")
    
    def open_verification_window(self, username):
        ver_win = tk.Toplevel(self.root)
        ver_win.title("Проверка кода")
        ver_win.geometry("300x200")
    
        ttk.Label(ver_win, text="Введите код из письма:", font=('Helvetica', 12)).pack(pady=10)
        code_entry = ttk.Entry(ver_win, font=('Helvetica', 12))
        code_entry.pack(pady=5)
    
        countdown_label = ttk.Label(ver_win, text="", font=('Helvetica', 12))
        countdown_label.pack(pady=5)
        
        resend_button = ttk.Button(ver_win, text="Отправить код заново", command=lambda: resend_code(), state="disabled")
        resend_button.pack(pady=5)
        
        self.countdown_time = 30

        def update_countdown():
            nonlocal resend_button
            if self.countdown_time > 0:
                countdown_label.config(text=f"Отправить код можно через {self.countdown_time} сек.")
                self.countdown_time -= 1
                ver_win.after(1000, update_countdown)
            else:
                countdown_label.config(text="Время истекло!")
                resend_button.config(state="normal")
        
        def resend_code():
            new_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
            self.verification_code = new_code

            user = fetch_user(username)
            if user:
                _, _, email, _ = user
                if email and EMAIL_REGEX.match(email):
                    send_login_notification(email, username, new_code)
                    messagebox.showinfo("Информация", "Новый код отправлен на вашу почту.")
                    self.countdown_time = 30
                    resend_button.config(state="disabled")
                    update_countdown()
                else:
                    messagebox.showerror("Ошибка", "Некорректный email в профиле")
        
        update_countdown()
        
        def verify():
            user_input = code_entry.get().strip()
            if user_input == self.verification_code:
                # Если установлен флажок "Запомнить меня", сохраняем сессионный токен через keyring
                if self.remember_var.get():
                    token = str(uuid.uuid4())
                    update_session_token(username, token)
                    save_session_token(token)
                messagebox.showinfo("Успех", "Верификация успешна!")
                ver_win.destroy()
                self.root.destroy()
                main_root = tk.Tk()
                ExchangeRateApp(main_root, username)
                main_root.mainloop()
            else:
                messagebox.showerror("Ошибка", "Неверный код")
        
        ttk.Button(ver_win, text="Подтвердить", command=verify).pack(pady=10)

    
    def register(self):
        username = self.reg_username_entry.get().strip()
        password = self.reg_password_entry.get().strip()
        email = self.reg_email_entry.get().strip()
        if not username or not password or not email:
            messagebox.showwarning("Предупреждение", "Заполните все поля")
            return
        if not EMAIL_REGEX.match(email):
            messagebox.showerror("Ошибка", "Некорректный email")
            return
        if fetch_user(username):
            messagebox.showwarning("Предупреждение", "Пользователь с таким логином уже существует")
            return
        if fetch_user_by_email(email):
            messagebox.showwarning("Предупреждение", "Пользователь с таким email уже существует")
            return
        if register_user(username, password, email):
            messagebox.showinfo("Успех", "Регистрация успешна! Теперь войдите.")
        else:
            messagebox.showerror("Ошибка", "Ошибка регистрации. Попробуйте позже.")

###############################################################################
# Функции для работы с валютами

def check_if_date_available(date):
    try:
        response = requests.get(f"{API_URL}/archive/{date}/daily_json.js")
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        logger.error(f"Error checking date availability: {e}")
        return False

def get_available_currencies():
    try:
        response = requests.get(f"{API_URL}/daily_json.js")
        response.raise_for_status()
        data = response.json()
        return data['Valute']
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching available currencies: {e}")
        messagebox.showerror("Ошибка", "Не удалось получить список валют.")
        return {}

def get_time_series_between_currencies(from_currency, to_currency, start_date, end_date):
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    delta = timedelta(days=1)
    series = {}
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        rate_from = fetch_exchange_rate(from_currency, date_str)
        rate_to = fetch_exchange_rate(to_currency, date_str)
        if rate_from is None or rate_to is None:
            rate_from = get_exchange_rate_from_rub(from_currency, date_str)
            rate_to = get_exchange_rate_from_rub(to_currency, date_str)
            if rate_from is not None:
                insert_exchange_rate(from_currency, rate_from, date_str)
            if rate_to is not None:
                insert_exchange_rate(to_currency, rate_to, date_str)
        if rate_from is not None and rate_to is not None:
            series[date_str] = rate_from / rate_to
        else:
            print(f"Курс для {from_currency} или {to_currency} на {date_str} отсутствует")
        current_date += delta
    if not series:
        messagebox.showwarning("Нет данных", "Не удалось получить данные за указанный период.")
    return series

def get_exchange_rate_from_rub(currency_code, date):
    stored_rate = fetch_exchange_rate(currency_code, date)
    if stored_rate:
        print(f"Курс из БД для {currency_code} на {date}: {stored_rate}")
        return stored_rate
    formatted_date = date.replace('-', '/')
    print(f"Запрос к API для {currency_code} на {formatted_date}")
    try:
        response = requests.get(f"{API_URL}/archive/{formatted_date}/daily_json.js")
        print(f"URL запроса: {response.url}")
        response.raise_for_status()
        data = response.json()
        if currency_code in data['Valute']:
            valute_data = data['Valute'][currency_code]
            rate = valute_data['Value'] / valute_data['Nominal']
            insert_exchange_rate(currency_code, rate, date)
            print(f"Курс {currency_code} на {date} из API: {rate}")
            return rate
        else:
            print(f"{currency_code} не найден в API на {formatted_date}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка API: {e}")
        return None

def get_exchange_rate_between_currencies(from_currency, to_currency, date):
    rate_from = get_exchange_rate_from_rub(from_currency, date)
    rate_to = get_exchange_rate_from_rub(to_currency, date)
    if rate_from is not None and rate_to is not None:
        return rate_from / rate_to
    return None

def get_exchange_rate_to_all(from_currency, date):
    try:
        response = requests.get(f"{API_URL}/archive/{date}/daily_json.js")
        response.raise_for_status()
        data = response.json()
        currencies = data['Valute']
        rates = {}
        rate_from = get_exchange_rate_from_rub(from_currency, date)
        if rate_from:
            for code, valute_data in currencies.items():
                rate_to = valute_data['Value'] / valute_data['Nominal']
                rates[code] = rate_from / rate_to
            return rates
        else:
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка получения курсов: {e}")
        return None

def get_time_series(from_currency, to_currency, start_date, end_date):
    current_date = datetime.strptime(start_date, '%Y/%m/%d')
    end_date = datetime.strptime(end_date, '%Y/%m/%d')
    delta = timedelta(days=1)
    result = {}
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        rate = get_exchange_rate_between_currencies(from_currency, to_currency, date_str)
        if rate is not None:
            result[date_str] = rate
        current_date += delta
    if not result:
        messagebox.showwarning("Нет данных", "Данных за период не найдено.")
    return result

def analyze_time_series(time_series):
    df = pd.DataFrame(list(time_series.items()), columns=['Date', 'Rate'])
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df['Rate'] = pd.to_numeric(df['Rate'], errors='coerce')
    df = df.asfreq('D')
    df['Rate'] = df['Rate'].interpolate(method='time')
    if len(df) < 14:
        return "Недостаточно данных для анализа временного ряда."
    stl = STL(df['Rate'], period=7)
    result = stl.fit()
    result_adf = adfuller(df['Rate'].dropna())
    is_stationary = result_adf[1] < 0.05
    autocorr = acf(df['Rate'].dropna(), nlags=5, fft=False)
    analysis = {
        "trend": result.trend.dropna(),
        "seasonal": result.seasonal.dropna(),
        "residual": result.resid.dropna(),
        "stationarity_p_value": result_adf[1],
        "is_stationary": is_stationary,
        "autocorrelation": autocorr.tolist()
    }
    return analysis

def forecast_with_holt_winters(time_series, forecast_days=10):
    df = pd.Series(time_series)
    df.index = pd.to_datetime(df.index)
    df = df.asfreq('D')
    df = pd.to_numeric(df, errors='coerce').interpolate(method='time')
    try:
        model = ExponentialSmoothing(df, trend="add", seasonal="add", seasonal_periods=3)
        model_fit = model.fit()
    except Exception as e:
        logger.error(f"Ошибка модели Holt-Winters: {e}")
        messagebox.showerror("Ошибка", "Не удалось построить прогноз.")
        return None, None, None, None
    in_sample_forecast = model_fit.fittedvalues
    future_start_date = df.index[-1] + timedelta(days=1)
    out_sample_forecast = model_fit.forecast(steps=forecast_days)
    future_dates = pd.date_range(start=future_start_date, periods=forecast_days, freq='D')
    out_sample_forecast.index = future_dates
    return df, in_sample_forecast, out_sample_forecast, model_fit

def forecast_with_arima(time_series, train_ratio=0.8, test_ratio=0.2, forecast_days=7):
    df = pd.Series(time_series)
    df.index = pd.to_datetime(df.index)
    df = df.asfreq('D')
    df = df.interpolate(method='time')
    train_size = int(len(df) * train_ratio)
    test_size = int(len(df) * test_ratio)
    train_data = df[:train_size]
    test_data = df[train_size:train_size + test_size]
    future_start_date = test_data.index[-1] + timedelta(days=1)
    try:
        model = auto_arima(train_data, seasonal=True, m=7, trace=False, error_action='ignore', suppress_warnings=True)
        model_fit = model.fit(train_data)
    except Exception as e:
        logger.error(f"Ошибка ARIMA: {e}")
        messagebox.showerror("Ошибка", "Не удалось построить ARIMA модель.")
        return None, None, None, None, None, None
    in_sample_forecast = model_fit.predict(n_periods=len(test_data))
    in_sample_forecast = pd.Series(in_sample_forecast, index=test_data.index)
    out_sample_forecast = model_fit.predict(n_periods=forecast_days)
    future_dates = pd.date_range(start=future_start_date, periods=forecast_days, freq='D')
    out_sample_forecast = pd.Series(out_sample_forecast, index=future_dates)
    mae = mean_absolute_error(test_data, in_sample_forecast)
    rmse = np.sqrt(mean_squared_error(test_data, in_sample_forecast))
    return train_data, test_data, in_sample_forecast, out_sample_forecast, mae, rmse

###############################################################################
# Основное приложение

class ExchangeRateApp:
    def __init__(self, root, username):
        self.root = root
        self.username = username  # текущий пользователь
        self.root.title("Анализ курса валют")
        self.root.geometry("800x600")
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TButton', font=('Helvetica', 12))
        self.style.configure('TLabel', font=('Helvetica', 12))
        self.style.configure('TEntry', font=('Helvetica', 12))
        self.style.configure('TCombobox', font=('Helvetica', 12))
        self.style.configure('TNotebook.Tab', font=('Helvetica', 12))
        self.currencies = get_available_currencies()
        self.currency_codes = ["RUB"] + list(self.currencies.keys())
        self.create_menu()
        self.news_images = []  # Для хранения изображений новостей

    def create_menu(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        frame = ttk.Frame(self.root)
        frame.pack(pady=20, expand=True)
        ttk.Label(frame, text=f"Главное меню (Пользователь: {self.username})", font=("Arial", 16)).grid(row=0, column=0, columnspan=2, pady=10)
        ttk.Button(frame, text="1. Курс валюты к другой валюте", command=self.show_exchange_rate_interface).grid(row=1, column=0, padx=5, pady=5, sticky='ew')
        ttk.Button(frame, text="2. Курс валюты ко всем валютам", command=self.show_all_rates_interface).grid(row=2, column=0, padx=5, pady=5, sticky='ew')
        ttk.Button(frame, text="3. Курс валюты в промежуток времени", command=self.show_time_series_interface).grid(row=3, column=0, padx=5, pady=5, sticky='ew')
        ttk.Button(frame, text="4. Новости", command=self.show_news_interface).grid(row=4, column=0, padx=5, pady=5, sticky='ew')
        ttk.Button(frame, text="Профиль", command=self.show_profile_window).grid(row=5, column=0, padx=5, pady=5, sticky='ew')
        ttk.Button(frame, text="0. Выход", command=self.root.quit).grid(row=6, column=0, padx=5, pady=5, sticky='ew')
        frame.columnconfigure(0, weight=1)

    def show_profile_window(self):
        prof_win = tk.Toplevel(self.root)
        prof_win.title("Профиль")
        prof_win.geometry("350x400")
        user = fetch_user(self.username)
        if not user:
            messagebox.showerror("Ошибка", "Не удалось загрузить данные пользователя")
            return
        username, password, email, _ = user
        
        ttk.Label(prof_win, text="Данные пользователя", font=('Helvetica', 14)).pack(pady=10)
        ttk.Label(prof_win, text=f"Логин: {username}", font=('Helvetica', 12)).pack(pady=5)
        ttk.Label(prof_win, text=f"Email: {email}", font=('Helvetica', 12)).pack(pady=5)
        
        ttk.Label(prof_win, text="Сменить логин", font=('Helvetica', 12)).pack(pady=5)
        new_login_entry = ttk.Entry(prof_win, font=('Helvetica', 12))
        new_login_entry.pack(pady=5)
        
        ttk.Label(prof_win, text="Сменить пароль", font=('Helvetica', 12)).pack(pady=5)
        new_pass_entry = ttk.Entry(prof_win, show="*", font=('Helvetica', 12))
        new_pass_entry.pack(pady=5)
        
        def update_credentials():
            new_login = new_login_entry.get().strip()
            new_pass = new_pass_entry.get().strip()
            if new_login:
                if update_username(self.username, new_login):
                    messagebox.showinfo("Успех", "Логин обновлён")
                    self.username = new_login
                else:
                    messagebox.showerror("Ошибка", "Не удалось обновить логин")
            if new_pass:
                if update_password(self.username, new_pass):
                    messagebox.showinfo("Успех", "Пароль обновлён")
                else:
                    messagebox.showerror("Ошибка", "Не удалось обновить пароль")
            self.create_menu()
            prof_win.destroy()
        ttk.Button(prof_win, text="Сохранить изменения", command=update_credentials).pack(pady=10)
        
        def logout():
            delete_session_token()
            messagebox.showinfo("Выход", "Вы вышли из аккаунта")
            self.root.destroy()
        ttk.Button(prof_win, text="Выйти из аккаунта", command=logout).pack(pady=10)
        
        def delete_account():
            if messagebox.askyesno("Подтверждение", "Вы уверены, что хотите удалить аккаунт?"):
                if delete_user(self.username):
                    # Удаляем сессионный токен из keyring при удалении пользователя
                    delete_session_token()
                    messagebox.showinfo("Удалено", "Аккаунт удалён")
                    self.root.destroy()
                else:
                    messagebox.showerror("Ошибка", "Не удалось удалить аккаунт")
        ttk.Button(prof_win, text="Удалить аккаунт", command=delete_account).pack(pady=10)

    def show_exchange_rate_interface(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        frame = ttk.Frame(self.root)
        frame.pack(pady=20, expand=True)
        ttk.Label(frame, text="Курс валюты к другой валюте", font=("Arial", 14)).grid(row=0, column=0, columnspan=2, pady=10)
        ttk.Label(frame, text="Исходная валюта:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        from_currency = ttk.Combobox(frame, values=self.currency_codes)
        from_currency.grid(row=1, column=1, padx=5, pady=5, sticky='w')
        ttk.Label(frame, text="Целевая валюта:").grid(row=2, column=0, padx=5, pady=5, sticky='e')
        to_currency = ttk.Combobox(frame, values=self.currency_codes)
        to_currency.grid(row=2, column=1, padx=5, pady=5, sticky='w')
        ttk.Label(frame, text="Дата:", font=("Arial", 12)).grid(row=3, column=0, padx=5, pady=5, sticky='e')
        date_entry = DateEntry(frame, date_pattern='yyyy/mm/dd')
        date_entry.grid(row=3, column=1, padx=5, pady=5, sticky='w')
        result_label = ttk.Label(frame, text="", foreground="blue")
        result_label.grid(row=5, column=0, columnspan=2)
        def get_rate():
            from_curr = from_currency.get()
            to_curr = to_currency.get()
            date = date_entry.get()
            if from_curr and to_curr and date:
                rate = get_exchange_rate_between_currencies(from_curr, to_curr, date)
                if rate:
                    result_label.config(text=f"Курс {from_curr} к {to_curr} на {date}: {rate:.5f}")
                else:
                    result_label.config(text="Не удалось получить курс.")
            else:
                result_label.config(text="Заполните все поля.")
        ttk.Button(frame, text="Получить курс", command=get_rate).grid(row=4, column=0, columnspan=2, pady=10)
        ttk.Button(frame, text="Назад", command=self.create_menu).grid(row=6, column=0, columnspan=2, pady=10)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

    def show_all_rates_interface(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        frame = ttk.Frame(self.root)
        frame.pack(pady=20, expand=True)
        ttk.Label(frame, text="Курс валюты ко всем валютам", font=("Arial", 14)).grid(row=0, column=0, columnspan=2, pady=10)
        ttk.Label(frame, text="Исходная валюта:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        from_currency = ttk.Combobox(frame, values=self.currency_codes)
        from_currency.grid(row=1, column=1, padx=5, pady=5, sticky='w')
        ttk.Label(frame, text="Дата:", font=("Arial", 12)).grid(row=2, column=0, padx=5, pady=5, sticky='e')
        date_entry = DateEntry(frame, date_pattern='yyyy/mm/dd')
        date_entry.grid(row=2, column=1, padx=5, pady=5, sticky='w')
        result_text = tk.Text(frame, wrap="word", height=15, width=60)
        result_text.grid(row=4, column=0, columnspan=2, pady=10)
        def get_all_rates():
            from_curr = from_currency.get()
            date = date_entry.get()
            if from_curr and date:
                rates = get_exchange_rate_to_all(from_curr, date)
                if rates:
                    result_text.delete("1.0", tk.END)
                    result_text.insert(tk.END, f"Курсы {from_curr} на {date}:\n")
                    for currency, rate in rates.items():
                        result_text.insert(tk.END, f"{currency}: {rate:.5f}\n")
                else:
                    result_text.delete("1.0", tk.END)
                    result_text.insert(tk.END, "Не удалось получить курсы.")
            else:
                result_text.delete("1.0", tk.END)
                result_text.insert(tk.END, "Заполните все поля.")
        ttk.Button(frame, text="Получить курсы", command=get_all_rates).grid(row=3, column=0, columnspan=2, pady=10)
        ttk.Button(frame, text="Назад", command=self.create_menu).grid(row=5, column=0, columnspan=2, pady=10)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

    def show_time_series_interface(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        frame = ttk.Frame(self.root)
        frame.pack(pady=20, expand=True)
        ttk.Label(frame, text="Курс валюты в промежуток времени", font=("Arial", 14)).grid(row=0, column=0, columnspan=2, pady=10)
        ttk.Label(frame, text="Исходная валюта:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        from_currency = ttk.Combobox(frame, values=self.currency_codes)
        from_currency.grid(row=1, column=1, padx=5, pady=5, sticky='w')
        ttk.Label(frame, text="Целевая валюта:").grid(row=2, column=0, padx=5, pady=5, sticky='e')
        to_currency = ttk.Combobox(frame, values=self.currency_codes)
        to_currency.grid(row=2, column=1, padx=5, pady=5, sticky='w')
        ttk.Label(frame, text="Начальная дата:", font=("Arial", 12)).grid(row=3, column=0, padx=5, pady=5, sticky='e')
        start_date_entry = DateEntry(frame, date_pattern='yyyy-mm-dd')
        start_date_entry.grid(row=3, column=1, padx=5, pady=5, sticky='w')
        ttk.Label(frame, text="Конечная дата:", font=("Arial", 12)).grid(row=4, column=0, padx=5, pady=5, sticky='e')
        end_date_entry = DateEntry(frame, date_pattern='yyyy-mm-dd')
        end_date_entry.grid(row=4, column=1, padx=5, pady=5, sticky='w')
        def get_time_series_data():
            from_curr = from_currency.get()
            to_curr = to_currency.get()
            start_date = start_date_entry.get()
            end_date = end_date_entry.get()
            if from_curr and to_curr and start_date and end_date:
                time_series = get_time_series_between_currencies(from_curr, to_curr, start_date, end_date)
                if time_series:
                    analysis = analyze_time_series(time_series)
                    if isinstance(analysis, str):
                        messagebox.showinfo("Информация", analysis)
                    else:
                        self.show_analysis_tabs(time_series, analysis)
                else:
                    messagebox.showwarning("Нет данных", "Данных за период не найдено.")
            else:
                messagebox.showwarning("Предупреждение", "Заполните все поля.")
        ttk.Button(frame, text="Получить временной ряд", command=get_time_series_data).grid(row=5, column=0, columnspan=2, pady=10)
        ttk.Button(frame, text="Назад", command=self.create_menu).grid(row=6, column=0, columnspan=2, pady=10)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

    def show_analysis_tabs(self, time_series, analysis):
        for widget in self.root.winfo_children():
            widget.destroy()
        notebook = ttk.Notebook(self.root)
        notebook.pack(expand=1, fill='both')
        # Вкладка "Исходный график"
        tab1 = ttk.Frame(notebook)
        notebook.add(tab1, text="Исходный график")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=list(time_series.keys()), y=list(time_series.values()),
                                  mode='lines', name='Исходные данные', line=dict(color='blue')))
        fig1.update_layout(title='Исходный график', xaxis_title='Дата', yaxis_title='Курс', template='plotly_white')
        with open("original_data.html", "w", encoding='utf-8') as f:
            f.write(pio.to_html(fig1, full_html=False))
        webbrowser.open('file://' + os.path.realpath("original_data.html"))
        # Вкладка "Тренд, Сезонность и Шум"
        tab2 = ttk.Frame(notebook)
        notebook.add(tab2, text="Тренд, Сезонность и Шум")
        fig2 = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("Тренд", "Сезонность", "Шум"))
        fig2.add_trace(go.Scatter(x=analysis["trend"].index, y=analysis["trend"], mode='lines', name='Тренд'), row=1, col=1)
        fig2.add_trace(go.Scatter(x=analysis["seasonal"].index, y=analysis["seasonal"], mode='lines', name='Сезонность'), row=2, col=1)
        fig2.add_trace(go.Scatter(x=analysis["residual"].index, y=analysis["residual"], mode='lines', name='Шум'), row=3, col=1)
        fig2.update_layout(title='Тренд, Сезонность и Шум', xaxis_title='Дата', template='plotly_white')
        with open("trend_seasonality_residual.html", "w", encoding='utf-8') as f:
            f.write(pio.to_html(fig2, full_html=False))
        webbrowser.open('file://' + os.path.realpath("trend_seasonality_residual.html"))
        # Вкладка "Вывод анализа"
        tab3 = ttk.Frame(notebook)
        notebook.add(tab3, text="Вывод анализа")
        analysis_text = tk.Text(tab3, wrap="word")
        analysis_text.insert("1.0", f"Стационарность (p-value): {analysis['stationarity_p_value']:.5f}\n")
        analysis_text.insert("end", f"Ряд стационарен: {'Да' if analysis['is_stationary'] else 'Нет'}\n")
        analysis_text.insert("end", "Тренд:\n" + analysis["trend"].head().to_string() + "\n")
        analysis_text.insert("end", "Сезонность:\n" + analysis["seasonal"].head().to_string() + "\n")
        analysis_text.insert("end", "Шум:\n" + analysis["residual"].head().to_string() + "\n")
        analysis_text.pack(expand=1, fill="both")
        # Вкладка "Прогноз"
        tab4 = ttk.Frame(notebook)
        notebook.add(tab4, text="Прогноз")
        full_data, in_sample_forecast, out_sample_forecast, _ = forecast_with_holt_winters(time_series)
        total_points = len(full_data)
        split_point = total_points // 3
        original_data_blue = full_data[:split_point]
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=full_data.index, y=full_data, mode='lines', name='Исходные данные (все)', line=dict(color='lightgray')))
        fig4.add_trace(go.Scatter(x=original_data_blue.index, y=original_data_blue, mode='lines', name='Обучение', line=dict(color='blue')))
        remaining_data_green = in_sample_forecast[split_point:]
        fig4.add_trace(go.Scatter(x=remaining_data_green.index, y=remaining_data_green, mode='lines', name='Прогноз на известные данные', line=dict(color='green', dash='dash')))
        fig4.add_trace(go.Scatter(x=out_sample_forecast.index, y=out_sample_forecast, mode='lines', name='Прогноз на будущее', line=dict(color='orange', dash='dash')))
        fig4.update_layout(title='Общий прогноз курса валюты', xaxis_title='Дата', yaxis_title='Курс', template='plotly_white')
        with open("forecast.html", "w", encoding='utf-8') as f:
            f.write(pio.to_html(fig4, full_html=False))
        webbrowser.open('file://' + os.path.realpath("forecast.html"))

    def show_news_interface(self):
        """Интерфейс для отображения новостей с новостными карточками."""
        news_win = tk.Toplevel(self.root)
        news_win.title("Новости о валютном рынке")
        news_win.geometry("800x600")
        
        # Создаем прокручиваемый холст
        canvas = tk.Canvas(news_win, background="#ffffff")
        scrollbar = ttk.Scrollbar(news_win, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)
        
        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        articles = fetch_currency_news()
        if not articles:
            return
        
        self.news_images = []  # Чтобы изображения не удалялись сборщиком мусора
        
        for article in articles:
            card = ttk.Frame(scroll_frame, relief="raised", borderwidth=1, padding=10)
            card.pack(padx=10, pady=10, fill="x")
            
            image_url = article.get("urlToImage")
            if image_url:
                try:
                    img_response = requests.get(image_url)
                    img_response.raise_for_status()
                    image_data = img_response.content
                    image = Image.open(io.BytesIO(image_data))
                    image.thumbnail((200, 150))
                    photo = ImageTk.PhotoImage(image)
                    img_label = ttk.Label(card, image=photo)
                    img_label.image = photo
                    img_label.grid(row=0, column=0, rowspan=3, padx=5, pady=5)
                    self.news_images.append(photo)
                except Exception as e:
                    print(f"Ошибка загрузки изображения: {e}")
            
            title = article.get("title", "Без заголовка")
            title_label = ttk.Label(card, text=title, font=("Helvetica", 14, "bold"), wraplength=500)
            title_label.grid(row=0, column=1, sticky="w")
            
            description = article.get("description", "")
            desc_label = ttk.Label(card, text=description, font=("Helvetica", 12), wraplength=500)
            desc_label.grid(row=1, column=1, sticky="w", pady=5)
            
            published_at = article.get("publishedAt", "")
            date_label = ttk.Label(card, text=published_at, font=("Helvetica", 10, "italic"))
            date_label.grid(row=2, column=1, sticky="w")
            
            def open_article(url=article.get("url")):
                if url:
                    webbrowser.open(url)
            open_btn = ttk.Button(card, text="Открыть статью", command=open_article)
            open_btn.grid(row=0, column=2, rowspan=3, padx=10)
            
###############################################################################
# Запуск приложения

if __name__ == "__main__":
    create_users_table()  # Создаем таблицу пользователей (если еще нет)
    
    # Пытаемся загрузить сессионный токен из keyring
    session_token = load_session_token()
    if session_token:
        user = fetch_user_by_token(session_token)
        if user:
            username = user[0]
            main_root = tk.Tk()
            ExchangeRateApp(main_root, username)
            main_root.mainloop()
        else:
            delete_session_token()
            root = tk.Tk()
            AuthWindow(root)
            root.mainloop()
    else:
        root = tk.Tk()
        AuthWindow(root)
        root.mainloop()
