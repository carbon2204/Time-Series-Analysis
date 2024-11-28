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
from db import fetch_exchange_rate, insert_exchange_rate


logging.basicConfig(level=logging.ERROR, stream=sys.stdout)
logger = logging.getLogger()

API_URL = "https://www.cbr-xml-daily.ru"

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
    """
    Возвращает временной ряд курса валюты `from_currency` к валюте `to_currency`
    за указанный период и сохраняет данные в базу, если их нет.
    """
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
    """Получает курс валюты к рублю. Сначала проверяет базу данных, затем API."""
    stored_rate = fetch_exchange_rate(currency_code, date)
    if stored_rate:
        print(f"Курс из базы данных для {currency_code} на {date}: {stored_rate}")
        return stored_rate

    formatted_date = date.replace('-', '/')
    print(f"Отправляем запрос к API для {currency_code} на дату {formatted_date}")

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
            print(f"{currency_code} не найден в данных API на {formatted_date}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка получения курса из API: {e}")
        return None



def get_exchange_rate_between_currencies(from_currency, to_currency, date):
    """Получает курс валюты `from_currency` по отношению к `to_currency` на указанную дату."""
    rate_from = get_exchange_rate_from_rub(from_currency, date)
    rate_to = get_exchange_rate_from_rub(to_currency, date)
    if rate_from is not None and rate_to is not None:
        return rate_from / rate_to
    return None



def get_exchange_rate_to_all(from_currency, date):
    """Получает курсы валюты `from_currency` по отношению к другим валютам на указанную дату."""
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
        logger.error(f"Error fetching exchange rates to all currencies: {e}")
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
        messagebox.showwarning("Нет данных", "Не удалось получить данные за указанный период.")
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

from statsmodels.tsa.holtwinters import ExponentialSmoothing

def forecast_with_holt_winters(time_series, forecast_days=10):
    df = pd.Series(time_series)
    df.index = pd.to_datetime(df.index)
    df = df.asfreq('D')
    df = pd.to_numeric(df, errors='coerce').interpolate(method='time')  
    try:
        model = ExponentialSmoothing(df, trend="add", seasonal="add", seasonal_periods=3)
        model_fit = model.fit()
    except Exception as e:
        logger.error(f"Error fitting Holt-Winters model: {e}")
        messagebox.showerror("Ошибка", "Не удалось построить модель прогнозирования.")
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
    print("future_start_date:", future_start_date)

    try:
        model = model = auto_arima(train_data, seasonal=True, m=7, trace=False, error_action='ignore', suppress_warnings=True)
        model_fit = model.fit(train_data)
    except Exception as e:
        logger.error(f"Error fitting ARIMA model: {e}")
        messagebox.showerror("Ошибка", "Не удалось построить модель ARIMA.")
        return None, None, None, None, None, None

    
    in_sample_forecast = model_fit.predict(n_periods=len(test_data))
    in_sample_forecast = pd.Series(in_sample_forecast, index=test_data.index)

    
    out_sample_forecast = model_fit.predict(n_periods=forecast_days)
    future_dates = pd.date_range(start=future_start_date, periods=forecast_days, freq='D')
    out_sample_forecast = pd.Series(out_sample_forecast, index=future_dates)

    print("out_sample_forecast index:", out_sample_forecast.index)
    print("out_sample_forecast values:", out_sample_forecast.head())

    
    mae = mean_absolute_error(test_data, in_sample_forecast)
    rmse = np.sqrt(mean_squared_error(test_data, in_sample_forecast))

    return train_data, test_data, in_sample_forecast, out_sample_forecast, mae, rmse


class ExchangeRateApp:
    def __init__(self, root):
        self.root = root
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

    def create_menu(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        frame = ttk.Frame(self.root)
        frame.pack(pady=20, expand=True)

        ttk.Label(frame, text="Главное меню", font=("Arial", 16)).grid(row=0, column=0, columnspan=2, pady=10)

        ttk.Button(frame, text="1. Курс валюты к другой валюте", command=self.show_exchange_rate_interface).grid(row=1, column=0, padx=5, pady=5, sticky='ew')
        ttk.Button(frame, text="2. Курс валюты ко всем валютам", command=self.show_all_rates_interface).grid(row=2, column=0, padx=5, pady=5, sticky='ew')
        ttk.Button(frame, text="3. Курс валюты в промежуток времени", command=self.show_time_series_interface).grid(row=3, column=0, padx=5, pady=5, sticky='ew')
        ttk.Button(frame, text="0. Выход", command=self.root.quit).grid(row=4, column=0, padx=5, pady=5, sticky='ew')

        frame.columnconfigure(0, weight=1)

    def show_exchange_rate_interface(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        frame = ttk.Frame(self.root)
        frame.pack(pady=20, expand=True)

        ttk.Label(frame, text="Курс валюты к другой валюте", font=("Arial", 14)).grid(row=0, column=0, columnspan=2, pady=10)

        ttk.Label(frame, text="Выберите исходную валюту:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        from_currency = ttk.Combobox(frame, values=self.currency_codes)
        from_currency.grid(row=1, column=1, padx=5, pady=5, sticky='w')

        ttk.Label(frame, text="Выберите целевую валюту:").grid(row=2, column=0, padx=5, pady=5, sticky='e')
        to_currency = ttk.Combobox(frame, values=self.currency_codes)
        to_currency.grid(row=2, column=1, padx=5, pady=5, sticky='w')

        ttk.Label(frame, text="Выберите дату:").grid(row=3, column=0, padx=5, pady=5, sticky='e')
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
                    result_label.config(text="Не удалось получить курс на указанную дату.")
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

        ttk.Label(frame, text="Выберите исходную валюту:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        from_currency = ttk.Combobox(frame, values=self.currency_codes)
        from_currency.grid(row=1, column=1, padx=5, pady=5, sticky='w')

        ttk.Label(frame, text="Выберите дату:").grid(row=2, column=0, padx=5, pady=5, sticky='e')
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
                    result_text.insert(tk.END, "Не удалось получить курсы на указанную дату.")
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

        ttk.Label(frame, text="Выберите исходную валюту:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        from_currency = ttk.Combobox(frame, values=self.currency_codes)
        from_currency.grid(row=1, column=1, padx=5, pady=5, sticky='w')

        ttk.Label(frame, text="Выберите целевую валюту:").grid(row=2, column=0, padx=5, pady=5, sticky='e')
        to_currency = ttk.Combobox(frame, values=self.currency_codes)
        to_currency.grid(row=2, column=1, padx=5, pady=5, sticky='w')

        ttk.Label(frame, text="Выберите начальную дату:").grid(row=3, column=0, padx=5, pady=5, sticky='e')
        start_date_entry = DateEntry(frame, date_pattern='yyyy-mm-dd')
        start_date_entry.grid(row=3, column=1, padx=5, pady=5, sticky='w')

        ttk.Label(frame, text="Выберите конечную дату:").grid(row=4, column=0, padx=5, pady=5, sticky='e')
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
                    messagebox.showwarning("Нет данных", "Не удалось получить данные за указанный период.")
            else:
                messagebox.showwarning("Недостаточно данных", "Заполните все поля.")

        ttk.Button(frame, text="Получить временной ряд", command=get_time_series_data).grid(row=5, column=0, columnspan=2, pady=10)
        ttk.Button(frame, text="Назад", command=self.create_menu).grid(row=6, column=0, columnspan=2, pady=10)

        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)


    def show_analysis_tabs(self, time_series, analysis):
        for widget in self.root.winfo_children():
            widget.destroy()

        notebook = ttk.Notebook(self.root)
        notebook.pack(expand=1, fill='both')

        
        tab1 = ttk.Frame(notebook)
        notebook.add(tab1, text="Исходный график")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=list(time_series.keys()), y=list(time_series.values()), mode='lines', name='Исходные данные', line=dict(color='blue')))
        fig1.update_layout(title='Исходный график', xaxis_title='Дата', yaxis_title='Курс', template='plotly_white')
        with open("original_data.html", "w", encoding='utf-8') as f:
            f.write(pio.to_html(fig1, full_html=False))
        webbrowser.open('file://' + os.path.realpath("original_data.html"))

       
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

        
        tab3 = ttk.Frame(notebook)
        notebook.add(tab3, text="Вывод анализа")
        analysis_text = tk.Text(tab3, wrap="word")
        analysis_text.insert("1.0", f"Стационарность (p-value): {analysis['stationarity_p_value']:.5f}\n")
        analysis_text.insert("end", f"Ряд стационарен: {'Да' if analysis['is_stationary'] else 'Нет'}\n")
        analysis_text.insert("end", "Тренд:\n" + analysis["trend"].head().to_string() + "\n")
        analysis_text.insert("end", "Сезонность:\n" + analysis["seasonal"].head().to_string() + "\n")
        analysis_text.insert("end", "Шум:\n" + analysis["residual"].head().to_string() + "\n")
        analysis_text.pack(expand=1, fill="both")

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


if __name__ == "__main__":
    root = tk.Tk()
    app = ExchangeRateApp(root)
    root.mainloop()