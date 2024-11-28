import requests
from datetime import datetime
from db.database import insert_exchange_rate

API_URL = "https://www.cbr-xml-daily.ru"

def update_daily_rates():
    try:
        response = requests.get(f"{API_URL}/daily_json.js")
        response.raise_for_status()
        data = response.json()
        rates = data.get('Valute', {})
        
        # Use `fromisoformat` to handle the timezone offset in the date string
        date = datetime.fromisoformat(data['Date']).date()

        for code, details in rates.items():
            rate_to_rub = details['Value'] / details['Nominal']
            insert_exchange_rate(code, rate_to_rub, date)

        print(f"Курсы валют обновлены на {date}.")
    except Exception as e:
        print(f"Ошибка обновления курсов валют: {e}")

if __name__ == "__main__":
    update_daily_rates()
