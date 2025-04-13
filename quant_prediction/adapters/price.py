"""
pip install httpx selenium webdriver-manager
"""
import datetime
import httpx
import json
import time

# valid ranges: "1d", "5d", "1mo", "3mo", "6mo","1y","2y","5y","10y","ytd","max"
ts_start = int(
    time.mktime(time.strptime((datetime.datetime.now() - datetime.timedelta(days=3)).strftime("%Y-%m-%d"), "%Y-%m-%d")))
ts_end = int(time.mktime(time.strptime(datetime.datetime.now().strftime("%Y-%m-%d"), "%Y-%m-%d")))
interval = "1d"


def parse_to_time_based_json(data):
    """
    Parse the Yahoo Finance data to a time-based JSON format.
    :param data: The data returned from Yahoo Finance API.
    :return: A dictionary containing the parsed data.
    """
    if not data or "chart" not in data or not data["chart"]["result"]:
        print("Empty data or invalid format")
        return None

    result = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    quotes = result["indicators"]["quote"][0]

    keys = ['open', 'close', 'high', 'low', 'volume']

    parsed_data = []
    for i, timestamp in enumerate(timestamps):
        readable_time = datetime.datetime.fromtimestamp(timestamp).strftime('%m/%d %I:%M %p')
        point_data = {"Date": readable_time}
        for key in keys:
            if key in quotes and i < len(quotes[key]):
                point_data[key.capitalize()] = quotes[key][i]
            else:
                point_data[key.capitalize()] = None
        parsed_data.append(point_data)

    final_json = {
        "chart": {
            "result": [{
                "meta": result.get("meta", {}),
                "indicators": {
                    "quote": parsed_data
                }
            }],
            "error": None
        }
    }

    return final_json


# construct url 构造url
url = f"https://query1.finance.yahoo.com/v8/finance/chart/SOL-USD?period1={ts_start}&period2={ts_end}&interval={interval}&includePrePost=true&events=div%7Csplit%7Cearn&lang=en-GB&region=GB&source=cosaic"

# construct headers 构造请求头
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    'Accept': '*/*',
    'Accept-Encoding': 'gzip, deflate, br, zstd',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,zh-TW;q=0.7',
    'Cache-Control': 'no-cache',
    'Origin': 'https://uk.finance.yahoo.com',
    'Pragma': 'no-cache',
    'Referer': 'https://uk.finance.yahoo.com/quote/SOL-USD/?guccounter=1',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Cookie': 'A1=d=AQABBIibZmcCEH0D9yCZhtWZEnYY5E_ZBewFEgABCAF0-2cqaPZ0rXYBAiAAAAcIiJtmZ0_ZBew&S=AQAAArG3jeUCh81GACImnGUDqJ4;A3=d=AQABBIibZmcCEH0D9yCZhtWZEnYY5E_ZBewFEgABCAF0-2cqaPZ0rXYBAiAAAAcIiJtmZ0_ZBew&S=AQAAArG3jeUCh81GACImnGUDqJ4;cmp=t=1744448342&j=1&u=1---&v=75;GUC=AQABCAFn-3RoKkIhQASk&s=AQAAAFbPSKmh&g=Z_orXg;PRF=t%3DSOL-USD;A1S=d=AQABBIibZmcCEH0D9yCZhtWZEnYY5E_ZBewFEgABCAF0-2cqaPZ0rXYBAiAAAAcIiJtmZ0_ZBew&S=AQAAArG3jeUCh81GACImnGUDqJ4;dflow=587;EuConsent=CQPwMIAQPwMIAAOACKENBkFgAAAAAAAAACiQAAAAAAAA;GUCS=AVuhB2Yf',
}


# do the request
def run(*args, **kwargs):
    """
    run the request and parse the data
    """
    if args:
        pass
    try:
        response = httpx.get(url, headers=headers)
        response.raise_for_status()  # Raise an error for bad responses
        data = response.json()  # Parse the JSON response
        parsed_data = parse_to_time_based_json(data)
        if parsed_data:
            print(json.dumps(parsed_data, indent=2))  # Pretty print the parsed data
            return parsed_data
        else:
            raise ValueError("Parsed data is empty or None")
    except:
        raise ValueError("Parsed data is empty or None")


if __name__ == "__main__":
    run()
