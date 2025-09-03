import requests
import pandas as pd

API_KEY = 'GRJT9K2CK6DSBGL7R7EH'
url = f'https://ecos.bok.or.kr/api/StatisticTableList/{API_KEY}/json/kr/1/10'

response = requests.get(url)
data = response.json()

# 결과 출력
print(data)

# 원하는 데이터 추출
for item in data['StatisticTableList']['row']:
    print(item)

df = pd.DataFrame(data['StatisticTableList']['row'])
print(df)
