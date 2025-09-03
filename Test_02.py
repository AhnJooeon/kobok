import time

import requests
import pandas as pd
from functools import reduce

API_KEY = 'GRJT9K2CK6DSBGL7R7EH'
# stat_codes = ['722Y001', '901Y001', '731Y001']

# 한 번에 1000개까지 가능, 800개 정도이므로 충분
url = f'https://ecos.bok.or.kr/api/StatisticTableList/{API_KEY}/json/kr/1/1000/'

resp = requests.get(url)
data = resp.json()

# 방어적으로 처리
if 'StatisticTableList' in data and 'row' in data['StatisticTableList']:
    stat_rows = data['StatisticTableList']['row']
    df_stat = pd.DataFrame(stat_rows)
    print(df_stat[['STAT_CODE', 'STAT_NAME', 'CYCLE', 'ORG_NAME']].head())
    # 전체 코드 리스트 추출
    stat_codes = df_stat['STAT_CODE'].tolist()
else:
    print("통계표 리스트 가져오기 실패:", data)
    stat_codes = []

start_ymd = '20220101'
end_ymd = '20241231'


def get_date_format(cycle, start_ymd, end_ymd):
    start_y, start_m, start_d = start_ymd[:4], start_ymd[4:6], start_ymd[6:]
    end_y, end_m, end_d = end_ymd[:4], end_ymd[4:6], end_ymd[6:]
    if cycle == 'A' or cycle == 'Y':
        return start_y, end_y
    elif cycle == 'S':
        return f"{start_y}S1", f"{end_y}S2"
    elif cycle == 'Q':
        s_q = (int(start_m) - 1) // 3 + 1
        e_q = (int(end_m) - 1) // 3 + 1
        return f"{start_y}Q{s_q}", f"{end_y}Q{e_q}"
    elif cycle == 'M':
        return f"{start_y}{start_m}", f"{end_y}{end_m}"
    elif cycle == 'SM':
        return f"{start_y}{start_m}1", f"{end_y}{end_m}2"
    elif cycle == 'D':
        return start_ymd, end_ymd
    else:
        raise ValueError('지원하지 않는 주기 코드입니다.')


cycle_priority = ['M', 'Q', 'A', 'Y', 'S', 'SM', 'D']

all_dfs = []
all_rows = []
called_urls = set()

for stat_code in stat_codes:
    # 1. 항목코드 전체 조회
    item_url = f'https://ecos.bok.or.kr/api/StatisticItemList/{API_KEY}/json/kr/1/1000/{stat_code}/'
    item_resp = requests.get(item_url)
    item_data = item_resp.json()
    if 'StatisticItemList' not in item_data or 'row' not in item_data['StatisticItemList']:
        print(f"[{stat_code}] 항목코드 요청 실패! 응답: {item_data}")
        continue
    item_rows = item_data['StatisticItemList']['row']
    if isinstance(item_rows, dict):
        item_rows = [item_rows]

    for item in item_rows:
        time.sleep(1)
        item_code = item['ITEM_CODE']
        item_name = item['ITEM_NAME'].replace(" ", "_")
        cycle_list = item.get('CYCLE', '').split(',')
        cycle_list = [c.strip() for c in cycle_list if c.strip()]
        found = False
        # 실제 데이터가 나오는 주기부터 채택(M→Q→A/Y→S→SM→D)
        for cycle in cycle_priority:
            if cycle in cycle_list:
                try:
                    start, end = get_date_format(cycle, start_ymd, end_ymd)
                except Exception as e:
                    print(f"[{stat_code}:{item_code}] 날짜포맷 실패: {e}")
                    continue
                url = f"https://ecos.bok.or.kr/api/StatisticSearch/{API_KEY}/json/kr/1/1000/{stat_code}/{cycle}/{start}/{end}/"
                if url in called_urls:
                    # 이미 동일 요청 했으면 skip
                    found = True
                    break
                print(f"url : {url}")
                called_urls.add(url)
                data_resp = requests.get(url)
                time.sleep(1)
                data = data_resp.json()
                data_rows = data.get('StatisticSearch', {}).get('row', [])
                if data_rows:
                    for row in data_rows:
                        all_rows.append(row)  # "row"의 모든 key/value 저장!
                    found = True
                    print(f"[OK] {stat_code}:{item_code} {cycle} {start}-{end}")
                    break
                else:
                    if 'RESULT' in data:
                        print(f"[에러] {stat_code}:{item_code} {cycle} {start}~{end} | {data['RESULT']}")
                    else:
                        print(f"[노데이터] {stat_code}:{item_code} {cycle} {start}~{end}")
                time.sleep(0.05)  # 과호출 방지
                if not found:
                    print(f"[SKIP] {stat_code}:{item_code} (지원주기 내 데이터 없음)")

            # ③ long포맷 DataFrame으로 정리
            df_long = pd.DataFrame(all_rows)
            # print("Long format (row 일부):")
            # print(df_long.head())
            df_long.to_csv('./bok_tmp.csv', encoding='utf-8-sig', index=False)

            # 필요시 wide로 피벗 (예: 항목명 기준)
            # df_wide = df_long.pivot_table(index="TIME", columns="ITEM_NAME", values="VALUE", aggfunc='first')
            # print("Wide format (row 일부):")
            # print(df_wide.head())