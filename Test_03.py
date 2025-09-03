import requests
import pandas as pd
import time

API_KEY = 'GRJT9K2CK6DSBGL7R7EH'

# 통계코드 전체 수집 (최대 1000개)
stat_url = f'https://ecos.bok.or.kr/api/StatisticTableList/{API_KEY}/json/kr/1/1000/'
stat_resp = requests.get(stat_url)
stat_data = stat_resp.json()
if 'StatisticTableList' in stat_data and 'row' in stat_data['StatisticTableList']:
    stat_rows = stat_data['StatisticTableList']['row']
    df_stat = pd.DataFrame(stat_rows)
    stat_codes = df_stat['STAT_CODE'].tolist()
else:
    print('통계코드 수집 실패:', stat_data)
    stat_codes = []

def get_date_format(cycle, start_ymd, end_ymd):
    start_y, start_m = start_ymd[:4], start_ymd[4:6]
    end_y, end_m = end_ymd[:4], end_ymd[4:6]
    if cycle == 'A' or cycle == 'Y':
        return start_y, end_y
    elif cycle == 'S':
        return f"{start_y}S1", f"{end_y}S2"
    elif cycle == 'Q':
        s_q = (int(start_m)-1)//3 + 1
        e_q = (int(end_m)-1)//3 + 1
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

def get_all_ecos_rows(api_key, stat_code, cycle, item_code, start, end, max_rows=10000, page_size=1000):
    all_rows = []
    for start_row in range(1, max_rows+1, page_size):
        end_row = min(start_row + page_size - 1, max_rows)
        url = f"https://ecos.bok.or.kr/api/StatisticSearch/{api_key}/json/kr/{start_row}/{end_row}/{stat_code}/{cycle}/{start}/{end}/{item_code}"
        data_resp = requests.get(url)
        data = data_resp.json()
        data_rows = data.get('StatisticSearch', {}).get('row', [])
        if not data_rows:
            break
        all_rows.extend(data_rows)
        if len(data_rows) < page_size:
            break
        time.sleep(1)
    return all_rows

start_ymd = '20000101'
end_ymd = '20241231'
called_urls = set()

for stat_code in stat_codes:
    stat_rows = []
    item_url = f'https://ecos.bok.or.kr/api/StatisticItemList/{API_KEY}/json/kr/1/1000/{stat_code}/'
    item_resp = requests.get(item_url)
    time.sleep(1)
    item_data = item_resp.json()
    if 'StatisticItemList' not in item_data or 'row' not in item_data['StatisticItemList']:
        print(f"[{stat_code}] 항목코드 요청 실패! 응답: {item_data}")
        continue
    item_rows = item_data['StatisticItemList']['row']
    if isinstance(item_rows, dict):
        item_rows = [item_rows]
    for item in item_rows:
        item_code = item['ITEM_CODE']
        cycle_list = item.get('CYCLE', '').split(',')
        cycle_list = [c.strip() for c in cycle_list if c.strip()]
        found = False
        for cycle in cycle_priority:
            if cycle in cycle_list:
                time.sleep(1)
                try:
                    start, end = get_date_format(cycle, start_ymd, end_ymd)
                except Exception:
                    continue
                data_rows = get_all_ecos_rows(API_KEY, stat_code, cycle, item_code, start, end)
                if data_rows:
                    for row in data_rows:
                        stat_rows.append(row)
                    found = True
                    print(f"[OK] {stat_code}:{item_code} {cycle} {start}~{end} ({len(data_rows)}건)")
                    break
        if not found:
            print(f"[SKIP] {stat_code}:{item_code} (지원주기 내 데이터 없음)")

    # [★] 통계코드별 csv 저장
    if stat_rows:
        df_this = pd.DataFrame(stat_rows)
        df_this.to_csv(f"./Result/ecos_{stat_code}.csv", index=False, encoding='utf-8-sig')
        # print(f">>> [저장 완료] ecos_{stat_code}.csv (row: {len(df_this)})")
    else:
        print(f">>> [SKIP] {stat_code}: 수집된 데이터 없음")
