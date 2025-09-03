import time
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import pandas as pd
from PublicDataReader import Kosis
import requests
from typing import Dict, Any, List, Optional
from tqdm.auto import tqdm
tqdm.pandas(desc="processing")   # 한 번만 설정


# KOSIS 공유서비스 Open API 사용자 인증키
service_key = "MmQzZTczMGVhMmViM2FkZWU5NTJkMGY1OWU0MWI5ODY="

# 인스턴스 생성하기
api = Kosis(service_key)

# 환산 매핑
MAP = {
    "격월": "M",
    "반기": "S",
    "분기": "Q",
    "개월": "M",
    "월": "D",
    "일": "D",
    "주간": "W",
    "주": "W",
    "2년": "F",
    "3년": "F",
    "4년": "F",
    "5년": "F",
    "10년": "F",
    "년": "Y",
    "부정기": "IR",
}

def Search():
    KEYWORDS = [
        # 인구
        "인구", "출생", "사망", "혼인", "이혼", "인구이동", "고령", "장래인구",
        # 경제
        "GDP", "국민계정", "고용", "실업", "임금", "산업생산", "소득", "수출입",
        # 물가
        "소비자물가", "CPI", "근원물가", "생활물가", "가격"
    ]
    df = pd.DataFrame()
    for keyword in KEYWORDS:
        df1 = api.get_data(
            "KOSIS통합검색",
            searchNm=keyword,
        )
        # print(df1.head(1))
        df = pd.concat([df, df1])

    print(df.shape)
    print(df)
    df.to_csv('./Search_List.csv', index=False, encoding="utf-8-sig")

def Table_Info():
    df = pd.read_csv('./Search_List.csv')
    table_df = pd.DataFrame()
    for idx, row in df.iterrows():
        print(f"기관ID  {row['기관ID']} / 통계표ID : {row['통계표ID']}")
        df2 = api.get_data(
            "통계표설명",
            "수록정보",
            orgId=row['기관ID'],
            tblId=row['통계표ID'],
            # detail = "Y",
        )
        df2["기관ID"] = row['기관ID']
        df2["통계표ID"] = row['통계표ID']
        table_df = pd.concat([table_df, df2])

    table_df.to_csv('./Table_Info.csv', index=False, encoding="utf-8-sig")
    # target_df = pd.concat([df.reset_index(drop=True), table_df.reset_index(drop=True)], axis=1)

    prio = {'년': 2, '분기': 1, '월': 0}  # 원하는 우선순위로 바꿔도 됨

    right_u = (table_df.assign(_p=table_df['수록주기'].map(prio))
               .sort_values(['통계표ID','_p'])  # 필요하면 ['a','_p','date']
               .drop_duplicates('통계표ID', keep='first')
               .drop(columns='_p'))

    target_df = pd.merge(df, right_u, on="통계표ID", how="left", suffixes=("", "_r"), validate="many_to_one")
    target_df.to_csv('./target.csv', index=False, encoding="utf-8-sig")


def norm_unit(x):
    if x is None:
        return None
    s = str(x)
    for k in sorted(MAP, key=len, reverse=True):  # 긴 키부터 치환 (예: '격월'이 '월'보다 먼저)
        s = s.replace(k, MAP[k])
    return " ".join(s.split())  # 공백 정리

import re, math

def drop_decimal(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return int(math.trunc(x))
    s = str(x)
    return re.sub(r'(-?\d+)\.\d+', r'\1', s)  # "1.9년" -> "1년", "3.00" -> "3"

def to_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows if isinstance(rows, list) else [rows])

    # 일반적으로 KOSIS 반환 컬럼 예시: "PRD_DE", "C1_NM", "ITM_NM", "DT" ...
    # 숫자 변환
    for c in ("DT", "dt", "DATA_VALUE"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # 중복 제거
    df = df.drop_duplicates()
    return df

def make_session():
    s = requests.Session()
    retry = Retry(
        total=5, connect=5, read=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET","POST","PUT","DELETE","HEAD","OPTIONS"])
    )
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=100, pool_maxsize=100))
    s.mount("http://",  HTTPAdapter(max_retries=retry, pool_connections=100, pool_maxsize=100))

    adapter = HTTPAdapter(max_retries=retry, pool_connections=100, pool_maxsize=100)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    # 오래된 keep-alive 커넥션으로 끊기는 문제 회피
    s.headers["Connection"] = "close"
    return s

def Get_Data():
    S = make_session()
    try:
        df = pd.read_csv('./target.csv')
        result_df = pd.DataFrame()

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="rows"):
            API_URL = f"https://kosis.kr/openapi/Param/statisticsParameterData.do?method=getList&apiKey={service_key}" \
                      f"&itmId=ALL&objL1=ALL" \
                      f"&objL2=ALL&objL3=&objL4=&objL5=&objL6=&objL7=&objL8=" \
                      f"&format=json&jsonVD=Y" \
                      f"&prdSe={norm_unit(row['수록주기'])}&startPrdDe={drop_decimal(row['수록기간시작일'])}&endPrdDe={drop_decimal(row['수록기간종료일'])}" \
                      f"&orgId={drop_decimal(row['기관ID'])}&tblId={row['통계표ID']}"
            # print(API_URL)
            resp = S.get(url=API_URL, timeout=(10, 60))
            resp.raise_for_status()
            data = resp.json()
            # print(data)
            if data is None:
                print(API_URL)
            else:
                tmp_df = to_dataframe(data)
                result_df = pd.concat([result_df, tmp_df])
                result_df.to_csv('./result_df.csv', index=False, encoding="utf-8-sig")
                time.sleep(0.5)
    except (requests.ConnectionError, requests.ReadTimeout, requests.ChunkedEncodingError):
        # 세션 새로 열고 1회만 재시도 (RemoteDisconnected 포함)
        S.close()
        S = make_session()

    except Exception as ex:
        print(f"Get_Data : {ex}")


Search()
Table_Info()
Get_Data()




