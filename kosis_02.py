#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 실행: .env 에 KOSIS_API_KEY=... 넣고 `python kosis_probe_and_fetch.py`
# 동작: TARGETS 표들 대상으로 → 주기 자동감지 → 되는 조합만 newEstPrdCnt로 프리플라이트
#       → 확정된 조합으로 2000~현재까지 CSV 저장. 모든 요청 URL 출력.

import os, re, json, time, pathlib, requests
from datetime import datetime
import pandas as pd

# ===== 사용자 설정 =====
TARGETS = [
    ("101", "DT_1BPA403"),  # 네가 올린 실패 URL의 표, 예시로 포함
    # ("기관ID","통계표ID") 원하는 만큼 추가
]
FROM_YEAR  = 2000
RECENT_LAG = 1
PAUSE_S    = 0.2

# ===== 키 로딩 (.env/환경변수) =====
try:
    from dotenv import load_dotenv, find_dotenv
    p = find_dotenv(usecwd=True); load_dotenv(p or None)
except Exception:
    pass
API_KEY = (os.getenv("KOSIS_API_KEY") or os.getenv("KOSIS_SERVICE_KEY") or "").strip()
if not API_KEY:
    raise SystemExit("KOSIS_API_KEY가 없습니다. .env/환경변수 확인하세요.")

# ===== 공통 =====
OUT = pathlib.Path("data"); OUT.mkdir(exist_ok=True)
API_URL = "https://kosis.kr/openapi/Param/statisticsParameterData.do"
S = requests.Session(); S.headers.update({"User-Agent":"kosis-probe/1.3"})

def sanitize(d): return {k:v for k,v in d.items() if v not in (None,"")}

def build_url(p):
    base = {"method":"getList","apiKey":API_KEY,"format":"json"}
    req = requests.Request("GET", API_URL, params={**base, **sanitize(p)}).prepare()
    return req.url

def _lenient_json(text: str):
    try: return json.loads(text)
    except Exception: pass
    t = re.sub(r'([{,\s])([A-Za-z_][A-Za-z0-9_]*)(\s*:)', r'\1"\2"\3', text.strip())
    t = re.sub(r',\s*([}\]])', r'\1', t)
    try: return json.loads(t)
    except Exception:
        import ast
        t = t.replace("null","None").replace("true","True").replace("false","False")
        return ast.literal_eval(t)

def call(p):
    url = build_url(p)
    print("[URL]", url)
    time.sleep(PAUSE_S)
    r = S.get(url, timeout=45)
    try: data = r.json()
    except Exception: data = _lenient_json(r.text)
    # 에러면 원문 반환
    if isinstance(data, dict) and (data.get("err") or data.get("errCd")):
        return None, data
    rows = None
    if isinstance(data, list): rows = data
    elif isinstance(data, dict):
        for k in ("list","STAT_DATA","data","rows","result","STAT_LIST"):
            if isinstance(data.get(k), list): rows = data[k]; break
    return rows, None

# ===== 시점 유틸 =====
def recent(prd):
    now = datetime.now(); p = (prd or "").upper()
    if p in ("A","Y"):
        y = now.year - RECENT_LAG; return f"{y:04d}", f"{y:04d}"
    if p == "M":
        y, m = now.year, now.month - RECENT_LAG
        while m <= 0: m += 12; y -= 1
        return f"{y:04d}{m:02d}", f"{y:04d}{m:02d}"
    if p == "Q":
        y, q = now.year, (now.month-1)//3 + 1 - RECENT_LAG
        while q <= 0: q += 4; y -= 1
        return f"{y:04d}{q:02d}", f"{y:04d}{q:02d}"
    if p == "H":
        y, h = now.year, (1 if now.month<=6 else 2) - RECENT_LAG
        while h <= 0: h += 2; y -= 1
        return f"{y:04d}{h:02d}", f"{y:04d}{h:02d}"

def year_windows(prd):
    now = datetime.now(); p = (prd or "").upper()
    for y in range(FROM_YEAR, now.year+1):
        if p == "M":
            y2, endm = y, (12 if y<now.year else now.month-RECENT_LAG)
            if endm <= 0: endm, y2 = 12, y-1
            yield f"{y:04d}01", f"{y2:04d}{endm:02d}"
        elif p == "Q":
            endq = 4 if y<now.year else ((now.month-1)//3 + 1 - RECENT_LAG)
            while endq <= 0: endq += 4
            yield f"{y:04d}01", f"{y:04d}{endq:02d}"
        elif p == "H":
            endh = 2 if y<now.year else ((1 if now.month<=6 else 2) - RECENT_LAG)
            while endh <= 0: endh += 2
            yield f"{y:04d}01", f"{y:04d}{endh:02d}"
        elif p in ("Y","A"):
            yield f"{y:04d}", f"{y:04d}"

# ===== statHtml: 완전 경로 + itm (교차곱 금지) =====
def _lenient_obj(s: str):
    try: return json.loads(s)
    except Exception:
        t = re.sub(r'([{,\s])([A-Za-z_][A-Za-z0-9_]*)(\s*:)', r'\1"\2"\3', s.strip())
        t = re.sub(r',\s*([}\]])', r'\1', t)
        try: return json.loads(t)
        except Exception:
            import ast
            t = t.replace("null","None").replace("true","True").replace("false","False")
            return ast.literal_eval(t)

def scrape_paths_itms(org, tbl):
    html = S.get("https://kosis.kr/statHtml/statHtml.do", params={"orgId":org,"tblId":tbl}, timeout=30).text
    itms = list(dict.fromkeys(re.findall(r'ITM_ID"\s*:\s*"([^"]+)"', html)))
    itms = [c for c in itms if c and c.upper() not in ("ALL","T")]
    # 객체 단위로 경로 파싱
    paths = []
    for m in re.finditer(r'\{[^{}]{0,2000}"C1"\s*:\s*".+?"[^{}]*\}', html):
        obj = _lenient_obj(m.group(0))
        if not isinstance(obj, dict): continue
        path = {}; contig=True; seen_empty=False
        for lv in range(1,9):
            ck=f"C{lv}"; cv=str(obj.get(ck,"") or "").strip()
            if cv:
                if seen_empty: contig=False; break
                path[f"objL{lv}"]=cv
            else:
                seen_empty=True
        if contig and path:
            paths.append(path)
    # 중복 제거
    uniq, seen = [], set()
    for p in paths:
        key = tuple((k,p[k]) for k in sorted(p.keys()))
        if key not in seen:
            seen.add(key); uniq.append(p)
    return itms, uniq

# ===== 주기 자동 감지 =====
def detect_prdSe(org, tbl):
    html = S.get("https://kosis.kr/statHtml/statHtml.do", params={"orgId":org,"tblId":tbl}, timeout=30).text
    found = set(re.findall(r'"PRD_SE"\s*:\s*"([A-Z]+)"', html, flags=re.I))
    found |= set(re.findall(r'prdSe"\s*:\s*"([A-Z]+)"', html, flags=re.I))
    return [x for x in ["M","Q","Y","H","IR"] if x in found] or ["M","Q","Y","H","IR"]

# ===== 프리플라이트 (실패 조합은 버리고 계속) =====
def find_base(org, tbl):
    itms, paths = scrape_paths_itms(org, tbl)
    has_itm, has_path = bool(itms), bool(paths)
    itms  = (itms[:5]  if has_itm  else [None])
    paths = (paths[:5] if has_path else [{}])

    # 빈 조합 금지: 뭔가라도 있으면 최소 하나는 포함
    combos=[]
    for P in paths:
        for I in itms:
            combos.append((P,I))      # path+itm
            combos.append((P,None))   # path만
    if has_itm and not has_path:
        for I in itms: combos.append(({},I))
    if not has_itm and not has_path:
        combos.append(({},None))      # 진짜 아무것도 없을 때만

    prds = detect_prdSe(org, tbl)

    # 1단계: 최신 N건으로 빠르게 ‘되는’ 조합 찾기
    for prd in prds:
        for P,I in combos:
            p = dict(orgId=org, tblId=tbl, prdSe=prd, newEstPrdCnt=2)
            if I: p["itmId"]=I
            for k,v in P.items(): p[k]=v
            rows, err = call(p)
            if err or not rows: continue
            base = dict(orgId=org, tblId=tbl, prdSe=prd)
            if I: base["itmId"]=I
            for k,v in P.items(): base[k]=v
            print(f"[PROBE OK] {org}/{tbl} prd={prd} rows={len(rows)}")
            return base

    # 최후수단: 플레이스홀더로 한 번만 더
    placeholders = [{"objL1":"ALL"},{"objL1":"00"},{"objL1":"0"},{"objL1":"TOT"},{"objL1":"T"}]
    for prd in prds:
        for P in placeholders:
            p = dict(orgId=org, tblId=tbl, prdSe=prd, newEstPrdCnt=2, **P)
            rows, err = call(p)
            if err or not rows: continue
            base = dict(orgId=org, tblId=tbl, prdSe=prd, **P)
            print(f"[PROBE OK*] {org}/{tbl} prd={prd} rows={len(rows)} (placeholder)")
            return base

    return None

# ===== 수집 (2000~현재) =====
def fetch_all(base):
    org, tbl, prd = base["orgId"], base["tblId"], base["prdSe"]
    out_dir = OUT / f"{org}_{tbl}"; out_dir.mkdir(parents=True, exist_ok=True)

    if prd == "IR":
        # 불규칙은 건수로 한 방에
        rows, err = call({**base, "newEstPrdCnt": 9999})
        if rows:
            pd.DataFrame(rows).to_csv(out_dir/f"{tbl}_IR_all.csv", index=False, encoding="utf-8-sig")
            print(f"[SAVE] {out_dir}/{tbl}_IR_all.csv rows={len(rows)}")
        else:
            print("[WARN] IR 응답 없음", err or "")
        return

    saved = 0
    for s,e in year_windows(prd):
        rows, err = call({**base, "startPrdDe": s, "endPrdDe": e})
        if err:
            # 파라미터 불일치면 그 연도만 패스
            continue
        if not rows:
            continue
        df = pd.DataFrame(rows)
        fp = out_dir / f"{tbl}_{prd}_{s}_{e}.csv"
        df.to_csv(fp, index=False, encoding="utf-8-sig")
        saved += 1
        print(f"[SAVE] {fp} rows={len(df)}")
    if saved == 0:
        print("[WARN] 저장된 CSV가 없습니다.")

# ===== main =====
def main():
    for org, tbl in TARGETS:
        print(f"\n[START] {org}/{tbl}")
        base = find_base(org, tbl)
        if not base:
            print(f"  -> FAIL: {org}/{tbl} 유효 조합 없음 (프리플라이트 실패)"); continue
        print("[BASE]", base)
        print("[FETCH] 2000년부터 수집…")
        fetch_all(base)
    print("\n[DONE] data/ 폴더 확인")

if __name__ == "__main__":
    main()
