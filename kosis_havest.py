# kosis_autoharvest.py — no fixed table IDs (catalog → discover → harvest)
import os, re, json, csv, time, pathlib, requests

# .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ==== 기본 ====
API_KEY = os.getenv("KOSIS_API_KEY", "").strip()
if not API_KEY:
    raise SystemExit("ERROR: KOSIS_API_KEY가 없습니다. .env 또는 환경변수로 설정하세요.")

BASE = pathlib.Path(__file__).resolve().parent
OUT  = BASE / "data"
OUT.mkdir(parents=True, exist_ok=True)

DEBUG = False
SLEEP = 0.25
S = requests.Session()
S.headers.update({"User-Agent": "kosis-auto/0.4"})

KEYWORDS = [
    # 인구
    "인구", "출생", "사망", "혼인", "이혼", "인구이동", "고령", "장래인구",
    # 경제
    "GDP", "국민계정", "고용", "실업", "임금", "산업생산", "소득", "수출입",
    # 물가
    "소비자물가", "CPI", "근원물가", "생활물가", "가격"
]

# ==== 관대한 JSON 파서 (KOSIS가 종종 준-JSON을 줌) ====
KEY = re.compile(r'([{,\s])([A-Za-z_][A-Za-z0-9_]*)(\s*:)')
TRAIL = re.compile(r',\s*([}\]])')
def parse_lenient(text):
    try:
        return json.loads(text)
    except Exception:
        pass
    t = text.strip()
    if (t.startswith("'") and t.endswith("'")) or (t.startswith('"') and t.endswith('"')):
        t = t[1:-1]
    t = KEY.sub(r'\1"\2"\3', t)
    t = TRAIL.sub(r'\1', t)
    try:
        return json.loads(t)
    except Exception:
        import ast
        t = t.replace("null","None").replace("true","True").replace("false","False")
        return ast.literal_eval(t)

def api_get(url, params, timeout=30):
    time.sleep(SLEEP)
    p = dict(params)
    p.setdefault("apiKey", API_KEY)
    p.setdefault("serviceKey", API_KEY)
    if DEBUG: print("[REQ]", url, {k:v for k,v in p.items() if k not in ("apiKey","serviceKey")})
    r = S.get(url, params=p, timeout=timeout)
    txt = r.text
    if DEBUG: print("[RESP]", r.status_code, r.headers.get("content-type"), txt[:180].replace("\n"," "))
    try:
        return r.json()
    except Exception:
        return parse_lenient(txt)

def as_rows(data):
    if isinstance(data, list): return data
    if isinstance(data, dict):
        if data.get("err") or data.get("errCd"): raise RuntimeError(f"KOSIS error ({data.get('errCd') or data.get('err')}): {data.get('errMsg')}")
        for k in ("list","STAT_DATA","data","rows","result","STAT_LIST"):
            if isinstance(data.get(k), list): return data[k]
        if any(str(v).upper()=="NODATA" for v in data.values()): return []
    raise RuntimeError("Unexpected response schema")

# ==== 1) 카탈로그 검색 (ID 고정 없음) ====
def search_catalog(keywords):
    url = "https://kosis.kr/openapi/statisticsSearch.do"
    seen = {}
    for kw in keywords:
        params = {"method":"getList","format":"json","content":"json","searchNm":kw}
        try:
            data = api_get(url, params, timeout=45)
            rows = as_rows(data)
        except Exception as e:
            if DEBUG: print("[WARN] search fail:", kw, e)
            continue
        for r in rows:
            org = str(r.get("ORG_ID") or r.get("orgId") or "").strip()
            tbl = str(r.get("TBL_ID") or r.get("tblId") or "").strip()
            if not org or not tbl: continue
            title = str(r.get("TBL_NM") or r.get("TITLE") or r.get("title") or "")
            key = (org, tbl)
            if key not in seen:
                seen[key] = {"orgId":org, "tblId":tbl, "title":title}
    return list(seen.values())

# ==== 2) 코드 후보: statHtml에서 추출 ====
ITM = re.compile(r'ITM_ID"\s*:\s*"([^"]+)"[^}]*ITM_NM"\s*:\s*"([^"]+)"')
OBJ = re.compile(r'(?:OBJ_ID|C1)"\s*:\s*"([^"]+)"[^}]*?(?:OBJ_NM|C1_NM)"\s*:\s*"([^"]+)"')
def discover_codes(org, tbl):
    url = "https://kosis.kr/statHtml/statHtml.do"
    html = S.get(url, params={"orgId":org,"tblId":tbl}, timeout=30).text
    items = ITM.findall(html)
    objs  = OBJ.findall(html)
    def sort_pref(pairs, prefs):
        return sorted(pairs, key=lambda x: (0 if any(p in x[1] for p in prefs) else 1, x[1]))
    items = sort_pref(items, ["합계","전체","총계"])
    objs  = sort_pref(objs,  ["전국","합계","전체","계"])
    return {
        "itmId": items[0][0] if items else "all",
        "objL1": objs[0][0] if objs else "ALL",
        "itmChoices": [c for c,_ in items][:10] or ["all","ALL","T","TOT","0"],
        "obj1Choices": [c for c,_ in objs][:10] or ["ALL","11*","T*"]
    }

# ==== 3) 주기 & 최근 시점 ====
def recent_windows(prdSe, n=6):
    from datetime import datetime
    now = datetime.now()
    p = prdSe.upper()
    wins = []
    if p in ("A","Y"):
        for i in range(n):
            y = now.year - i
            wins.append((f"{y:04d}", f"{y:04d}"))
    elif p == "M":
        y,m = now.year, now.month
        for _ in range(n):
            wins.append((f"{y:04d}{m:02d}", f"{y:04d}{m:02d}"))
            m -= 1
            if m==0: m=12; y-=1
    elif p == "Q":
        y,q = now.year, (now.month-1)//3+1
        for _ in range(n):
            wins.append((f"{y:04d}{q:02d}", f"{y:04d}{q:02d}"))  # YYYYQQ (01~04)
            q -= 1
            if q==0: q=4; y-=1
    elif p == "H":
        y,h = now.year, (1 if now.month<=6 else 2)
        for _ in range(n):
            wins.append((f"{y:04d}{h:02d}", f"{y:04d}{h:02d}"))  # YYYYHH (01~02)
            h -= 1
            if h==0: h=2; y-=1
    return wins

# ==== 4) Param 호출 (20/21/30 폴백) ====
def call_param(org, tbl, prdSe, start, end, extras):
    url = "https://kosis.kr/openapi/Param/statisticsParameterData.do"
    base = {
        "method":"getList","format":"json","jsonVD":"Y",
        "orgId":org,"tblId":tbl,"prdSe":prdSe,
        "startPrdDe":start,"endPrdDe":end,
        "itmId":extras.get("itmId","all"),
        "objL1":extras.get("objL1","ALL"),
    }
    def send(p):
        return api_get(url, p, timeout=45)

    data = send(base)

    # 20/21 → objL2/objL3 승급
    if isinstance(data, dict) and (data.get("err") in ("20","21") or data.get("errCd") in ("20","21")):
        p2 = dict(base); p2.setdefault("objL2","ALL"); data = send(p2)
        if isinstance(data, dict) and (data.get("err") in ("20","21") or data.get("errCd") in ("20","21")):
            p3 = dict(p2); p3.setdefault("objL3","ALL"); data = send(p3)

    # 30 → 최근 시점 × 코드 후보 재시도
    if isinstance(data, dict) and (data.get("err") == "30" or data.get("errCd") == "30"):
        itmC  = extras.get("itmChoices", [extras.get("itmId","all")])
        objC  = extras.get("obj1Choices", [extras.get("objL1","ALL")])
        for ss,ee in recent_windows(prdSe, n=6):
            for itm in itmC[:5]:
                for obj in objC[:5]:
                    p = dict(base, startPrdDe=ss, endPrdDe=ee, itmId=itm, objL1=obj)
                    data = send(p)
                    if isinstance(data, dict) and (data.get("err") in ("20","21") or data.get("errCd") in ("20","21")):
                        p2 = dict(p); p2.setdefault("objL2","ALL"); data = send(p2)
                        if isinstance(data, dict) and (data.get("err") in ("20","21") or data.get("errCd") in ("20","21")):
                            p3 = dict(p2); p3.setdefault("objL3","ALL"); data = send(p3)
                    try:
                        return as_rows(data)
                    except Exception:
                        continue
    return as_rows(data)

# ==== 5) 테이블별 디스커버리 → 매니페스트 ====
def discover_table(org, tbl):
    # 주기 후보: 일반적으로 많이 쓰는 순서
    prds = ["M","Y","Q","H","IR"]
    extras = discover_codes(org, tbl)
    for p in prds:
        for s,e in recent_windows(p, n=6) or [(None,None)]:
            try:
                rows = call_param(org, tbl, p, s, e, extras)
                if rows:
                    return {"orgId":org,"tblId":tbl,"prdSe":p,"start":s,"end":e,
                            "itmId":extras.get("itmId","all"),
                            "objL1":extras.get("objL1","ALL"),
                            "example_rows":len(rows)}
            except RuntimeError:
                continue
    return None

# ==== 6) 수집 (기간 확장) ====
def expand_periods(prdSe, start, end, span=24):
    # newEstPrdCnt 기반 대신, 단순히 과거로 n 구간 확장
    wins = recent_windows(prdSe, n=span)
    if not wins: return [(start, end)]
    return list(reversed(wins))  # 오래된→최신 순

def harvest_from_manifest(manifest, limit_per_table=3):
    for m in manifest:
        org, tbl, prd, s0, e0 = m["orgId"], m["tblId"], m["prdSe"], m["start"], m["end"]
        extras = {"itmId":m["itmId"], "objL1":m["objL1"]}
        wins = expand_periods(prd, s0, e0, span=limit_per_table)
        out_dir = OUT / f"{org}_{tbl}"
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        for (s,e) in wins:
            try:
                rows = call_param(org, tbl, prd, s, e, extras)
                if not rows: continue
                # 저장
                ts = time.strftime("%Y%m%d_%H%M%S")
                path = out_dir / f"{tbl}_{prd}_{s}_{e}_{ts}.jsonl"
                with open(path, "w", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                saved += 1
                if saved >= limit_per_table:
                    break
            except Exception as ex:
                if DEBUG: print("[ERR save]", org, tbl, prd, s, e, ex)
                continue

def main():
    # 1) 카탈로그 자동 수집 (ID 고정 X)
    print("[STEP1] 검색 중 …")
    catalog = search_catalog(KEYWORDS)
    if not catalog:
        print("카탈로그가 비어있습니다.")
        return
    with open(BASE/"catalog.csv","w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f, fieldnames=["orgId","tblId","title"])
        w.writeheader(); w.writerows(catalog)
    print(f"[OK] catalog.csv 저장: {len(catalog)}개 표")

    # 2) 각 표별로 '되는 조합' 자동 탐색 → manifest 작성
    print("[STEP2] 디스커버리(되는 조합 찾기) …")
    manifest = []
    for i, row in enumerate(catalog, 1):
        org, tbl, title = row["orgId"], row["tblId"], row.get("title","")
        try:
            res = discover_table(org, tbl)
            if res:
                res["title"] = title
                manifest.append(res)
                print(f"  [+] FOUND {org}/{tbl} → {res['prdSe']} {res['start']}-{res['end']} rows={res['example_rows']}")
        except Exception as ex:
            if DEBUG: print("  [skip]", org, tbl, ex)
        # 너무 많으면 일부만
        if len(manifest) >= 30:  # 처음엔 30표만
            break
    if not manifest:
        print("되는 조합을 찾지 못했습니다.")
        return
    with open(BASE/"manifest.csv","w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f, fieldnames=["orgId","tblId","prdSe","start","end","itmId","objL1","example_rows","title"])
        w.writeheader(); w.writerows(manifest)
    print(f"[OK] manifest.csv 저장: {len(manifest)}개 표")

    # 3) 수집 (기간 확장하며 일부만 저장)
    print("[STEP3] 수집 …")
    harvest_from_manifest(manifest, limit_per_table=5)
    print("[DONE] data/ 폴더 확인")

if __name__ == "__main__":
    main()
