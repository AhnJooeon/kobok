
# -*- coding: utf-8 -*-
"""배치 처리 v2: 가변 '구분' 지원 (--group auto|컬럼명)"""
import argparse, os
from glob import glob
from pathlib import Path
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from read_branch_status_fixed_v2 import read_branch_status, _detect_header_row_single

def detect_sheet_auto(path: str):
    import pandas as pd
    try:
        xl = pd.ExcelFile(path)
        for idx, sheet_name in enumerate(xl.sheet_names):
            try:
                h = _detect_header_row_single(path, sheet=idx)
                df_head = pd.read_excel(path, sheet_name=idx, header=h, nrows=2, dtype=str)
                if "코드" in df_head.columns and any(col in df_head.columns for col in ["은행명","기관명"]):
                    return idx, sheet_name, h
            except Exception:
                continue
        return 0, xl.sheet_names[0], 1
    except Exception:
        return 0, None, 1

def process_one(file_path: str, sheet_opt, group_opt, outdir: Path):
    out_subdir = outdir / Path(file_path).stem
    out_subdir.mkdir(parents=True, exist_ok=True)

    if isinstance(sheet_opt, int):
        sheet_idx, sheet_name, header_row = sheet_opt, None, None
    else:
        sheet_idx, sheet_name, header_row = detect_sheet_auto(file_path)

    log = {"file": file_path, "sheet_idx": sheet_idx, "sheet_name": sheet_name,
           "header_row": header_row, "group_opt": group_opt, "status": "ok", "msg": ""}
    try:
        g = group_opt if group_opt != "auto" else "auto"
        tidy, wide_code, wide_multi = read_branch_status(file_path, sheet=sheet_idx, group_col=g)

        tidy_path = out_subdir / "tidy_fixed.csv"
        wide_code_path = out_subdir / "wide_code_fixed.csv"
        wide_multi_path = out_subdir / "wide_multi_fixed.csv"
        tidy.to_csv(tidy_path, index=False)
        wide_code.to_csv(wide_code_path, index=False)
        wide_multi.to_csv(wide_multi_path, index=False)

        # 출처 정보
        for df in (tidy, wide_code, wide_multi):
            df["_source_file"] = os.path.basename(file_path)
            df["_sheet_idx"] = sheet_idx
            df["_sheet_name"] = sheet_name if sheet_name is not None else ""

        return log, tidy, wide_code, wide_multi
    except Exception as e:
        log["status"] = "error"
        log["msg"] = str(e)
        return log, None, None, None

def process_multiple_files(pattern: str, sheet="auto", group_col="auto", outdir="out_batch"):
    os.makedirs(outdir, exist_ok=True)
    tidy_all, wide_code_all, wide_multi_all = [], [], []

    for path in glob.glob(pattern):
        try:
            tidy, wide_code, wide_multi = read_branch_status(path, sheet=sheet, group_col=group_col)

            # 출처 태그 추가
            for df in (tidy, wide_code, wide_multi):
                df["_source_file"] = os.path.basename(path)

            # CSV 저장 (개별)
            base = os.path.splitext(os.path.basename(path))[0]
            tidy.to_csv(f"{outdir}/{base}_tidy.csv", index=False)
            wide_code.to_csv(f"{outdir}/{base}_wide_code.csv", index=False)
            wide_multi.to_csv(f"{outdir}/{base}_wide_multi.csv", index=False)

            tidy_all.append(tidy)
            wide_code_all.append(wide_code)
            wide_multi_all.append(wide_multi)

        except Exception as e:
            print(f"[ERROR] {path}: {e}")

    # 합본 저장
    if tidy_all:
        pd.concat(tidy_all).to_csv(f"{outdir}/all_tidy.csv", index=False)
    if wide_code_all:
        pd.concat(wide_code_all).to_csv(f"{outdir}/all_wide_code.csv", index=False)
    if wide_multi_all:
        pd.concat(wide_multi_all).to_csv(f"{outdir}/all_wide_multi.csv", index=False)

def process_files_per_excel(pattern: str, outdir: str, group_col="auto"):
    print("Start")
    os.makedirs(outdir, exist_ok=True)
    for p in glob(pattern):
        tidy, wide_code, wide_multi = read_branch_status(p, sheet=0, group_col=group_col)  # 시트 다르면 바꿔
        base = os.path.splitext(os.path.basename(p))[0]
        subdir = os.path.join(outdir, base)
        os.makedirs(subdir, exist_ok=True)
        base = os.path.splitext(os.path.basename(p))[0]
        prefix = base.split("_")[0]  # 언더바 앞부분만
        tidy.to_csv(os.path.join(outdir, f"{prefix}_tidy.csv"), index=False)
        wide_code.to_csv(os.path.join(outdir, f"{prefix}_wide_code.csv"), index=False)
        wide_multi.to_csv(os.path.join(outdir, f"{prefix}_wide_multi.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="폴더 경로 또는 글롭 패턴 (예: data/*.xlsx)")
    ap.add_argument("--sheet", default="auto", help="시트 인덱스(int) 또는 'auto'")
    ap.add_argument("--group", default="auto", help="'구분' 컬럼명 지정 또는 'auto'")
    ap.add_argument("--outdir", default="out_batch", help="출력 폴더")
    ap.add_argument("--workers", type=int, default=1, help="동시 처리 개수")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 입력 파일 목록
    if any(ch in args.input for ch in ["*", "?", "["]):
        paths = glob.glob(args.input)
    else:
        p = Path(args.input)
        if p.is_dir():
            paths = glob.glob(str(p / "*.xlsx"))
        elif p.is_file():
            paths = [str(p)]
        else:
            paths = []
    paths = sorted(paths)
    if not paths:
        print("[WARN] 입력 파일을 찾지 못했습니다:", args.input)
        return

    # sheet 옵션 처리
    sheet_opt = args.sheet
    if sheet_opt != "auto":
        try:
            sheet_opt = int(sheet_opt)
        except Exception:
            print("[WARN] --sheet 값이 잘못되었습니다. auto로 진행")
            sheet_opt = "auto"

    logs, all_tidy, all_wide_code, all_wide_multi = [], [], [], []
    if args.workers == 1:
        for f in paths:
            log, tidy, wc, wm = process_one(f, sheet_opt, args.group, outdir)
            logs.append(log)
            if tidy is not None: all_tidy.append(tidy)
            if wc is not None: all_wide_code.append(wc)
            if wm is not None: all_wide_multi.append(wm)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(process_one, f, sheet_opt, args.group, outdir) for f in paths]
            for fut in as_completed(futs):
                log, tidy, wc, wm = fut.result()
                logs.append(log)
                if tidy is not None: all_tidy.append(tidy)
                if wc is not None: all_wide_code.append(wc)
                if wm is not None: all_wide_multi.append(wm)

    if all_tidy:
        pd.concat(all_tidy, ignore_index=True).to_csv(outdir/"all_tidy.csv", index=False)
    if all_wide_code:
        pd.concat(all_wide_code, ignore_index=True).to_csv(outdir/"all_wide_code.csv", index=False)
    if all_wide_multi:
        pd.concat(all_wide_multi, ignore_index=True).to_csv(outdir/"all_wide_multi.csv", index=False)

    pd.DataFrame(logs).to_csv(outdir/"batch_log.csv", index=False)
    print("[DONE] files:", len(paths), "outdir:", outdir)

if __name__ == "__main__":
    # main()
    # process_multiple_files("./bss_result/일반현황/*.xlsx", sheet="auto", group_col="auto", outdir="./_parsed_out")
    process_files_per_excel("./bss_result/일반현황/*.xlsx", outdir="./_parsed_out/일반현황", group_col="auto")
