from bank_pipeline_v1 import read_one, save_one, process_many

# 1) 메모리로 바로 받기
# tidy, wide_code, wide_multi = read_one("영업점포 현황_20250820.xlsx", sheet="auto")

# 2) 파일로 저장(파일명 '_' 앞부분 prefix로 저장)
# save_one("자동화기기 설치현황_20250820.xlsx", outdir="out", sheet="auto")

# 3) 여러 파일 한꺼번에
process_many("./bss_result/일반현황/*.xlsx", outdir="out", sheet="auto")
