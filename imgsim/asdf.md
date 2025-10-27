
이미지 유사도 기반 분류 및 RAW 파일 동기화 도구

positional arguments:
  {classify,sync-raw}  실행할 명령
    classify           JPG 파일을 유사도 기준으로 분류
    sync-raw           RAW 파일을 JPG 위치에 맞춰 동기화

options:
  -h, --help           show this help message and exit

사용 예시:

  1. JPG 파일을 유사도 기준으로 분류:
    python image_sorter.py classify --source ./photos --output ./sorted --threshold 10

  2. 분류된 JPG에 맞춰 RW2 파일 동기화:
    python image_sorter.py sync-raw --source ./photos --sorted ./sorted --move

  3. 복사 모드로 안전하게 테스트:
    python image_sorter.py classify --source ./photos --output ./sorted
    python image_sorter.py sync-raw --source ./photos --sorted ./sorted
