import nbformat

# 노트북 파일 읽기
notebook_path = './step2/main.ipynb'  # 실제 파일 경로로 변경
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# widgets 메타데이터 삭제
if 'widgets' in nb.metadata:
    del nb.metadata['widgets']

# 수정된 노트북 저장
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)