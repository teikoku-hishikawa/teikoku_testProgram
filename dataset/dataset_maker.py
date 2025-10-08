import os
import json

from openpyxl import load_workbook

def QA_dataset(QAfolder_path, output_jsonl_path):
    with open(output_jsonl_path, 'w', encoding='utf-8') as jsonl_file:
        for filename in os.listdir(QAfolder_path):
            if filename.endswith((".xlsx",".xls",".xlsm")):
                file_path = os.path.join(QAfolder_path, filename)
                wb = load_workbook(file_path, data_only=True)  # マクロは無視、値のみ取得

                for sheet_name in wb.sheetnames:
                    ws = wb[sheet_name]

                    # ヘッダー取得
                    header_b = ws['B1'].value
                    header_c = ws['C1'].value

                    row = 2  # 2行目からデータ
                    while True:
                        cell_b = ws[f'B{row}'].value
                        cell_c = ws[f'C{row}'].value

                        # 終了条件（両方空なら終了）
                        if cell_b is None and cell_c is None:
                            break

                        record = {
                            "input": cell_b,
                            "output": cell_c
                        }

                        jsonl_file.write(json.dumps(record, ensure_ascii=False) + '\n')
                        row += 1

    print(f"完了：{output_jsonl_path} に保存されました。")

if __name__ == "__main__":
    import datetime
    makedate = datetime.datetime.now().strftime("%Y%m%d") 

    QAfolder_path = os.path.join(os.path.dirname(__file__), "ORG")
    output_jsonl_path = os.path.join(os.path.dirname(__file__), makedate, "dataset.jsonl")
    QA_dataset(QAfolder_path, output_jsonl_path)