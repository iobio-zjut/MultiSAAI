import os
import re
import csv
import sys

base_path = sys.argv[1]
existing_csv_file = os.path.join(base_path, "data.csv")
output_anarci_file = os.path.join(base_path, "output_anarci")
# existing_csv_file = "./example_20250605_140954/data.csv"
# 循环处理每个编码后的文件
# for file in os.listdir("./example_20250605_140954/output_anarci"):
for file in os.listdir(output_anarci_file):
    if file.endswith(".fasta"):
        antibody_name = file.split("_")[0]  # 提取抗体编号
        chain_type = file.split("_")[1].split(".")[0]  # 提取链类型

        # with open(os.path.join("./D42-2_20250605_105234/output_anarci", file), 'r') as f:
        with open(os.path.join(output_anarci_file, file), 'r') as f:
            lines = [line for line in f.readlines() if line.strip().startswith(("H", "L"))]
            antibody_name = file.split("_")[0]  # 提取抗体编号
            chain_type = file.split("_")[1].split(".")[0]  # 提取链类型
            # 找到LCDR和HCDR的起始和结束编号
            start_lines = [27, 56, 105]
            end_lines = [38, 65, 117]

            # 提取LCDR和HCDR的氨基酸序列
            cdr_sequences = []
            for start, end in zip(start_lines, end_lines):
                cdr_sequence = ""
                for line in lines:
                    line_number = int(line.split()[1])
                    AA = line.split()[-1]
                    if line_number >= start and line_number <= end and AA != "-":
                        cdr_sequence += AA
                cdr_sequences.append(cdr_sequence)

            print(cdr_sequences)
                # 读取已有的CSV文件
        # antibody_row = int(antibody_name) - 1
        antibody_row = int(antibody_name)
        with open(existing_csv_file, mode='r') as file:
            reader = csv.reader(file)
            # next(reader)
            data = list(reader)

        # 检查antibody_name及chain_type，更新CSV文件
        for idx, row in enumerate(data):
            if idx == antibody_row:
                if chain_type == "H":
                    row[6:9] = cdr_sequences
                elif chain_type == "L":
                    row[9:12] = cdr_sequences

        # 将更新后的数据写回CSV文件
        with open(existing_csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)

        print("CDR sequences have been added to the existing CSV file.")

