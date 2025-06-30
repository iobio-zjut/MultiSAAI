#!/bin/bash

BASE_FOLDER="$1"

# 创建存放ANARCI输出文件的文件夹
#output_folder="./example_20250605_140954/output_anarci"
#fasta_files="./example_20250605_140954/fasta_files"
#mkdir -p $output_folder
# 路径拼接
output_folder="${BASE_FOLDER}/output_anarci"
fasta_files="${BASE_FOLDER}/fasta_files"
mkdir -p "$output_folder"

# 创建存放错误文件名的文件
#error_file="error.txt"
#> $error_file  # 清空或创建error.txt文件
error_file="${BASE_FOLDER}/error.txt"
> "$error_file"  # 清空或创建 error.txt 文件

# 循环处理所有的fasta文件
for file in "$fasta_files"/*.fasta; do
    # 提取文件名（不包含路径）
    filename=$(basename $file)
    
    # 提取文件名的前缀（不包含扩展名）
    filename_no_ext="${filename%.*}"
    
    # 执行ANARCI命令
    if ANARCI --sequence $file --outfile $output_folder/$filename_no_ext.fasta --scheme imgt; then
        echo "File $filename encoded successfully."
    else
        echo "Error encoding file: $filename"
        echo "$filename" >> $error_file
    fi
done

echo "ANARCI编码完成。输出文件存放在$output_folder 文件夹中。"