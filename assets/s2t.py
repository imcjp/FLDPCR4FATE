import opencc
import argparse

def convert_simplified_to_traditional(input_file, output_file):
    converter = opencc.OpenCC('s2t.json')  # 使用s2t.json配置文件进行简繁体转换
    
    # 读取文件内容
    with open(input_file, 'r', encoding='utf-8') as fin:
        content = fin.read()
    
    # 转换简体到繁体
    converted_content = converter.convert(content)
    
    # 写入转换后的内容到文件
    with open(output_file, 'w', encoding='utf-8') as fout:
        fout.write(converted_content)

def main():
    parser = argparse.ArgumentParser(description='Convert Simplified Chinese text to Traditional Chinese.')
    parser.add_argument('-i', '--input_file', type=str, required=True, help='Path to the input text file')
    parser.add_argument('-o', '--output_file', type=str, required=True, help='Path to the output text file')
    args = parser.parse_args()

    convert_simplified_to_traditional(args.input_file, args.output_file)
    print(f"转换完成！已将简体字转换为繁体字并保存到 {args.output_file}")

if __name__ == "__main__":
    main()
