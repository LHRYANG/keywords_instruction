import os
import json

# 设置你的目录路径
input_dir = "./"

# 初始化输出字符串列表
output_lines = []

# 遍历目录下所有 .json 文件
for filename in sorted(os.listdir(input_dir)):
    if filename.endswith(".json"):
        file_path = os.path.join(input_dir, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                json_str = json.dumps(data, ensure_ascii=False)
                output_lines.append(f"{filename}: {json_str}")
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")

# 将结果拼接为完整字符串
output = "\n".join(output_lines)

# 可选择写入文件
with open("merged_output.txt", "w", encoding="utf-8") as f:
    f.write(output)

print("✅ Done. Output saved to merged_output.txt")
