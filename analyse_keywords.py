# analyze_keywords.py

import json
from pathlib import Path
from collections import defaultdict
from nltk.stem import PorterStemmer
from typing import List, Dict, Tuple, Set

ps = PorterStemmer()

def normalize_word(word: str) -> str:
    """Normalize word to its stem."""
    return ps.stem(word.lower().strip())

# ---------------------------
# 📦 分析类型1：关键词频率结构
# ---------------------------
def merge_freq_jsons(freq_json_paths: List[str], threshold: int = 10) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
    """
    合并 keyword_length 格式的 JSON 文件，并统计每个长度下频率大于阈值的词数量。
    
    返回：
        merged: 合并后的词频字典
        count_result: 每个长度下的关键词计数（freq > threshold）
    """
    merged = defaultdict(lambda: defaultdict(int))

    for path in freq_json_paths:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for length_key, kw_dict in data.items():
            for word, freq in kw_dict.items():
                merged[length_key][word] += freq

    count_result = {}
    for length_key, kw_dict in merged.items():
        count = sum(1 for freq in kw_dict.values() if freq > threshold)
        count_result[length_key] = count

    return dict(merged), count_result


# ---------------------------
# 📦 分析类型2：语义分类关键词结构
# ---------------------------
def merge_class_jsons(class_json_paths: List[str]) -> Set[str]:
    """从类别关键词 JSON 中提取所有关键词，不去重、不词频，仅聚合"""
    all_keywords = set()
    for path in class_json_paths:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for word_list in data.values():
            all_keywords.update(word_list)
    return all_keywords


# ---------------------------
# 🔄 合并频率类 + 分类类关键词，并按词干去重
# ---------------------------
def merge_and_stem_keywords(
    freq_json_paths: List[str] = None,
    class_json_paths: List[str] = None,
    threshold: int = 10
) -> Set[str]:
    freq_json_paths = freq_json_paths or []
    class_json_paths = class_json_paths or []

    all_keywords = set()

    # 加载频率类 JSON
    for path in freq_json_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for kw_dict in data.values():
            for word, freq in kw_dict.items():
                if freq > threshold:
                    all_keywords.add(word)

    # 加载分类类 JSON
    for path in class_json_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for word_list in data.values():
            all_keywords.update(word_list)

    # 词干化 + 去重
    stemmed = {normalize_word(w) for w in all_keywords}
    return stemmed


# ---------------------------
# ✅ 示例调用
# ---------------------------
if __name__ == "__main__":
    # 添加所有inside_keywords中的JSON文件到freq_files
    freq_files = [
        "inside_keywords/edit_type_action_change_change_by_length.json",
        "inside_keywords/edit_type_add_noun_freq_by_length.json",
        "inside_keywords/edit_type_appearance_alter_change_by_length.json",
        "inside_keywords/edit_type_attribute_modification_change_by_length.json",
        "inside_keywords/edit_type_background_change_noun_freq_by_length.json",
        "inside_keywords/edit_type_color_alter_change_by_length.json",
        "inside_keywords/edit_type_env_noun_freq_by_length.json",
        "inside_keywords/edit_type_remove_noun_freq_by_length.json",
        "inside_keywords/edit_type_replace_change_by_length.json",
        "inside_keywords/edit_type_style_noun_freq_by_length.json",
        "inside_keywords/edit_type_swap_change_by_length.json",
        "inside_keywords/edit_type_tone_transfer_change_by_length.json",
        "inside_keywords/edit_type_tune_transfer_change_by_length.json"
    ]
    
    # 添加所有outside_keywords中的JSON文件到class_files
    class_files = [
        "outside_keywords/keyword_action.json",
        "outside_keywords/keyword_appearance.json",
        "outside_keywords/keyword_brand.json",
        "outside_keywords/keyword_color.json",
        "outside_keywords/keyword_function_purpose.json",
        "outside_keywords/keyword_material.json",
        "outside_keywords/keyword_size_shape.json",
        "outside_keywords/keyword_spatial_position.json",
        "outside_keywords/keyword_temporal_attributes.json"
    ]

    # ✅ 1. 分析频率类关键词文件 (inside_keywords)
    print("\n" + "="*60)
    print("📊 分析频率类关键词文件 (inside_keywords)")
    print("="*60)
    
    total_freq_keywords = 0
    for i, freq_file in enumerate(freq_files, 1):
        print(f"\n{i}. 分析文件: {freq_file}")
        try:
            with open(freq_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            file_total = 0
            length_stats = {}
            
            for length_key, kw_dict in data.items():
                high_freq_count = sum(1 for freq in kw_dict.values() if freq > 10)
                total_words = len(kw_dict)
                length_stats[length_key] = {
                    'total': total_words,
                    'high_freq': high_freq_count
                }
                file_total += total_words
            
            print(f"   📈 总词数: {file_total}")
            print(f"   📊 按长度统计:")
            for length_key, stats in length_stats.items():
                print(f"      {length_key}: {stats['total']} 词 (频率>10: {stats['high_freq']})")
            
            total_freq_keywords += file_total
            
        except Exception as e:
            print(f"   ❌ 错误: {e}")
    
    print(f"\n📊 频率类文件总计: {total_freq_keywords} 个关键词")
    
    # ✅ 2. 分析语义分类关键词文件 (outside_keywords)
    print("\n" + "="*60)
    print("🏷️  分析语义分类关键词文件 (outside_keywords)")
    print("="*60)
    
    total_class_keywords = 0
    for i, class_file in enumerate(class_files, 1):
        print(f"\n{i}. 分析文件: {class_file}")
        try:
            with open(class_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            file_total = 0
            category_stats = {}
            
            for category, word_list in data.items():
                if isinstance(word_list, list):
                    category_stats[category] = len(word_list)
                    file_total += len(word_list)
                else:
                    category_stats[category] = 1
                    file_total += 1
            
            print(f"   📈 总词数: {file_total}")
            print(f"   📊 按类别统计:")
            for category, count in category_stats.items():
                print(f"      {category}: {count} 词")
            
            total_class_keywords += file_total
            
        except Exception as e:
            print(f"   ❌ 错误: {e}")
    
    print(f"\n🏷️  分类文件总计: {total_class_keywords} 个关键词")
    
    # # ✅ 3. 合并分析
    # print("\n" + "="*60)
    # print("🔗 合并分析")
    # print("="*60)
    
    # merged_freq, count_result = merge_freq_jsons(freq_files)
    # print("\n📊 合并后的频率统计 (频率 > 10):")
    # total_high_freq = 0
    # for k, v in sorted(count_result.items()):
    #     print(f"   {k}: {v} 词")
    #     total_high_freq += v
    # print(f"   总计高频词: {total_high_freq}")

    # class_keywords = merge_class_jsons(class_files)
    # print(f"\n🏷️  合并后的分类关键词: {len(class_keywords)} 个")

    # stemmed_keywords = merge_and_stem_keywords(freq_files, class_files)
    # print(f"\n🌱 词干化后的唯一关键词: {len(stemmed_keywords)} 个")
    # print("🔍 样本:", list(stemmed_keywords)[:20])
