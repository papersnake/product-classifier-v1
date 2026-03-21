'''
Author: papersnake cctv5cn@gmail.com
Date: 2026-03-21 10:00:00
LastEditors: papersnake cctv5cn@gmail.com
LastEditTime: 2026-03-21 10:00:00
FilePath: \myproject\test_word_segmentation.py
Description: jieba vs pkuseg 分词对比测试

Copyright (c) 2026 by papersnake, All Rights Reserved. 
'''

import pandas as pd
import jieba as jb
import pkuseg
import os
import time
from collections import Counter

# 配置路径
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PRODUCTS_CSV = os.path.join(DATA_DIR, "products.csv")
DICT_PATH = os.path.join(DATA_DIR, "productnames.dict")
OUTPUT_DIR = os.path.join(DATA_DIR, "segmentation_comparison")


def load_products():
    """加载产品数据"""
    print("加载产品数据...")
    df = pd.read_csv(PRODUCTS_CSV)
    df.columns = df.columns.str.strip()
    print(f"共加载 {len(df)} 条产品记录")
    return df


def setup_jieba():
    """设置jieba分词器"""
    print("\n初始化 jieba 分词器...")
    jb.load_userdict(DICT_PATH)
    print("jieba 词典加载完成")
    return jb


def setup_pkuseg():
    """设置pkuseg分词器"""
    print("\n初始化 pkuseg 分词器...")
    seg = pkuseg.pkuseg(user_dict=DICT_PATH)
    print("pkuseg 词典加载完成")
    return seg


def segment_jieba(text, segmenter):
    """使用jieba分词"""
    words = list(segmenter.cut(text))
    return words


def segment_pkuseg(text, segmenter):
    """使用pkuseg分词"""
    words = segmenter.cut(text)
    return words


def compare_segmentations(text, jieba_seg, pku_seg, sample_id):
    """对比两种分词结果"""
    jieba_words = segment_jieba(text, jieba_seg)
    pku_words = segment_pkuseg(text, pku_seg)
    
    jieba_set = set(jieba_words)
    pku_set = set(pku_words)
    
    common = jieba_set & pku_set
    only_jieba = jieba_set - pku_set
    only_pku = pku_set - jieba_set
    
    return {
        'sample_id': sample_id,
        'text': text,
        'jieba_words': jieba_words,
        'pku_words': pku_words,
        'jieba_count': len(jieba_words),
        'pku_count': len(pku_words),
        'common_count': len(common),
        'only_jieba': list(only_jieba),
        'only_pku': list(only_pku),
        'agreement_rate': len(common) / max(len(jieba_set | pku_set), 1)
    }


def run_comparison(df, n_samples=500):
    """运行分词对比测试"""
    print("\n" + "="*60)
    print("开始分词对比测试")
    print("="*60)
    
    # 初始化分词器
    jieba_seg = setup_jieba()
    pku_seg = setup_pkuseg()
    
    # 获取产品名称进行测试
    texts = df['ItemName'].dropna().astype(str).str.strip().unique()[:n_samples]
    
    results = []
    jieba_total_time = 0
    pku_total_time = 0
    
    print(f"\n测试 {len(texts)} 个产品名称...")
    
    for i, text in enumerate(texts):
        if i % 100 == 0:
            print(f"  进度: {i}/{len(texts)}")
        
        # jieba 计时
        start = time.time()
        jieba_words = list(jieba_seg.cut(text))
        jieba_time = time.time() - start
        jieba_total_time += jieba_time
        
        # pkuseg 计时
        start = time.time()
        pku_words = pku_seg.cut(text)
        pku_time = time.time() - start
        pku_total_time += pku_time
        
        # 对比
        jieba_set = set(jieba_words)
        pku_set = set(pku_words)
        common = jieba_set & pku_set
        agreement = len(common) / max(len(jieba_set | pku_set), 1)
        
        results.append({
            'text': text,
            'jieba_words': '|'.join(jieba_words),
            'pku_words': '|'.join(pku_words),
            'jieba_count': len(jieba_words),
            'pku_count': len(pku_words),
            'agreement_rate': agreement
        })
    
    return pd.DataFrame(results), jieba_total_time, pku_total_time


def analyze_results(results_df, jieba_time, pku_time):
    """分析对比结果"""
    print("\n" + "="*60)
    print("分词对比分析报告")
    print("="*60)
    
    n_samples = len(results_df)
    
    # 一致性统计
    avg_agreement = results_df['agreement_rate'].mean()
    high_agreement = (results_df['agreement_rate'] >= 0.8).sum()
    medium_agreement = (results_df['agreement_rate'] >= 0.5).sum()
    low_agreement = (results_df['agreement_rate'] < 0.5).sum()
    
    print(f"\n【一致性分析】")
    print(f"  平均一致率: {avg_agreement:.2%}")
    print(f"  高一致 (>=80%): {high_agreement} ({high_agreement/n_samples:.1%})")
    print(f"  中一致 (50-80%): {medium_agreement} ({medium_agreement/n_samples:.1%})")
    print(f"  低一致 (<50%): {low_agreement} ({low_agreement/n_samples:.1%})")
    
    # 分词数量统计
    avg_jieba_count = results_df['jieba_count'].mean()
    avg_pku_count = results_df['pku_count'].mean()
    
    print(f"\n【分词数量统计】")
    print(f"  jieba 平均每句分词数: {avg_jieba_count:.2f}")
    print(f"  pkuseg 平均每句分词数: {avg_pku_count:.2f}")
    print(f"  差异: {avg_pku_count - avg_jieba_count:.2f}")
    
    # 性能统计
    print(f"\n【性能对比】")
    print(f"  jieba 总耗时: {jieba_time:.3f}s")
    print(f"  pkuseg 总耗时: {pku_time:.3f}s")
    print(f"  速度比: jieba 是 pkuseg 的 {pku_time/jieba_time:.2f} 倍")
    
    # 找出差异最大的样本
    print(f"\n【典型差异案例】")
    diff_cases = results_df.nsmallest(5, 'agreement_rate')
    for _, row in diff_cases.iterrows():
        print(f"\n  原文: {row['text'][:50]}...")
        print(f"  jieba: {row['jieba_words'][:60]}...")
        print(f"  pkuseg: {row['pku_words'][:60]}...")
        print(f"  一致率: {row['agreement_rate']:.1%}")
    
    return {
        'avg_agreement': avg_agreement,
        'avg_jieba_count': avg_jieba_count,
        'avg_pku_count': avg_pku_count,
        'jieba_time': jieba_time,
        'pku_time': pku_time,
        'speed_ratio': pku_time / jieba_time
    }


def show_word_frequency(results_df):
    """分析词频差异"""
    print("\n" + "="*60)
    print("词频分析")
    print("="*60)
    
    jieba_all_words = []
    pku_all_words = []
    
    for _, row in results_df.iterrows():
        jieba_all_words.extend(row['jieba_words'].split('|'))
        pku_all_words.extend(row['pku_words'].split('|'))
    
    jieba_counter = Counter(jieba_all_words)
    pku_counter = Counter(pku_all_words)
    
    print(f"\njieba 词表大小: {len(jieba_counter)}")
    print(f"pkuseg 词表大小: {len(pku_counter)}")
    
    # 常见词差异
    print("\n【高频词对比 (Top 10)】")
    print(f"{'词':<15} {'jieba频次':<12} {'pkuseg频次':<12} {'差异'}")
    print("-" * 55)
    
    all_top_words = set([w for w, _ in jieba_counter.most_common(15)] + 
                        [w for w, _ in pku_counter.most_common(15)])
    
    diff_stats = []
    for word in list(all_top_words)[:10]:
        j_count = jieba_counter.get(word, 0)
        p_count = pku_counter.get(word, 0)
        diff = j_count - p_count
        diff_stats.append((word, j_count, p_count, diff))
        print(f"{word:<15} {j_count:<12} {p_count:<12} {diff:+d}")
    
    return jieba_counter, pku_counter


def main():
    """主函数"""
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载数据
    df = load_products()
    
    # 运行对比
    results_df, jieba_time, pku_time = run_comparison(df, n_samples=500)
    
    # 分析结果
    stats = analyze_results(results_df, jieba_time, pku_time)
    
    # 词频分析
    show_word_frequency(results_df)
    
    # 保存详细结果
    output_file = os.path.join(OUTPUT_DIR, "segmentation_comparison_results.csv")
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n详细结果已保存至: {output_file}")
    
    # 总结建议
    print("\n" + "="*60)
    print("结论与建议")
    print("="*60)
    
    if stats['avg_agreement'] >= 0.9:
        print("\n[OK] 两种分词方案高度一致，jieba 完全够用")
    elif stats['avg_agreement'] >= 0.7:
        print("\n[WARN] 两种分词方案有一定差异，可考虑切换到 pkuseg")
    else:
        print("\n[ERROR] 两种分词方案差异较大，建议进行离线测试后切换")
    
    if stats['speed_ratio'] > 2:
        print("  注意: pkuseg 速度较慢 (~{:.1f}x)，生产环境需考虑性能".format(stats['speed_ratio']))
    
    return stats


if __name__ == "__main__":
    main()
