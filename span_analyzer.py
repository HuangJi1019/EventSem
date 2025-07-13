import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import json
from pathlib import Path
from tqdm import tqdm
import argparse
from collections import defaultdict

class StandaloneSpanAnalyzer:
    """
    独立的Span分析工具，直接处理保存的特征文件
    """
    
    def __init__(self, max_event_spans=200, span_width_threshold=0.5, device='cuda'):
        """
        初始化分析器
        
        Args:
            max_event_spans: 最大事件span数量
            span_width_threshold: span宽度阈值
            device: 计算设备
        """
        self.max_event_spans = max_event_spans
        self.span_width_threshold = span_width_threshold
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # 统计结果存储
        self.results = {}
        
        # 错误统计
        self.error_stats = {
            'total_files': 0,
            'successful_files': 0,
            'failed_files': 0,
            'error_types': defaultdict(int)
        }
        
        print(f"初始化分析器 - 设备: {self.device}")
        print(f"配置: max_event_spans={max_event_spans}, span_width_threshold={span_width_threshold}")
        
        # 设置torch默认数据类型为float32，避免类型不匹配
        torch.set_default_dtype(torch.float32)
    
    def load_feature_file(self, file_path):
        """
        加载单个特征文件
        
        Args:
            file_path: 特征文件路径
            
        Returns:
            features: 特征张量 [seq_len, hidden_dim]
            valid_length: 有效长度
        """
        try:
            # 支持多种格式
            if file_path.endswith('.pt') or file_path.endswith('.pth'):
                data = torch.load(file_path, map_location='cpu')  # 先加载到CPU
            elif file_path.endswith('.npy'):
                data = torch.from_numpy(np.load(file_path))
            elif file_path.endswith('.npz'):
                npz_data = np.load(file_path)
                # 尝试常见的键名
                possible_keys = ['features', 'feat', 'data', 'clips_feat', 'clip_feat']
                data = None
                for key in possible_keys:
                    if key in npz_data:
                        data = torch.from_numpy(npz_data[key])
                        break
                
                if data is None:
                    # 如果没找到常见键名，列出所有可用键
                    available_keys = list(npz_data.keys())
                    if len(available_keys) == 1:
                        # 如果只有一个键，直接使用
                        data = torch.from_numpy(npz_data[available_keys[0]])
                    else:
                        raise ValueError(f"NPZ文件中找不到特征数据。可用键: {available_keys}")
            else:
                raise ValueError(f"不支持的文件格式: {file_path}")
            
            # 确保数据类型为float32（解决HalfTensor问题）
            if data.dtype == torch.float16:
                data = data.float()  # 转换为float32
            elif data.dtype not in [torch.float32, torch.float64]:
                data = data.float()  # 确保是浮点类型
            
            # 移动到指定设备
            data = data.to(self.device)
            
            # 确保是2D张量 [seq_len, hidden_dim]
            if data.dim() == 3 and data.size(0) == 1:
                data = data.squeeze(0)  # 移除batch维度
            elif data.dim() == 1:
                # 如果是1D，假设是单帧特征，添加序列维度
                data = data.unsqueeze(0)
            elif data.dim() > 3:
                raise ValueError(f"不支持的张量维度: {data.dim()}D, shape: {data.shape}")
            
            if data.dim() != 2:
                raise ValueError(f"期望2D张量，得到{data.dim()}D: {data.shape}")
            
            seq_len, hidden_dim = data.shape
            
            # 检查数据有效性
            if seq_len == 0 or hidden_dim == 0:
                raise ValueError(f"无效的特征尺寸: {data.shape}")
            
            # 检查是否有NaN或Inf
            if torch.isnan(data).any() or torch.isinf(data).any():
                print(f"警告: 文件 {file_path} 包含NaN或Inf值，将被替换为0")
                data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 创建简单的mask（假设所有帧都有效）
            mask = torch.ones(seq_len, dtype=torch.bool, device=self.device)
            valid_length = seq_len
            
            return data, mask, valid_length
            
        except Exception as e:
            print(f"加载文件失败 {file_path}: {e}")
            return None, None, None
    
    def generate_pseudo_event_for_single_video(self, src_vid, src_vid_mask):
        """
        为单个视频生成伪事件span
        
        Args:
            src_vid: 视频特征 [seq_len, hidden_dim]
            src_vid_mask: 掩码 [seq_len]
            
        Returns:
            spans_info: 包含span信息的字典
        """
        # 添加batch维度
        src_vid = src_vid.unsqueeze(0)  # [1, seq_len, hidden_dim]
        src_vid_mask = src_vid_mask.unsqueeze(0)  # [1, seq_len]
        
        bsz, L_src, hidden_dim = src_vid.size()
        device = src_vid.device
        max_spans = self.max_event_spans
        
        # 初始化输出张量
        pseudo_event_spans = torch.zeros(bsz, max_spans, 2, device=device)
        pseudo_event_spans_used = torch.zeros(bsz, max_spans, 2, device=device)
        pseudo_event_spans_mask = torch.zeros(bsz, max_spans, dtype=torch.bool, device=device)
        
        # 归一化视频特征
        norm_factor = torch.norm(src_vid, dim=2, keepdim=True) + 1e-8
        norm_vid = src_vid / norm_factor
        
        # 计算时序相似度矩阵
        tsm = torch.bmm(norm_vid, norm_vid.transpose(1, 2))  # [1, L_src, L_src]
        
        # 创建检测掩码
        mask = torch.tensor([
            [1., 1., 0., -1., -1.],
            [1., 1., 0., -1., -1.],
            [0., 0., 0., 0., 0.],
            [-1., -1., 0., 1., 1.],
            [-1., -1., 0., 1., 1.]
        ], device=device)
        mask_size = mask.size(0)
        mask = mask.view(1, 1, mask_size, mask_size)
        
        # 应用卷积检测边界
        pad_tsm = nn.ZeroPad2d(mask_size // 2)(tsm.unsqueeze(1))
        score = F.conv2d(pad_tsm, mask).squeeze(1)  # [1, L_src, L_src]
        score = torch.diagonal(score, dim1=1, dim2=2)  # [1, L_src]
        
        # 计算阈值
        tau = score.mean(dim=1, keepdim=True).expand(-1, L_src)
        
        # 获取有效视频长度
        L_vid = torch.sum(src_vid_mask.to(torch.int), dim=1)  # [1]
        
        # 设置边界得分
        for i in range(bsz):
            score[i, 0] = 100  # 起始点
            if L_vid[i] > 1:
                score[i, L_vid[i]-1] = 100  # 结束点
        
        # 检测局部极大值
        score_r = torch.roll(score, shifts=1, dims=1)
        score_l = torch.roll(score, shifts=-1, dims=1)
        bnds = torch.where((score_r <= score) & (score_l <= score) & (tau <= score), 1., 0.)
        
        # 处理边界点（只处理第一个样本，因为batch_size=1）
        i = 0
        bnd_indices = torch.nonzero(bnds[i] == 1, as_tuple=False).squeeze(1)
        num_bnds = bnd_indices.size(0)
        
        spans_info = {
            'video_length': L_vid[i].item(),
            'boundary_points': bnd_indices.cpu().numpy().tolist(),
            'num_boundaries': num_bnds,
            'spans': [],
            'num_spans': 0
        }
        
        if num_bnds >= 2:
            # 处理相邻边界点对
            prev_indices = bnd_indices[:-1]
            curr_indices = bnd_indices[1:]
            
            # 计算跨度宽度
            span_widths = curr_indices - prev_indices
            
            # 筛选有效跨度
            valid_mask = span_widths <= L_vid[i] * self.span_width_threshold
            
            # 提取有效边界点对
            valid_prev = prev_indices[valid_mask]
            valid_curr = curr_indices[valid_mask]
            
            if len(valid_prev) > 0:
                valid_spans = torch.stack([valid_prev, valid_curr], dim=1)
                
                # 计算归一化的中心点和宽度
                centers = ((valid_spans[:, 0] + valid_spans[:, 1]) / 2 / L_vid[i].float())
                widths = ((valid_spans[:, 1] - valid_spans[:, 0]) / L_vid[i].float())
                
                # 限制span数量
                num_spans = min(valid_spans.size(0), max_spans)
                
                # 存储span信息
                for j in range(num_spans):
                    span_info = {
                        'start_idx': valid_spans[j, 0].item(),
                        'end_idx': valid_spans[j, 1].item(),
                        'length': valid_spans[j, 1].item() - valid_spans[j, 0].item() + 1,
                        'center_normalized': centers[j].item(),
                        'width_normalized': widths[j].item(),
                        'relative_length': (valid_spans[j, 1].item() - valid_spans[j, 0].item() + 1) / L_vid[i].item()
                    }
                    spans_info['spans'].append(span_info)
                
                spans_info['num_spans'] = num_spans
            else:
                # 使用默认span（整个视频）
                spans_info['spans'] = [{
                    'start_idx': 0,
                    'end_idx': L_vid[i].item() - 1,
                    'length': L_vid[i].item(),
                    'center_normalized': 0.5,
                    'width_normalized': 1.0,
                    'relative_length': 1.0
                }]
                spans_info['num_spans'] = 1
        else:
            # 使用默认span
            spans_info['spans'] = [{
                'start_idx': 0,
                'end_idx': L_vid[i].item() - 1,
                'length': L_vid[i].item(),
                'center_normalized': 0.5,
                'width_normalized': 1.0,
                'relative_length': 1.0
            }]
            spans_info['num_spans'] = 1
        
        return spans_info
    
    def analyze_directory(self, data_dir, dataset_name="unknown", file_pattern="*.pt", max_files=None):
        """
        分析整个目录中的特征文件
        
        Args:
            data_dir: 数据目录路径
            dataset_name: 数据集名称
            file_pattern: 文件匹配模式
            max_files: 最大处理文件数
        """
        print(f"\n开始分析数据集: {dataset_name}")
        print(f"数据目录: {data_dir}")
        print(f"文件模式: {file_pattern}")
        
        # 查找所有匹配的文件
        file_paths = glob.glob(os.path.join(data_dir, file_pattern))
        if not file_paths:
            file_paths = glob.glob(os.path.join(data_dir, "**", file_pattern), recursive=True)
        
        if not file_paths:
            print(f"在 {data_dir} 中没有找到匹配 {file_pattern} 的文件")
            return None
        
        if max_files:
            file_paths = file_paths[:max_files]
        
        print(f"找到 {len(file_paths)} 个文件")
        
        # 初始化统计数据
        all_spans_info = []
        span_counts = []
        span_lengths = []
        span_relative_lengths = []
        video_lengths = []
        
        # 处理每个文件
        success_count = 0
        failed_count = 0
        
        for i, file_path in enumerate(tqdm(file_paths, desc="处理文件")):
            self.error_stats['total_files'] += 1
            
            # 加载特征
            features, mask, valid_length = self.load_feature_file(file_path)
            
            if features is None:
                failed_count += 1
                self.error_stats['failed_files'] += 1
                continue
            
            try:
                # 生成伪事件
                spans_info = self.generate_pseudo_event_for_single_video(features, mask)
                spans_info['file_path'] = file_path
                spans_info['file_index'] = i
                
                all_spans_info.append(spans_info)
                
                # 收集统计数据
                span_counts.append(spans_info['num_spans'])
                video_lengths.append(spans_info['video_length'])
                
                for span in spans_info['spans']:
                    span_lengths.append(span['length'])
                    span_relative_lengths.append(span['relative_length'])
                
                success_count += 1
                self.error_stats['successful_files'] += 1
                
            except Exception as e:
                failed_count += 1
                self.error_stats['failed_files'] += 1
                
                # 记录错误类型
                error_type = type(e).__name__
                self.error_stats['error_types'][error_type] += 1
                
                # 只在前10个错误时打印详细信息，避免刷屏
                if failed_count <= 10:
                    print(f"处理文件失败 {file_path}: {e}")
                elif failed_count == 11:
                    print(f"... 更多错误将被静默记录（已失败{failed_count-1}个文件）")
                
                continue
        
        print(f"处理完成: 成功 {success_count}/{len(file_paths)} 个文件")
        if failed_count > 0:
            print(f"失败文件数: {failed_count}")
            print("错误类型统计:")
            for error_type, count in self.error_stats['error_types'].items():
                print(f"  {error_type}: {count}")
        
        if success_count == 0:
            print("没有成功处理的文件")
            return None
        
        # 计算统计摘要
        summary = self._compute_summary_stats(
            span_counts, span_lengths, span_relative_lengths, video_lengths
        )
        
        # 存储结果
        self.results[dataset_name] = {
            'summary': summary,
            'detailed': all_spans_info,
            'config': {
                'max_event_spans': self.max_event_spans,
                'span_width_threshold': self.span_width_threshold
            }
        }
        
        # 打印摘要
        self._print_summary(dataset_name, summary)
        
        return summary
    
    def analyze_file_list(self, file_list, dataset_name="unknown"):
        """
        分析文件列表
        
        Args:
            file_list: 文件路径列表
            dataset_name: 数据集名称
        """
        print(f"\n开始分析数据集: {dataset_name}")
        print(f"文件数量: {len(file_list)}")
        
        all_spans_info = []
        span_counts = []
        span_lengths = []
        span_relative_lengths = []
        video_lengths = []
        
        success_count = 0
        for i, file_path in enumerate(tqdm(file_list, desc="处理文件")):
            features, mask, valid_length = self.load_feature_file(file_path)
            
            if features is None:
                continue
            
            try:
                spans_info = self.generate_pseudo_event_for_single_video(features, mask)
                spans_info['file_path'] = file_path
                spans_info['file_index'] = i
                
                all_spans_info.append(spans_info)
                
                span_counts.append(spans_info['num_spans'])
                video_lengths.append(spans_info['video_length'])
                
                for span in spans_info['spans']:
                    span_lengths.append(span['length'])
                    span_relative_lengths.append(span['relative_length'])
                
                success_count += 1
                
            except Exception as e:
                print(f"处理文件失败 {file_path}: {e}")
                continue
        
        print(f"成功处理 {success_count}/{len(file_list)} 个文件")
        
        if success_count == 0:
            return None
        
        summary = self._compute_summary_stats(
            span_counts, span_lengths, span_relative_lengths, video_lengths
        )
        
        self.results[dataset_name] = {
            'summary': summary,
            'detailed': all_spans_info,
            'config': {
                'max_event_spans': self.max_event_spans,
                'span_width_threshold': self.span_width_threshold
            }
        }
        
        self._print_summary(dataset_name, summary)
        return summary
    
    def _compute_summary_stats(self, span_counts, span_lengths, span_relative_lengths, video_lengths):
        """计算统计摘要"""
        summary = {
            'total_videos': len(span_counts),
            'total_spans': sum(span_counts),
            'span_count_stats': {
                'mean': np.mean(span_counts),
                'median': np.median(span_counts),
                'std': np.std(span_counts),
                'min': np.min(span_counts),
                'max': np.max(span_counts),
                'percentiles': {
                    'p25': np.percentile(span_counts, 25),
                    'p50': np.percentile(span_counts, 50),
                    'p75': np.percentile(span_counts, 75),
                    'p90': np.percentile(span_counts, 90),
                    'p95': np.percentile(span_counts, 95),
                    'p99': np.percentile(span_counts, 99)
                }
            },
            'video_length_stats': {
                'mean': np.mean(video_lengths),
                'median': np.median(video_lengths),
                'std': np.std(video_lengths),
                'min': np.min(video_lengths),
                'max': np.max(video_lengths)
            }
        }
        
        if span_lengths:
            summary['span_length_stats'] = {
                'mean': np.mean(span_lengths),
                'median': np.median(span_lengths),
                'std': np.std(span_lengths),
                'min': np.min(span_lengths),
                'max': np.max(span_lengths),
                'percentiles': {
                    'p25': np.percentile(span_lengths, 25),
                    'p50': np.percentile(span_lengths, 50),
                    'p75': np.percentile(span_lengths, 75),
                    'p90': np.percentile(span_lengths, 90),
                    'p95': np.percentile(span_lengths, 95)
                }
            }
        
        if span_relative_lengths:
            summary['span_relative_length_stats'] = {
                'mean': np.mean(span_relative_lengths),
                'median': np.median(span_relative_lengths),
                'std': np.std(span_relative_lengths)
            }
        
        return summary
    
    def _print_summary(self, dataset_name, summary):
        """打印统计摘要"""
        print(f"\n{'='*60}")
        print(f"数据集统计摘要: {dataset_name}")
        print(f"{'='*60}")
        print(f"总视频数: {summary['total_videos']:,}")
        print(f"总span数: {summary['total_spans']:,}")
        
        span_stats = summary['span_count_stats']
        print(f"\nSpan数量统计:")
        print(f"  平均每视频: {span_stats['mean']:.2f}")
        print(f"  中位数: {span_stats['median']:.2f}")
        print(f"  标准差: {span_stats['std']:.2f}")
        print(f"  范围: [{span_stats['min']:.0f}, {span_stats['max']:.0f}]")
        
        print(f"\nSpan数量分布:")
        for p, v in span_stats['percentiles'].items():
            print(f"  {p}: {v:.1f}")
        
        # 推荐配置
        recommended_max = int(span_stats['percentiles']['p95']) + 1
        print(f"\n💡 推荐的max_event_spans设置: {recommended_max}")
        print(f"   (基于95分位数 + 1的安全边界)")
        
        if span_stats['mean'] <= 3:
            print(f"💡 数据集复杂度: 简单 (平均{span_stats['mean']:.1f}个spans)")
            print(f"💡 建议semantic_weight: 0.01-0.1")
        elif span_stats['mean'] <= 8:
            print(f"💡 数据集复杂度: 中等 (平均{span_stats['mean']:.1f}个spans)")
            print(f"💡 建议semantic_weight: 0.3-0.5")
        else:
            print(f"💡 数据集复杂度: 复杂 (平均{span_stats['mean']:.1f}个spans)")
            print(f"💡 建议semantic_weight: 0.6-0.8")
    
    def plot_statistics(self, dataset_name, save_path=None):
        """绘制统计图表"""
        if dataset_name not in self.results:
            print(f"数据集 {dataset_name} 不存在")
            return
        
        detailed_data = self.results[dataset_name]['detailed']
        
        # 提取数据
        span_counts = [item['num_spans'] for item in detailed_data]
        span_lengths = []
        video_lengths = [item['video_length'] for item in detailed_data]
        
        for item in detailed_data:
            for span in item['spans']:
                span_lengths.append(span['length'])
        
        # 创建图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Span数量分布
        axes[0, 0].hist(span_counts, bins=max(1, max(span_counts)), alpha=0.7, edgecolor='black')
        axes[0, 0].set_title(f'{dataset_name}: Span数量分布')
        axes[0, 0].set_xlabel('每视频Span数量')
        axes[0, 0].set_ylabel('频次')
        axes[0, 0].axvline(np.mean(span_counts), color='red', linestyle='--', 
                          label=f'均值: {np.mean(span_counts):.2f}')
        axes[0, 0].legend()
        
        # 2. Span数量箱线图
        axes[0, 1].boxplot(span_counts)
        axes[0, 1].set_title('Span数量箱线图')
        axes[0, 1].set_ylabel('Span数量')
        
        # 3. 视频长度分布
        axes[0, 2].hist(video_lengths, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 2].set_title('视频长度分布')
        axes[0, 2].set_xlabel('视频长度（帧数）')
        axes[0, 2].set_ylabel('频次')
        
        # 4. Span长度分布
        if span_lengths:
            axes[1, 0].hist(span_lengths, bins=50, alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Span长度分布')
            axes[1, 0].set_xlabel('Span长度（帧数）')
            axes[1, 0].set_ylabel('频次')
            axes[1, 0].axvline(np.mean(span_lengths), color='red', linestyle='--',
                              label=f'均值: {np.mean(span_lengths):.2f}')
            axes[1, 0].legend()
        
        # 5. Span数量累积分布
        sorted_counts = np.sort(span_counts)
        y = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
        axes[1, 1].plot(sorted_counts, y, linewidth=2)
        axes[1, 1].set_title('Span数量累积分布')
        axes[1, 1].set_xlabel('Span数量')
        axes[1, 1].set_ylabel('累积概率')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 添加95分位数线
        p95 = np.percentile(span_counts, 95)
        axes[1, 1].axvline(p95, color='red', linestyle='--', 
                          label=f'P95: {p95:.1f}')
        axes[1, 1].legend()
        
        # 6. Span数量vs视频长度散点图
        axes[1, 2].scatter(video_lengths, span_counts, alpha=0.6)
        axes[1, 2].set_title('Span数量 vs 视频长度')
        axes[1, 2].set_xlabel('视频长度（帧数）')
        axes[1, 2].set_ylabel('Span数量')
        
        # 添加趋势线
        z = np.polyfit(video_lengths, span_counts, 1)
        p = np.poly1d(z)
        axes[1, 2].plot(sorted(video_lengths), p(sorted(video_lengths)), "r--", alpha=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        plt.show()
    
    def compare_datasets(self, dataset_names=None):
        """对比多个数据集"""
        if dataset_names is None:
            dataset_names = list(self.results.keys())
        
        print(f"\n{'='*80}")
        print("数据集对比")
        print(f"{'='*80}")
        
        comparison_data = []
        for name in dataset_names:
            if name in self.results:
                stats = self.results[name]['summary']['span_count_stats']
                comparison_data.append({
                    'Dataset': name,
                    'Total Videos': self.results[name]['summary']['total_videos'],
                    'Avg Spans': f"{stats['mean']:.2f}",
                    'Median Spans': f"{stats['median']:.2f}",
                    'P95 Spans': f"{stats['percentiles']['p95']:.1f}",
                    'Max Spans': f"{stats['max']:.0f}",
                    'Recommended max_event_spans': int(stats['percentiles']['p95']) + 1
                })
        
        # 使用简单的表格打印
        if comparison_data:
            # 打印表头
            headers = list(comparison_data[0].keys())
            col_widths = [max(len(str(row[col])) for row in [{'Dataset': 'Dataset'}] + comparison_data) 
                         for col in headers]
            
            # 打印分隔线
            print('-' * (sum(col_widths) + len(headers) * 3 - 1))
            
            # 打印表头
            header_row = ' | '.join(h.ljust(w) for h, w in zip(headers, col_widths))
            print(header_row)
            print('-' * (sum(col_widths) + len(headers) * 3 - 1))
            
            # 打印数据行
            for row in comparison_data:
                data_row = ' | '.join(str(row[h]).ljust(w) for h, w in zip(headers, col_widths))
                print(data_row)
            
            print('-' * (sum(col_widths) + len(headers) * 3 - 1))
    
    def save_results(self, save_path):
        """保存分析结果"""
        # 转换numpy类型
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        def recursive_convert(data):
            if isinstance(data, dict):
                return {k: recursive_convert(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [recursive_convert(item) for item in data]
            else:
                return convert_numpy(data)
        
        converted_results = recursive_convert(self.results)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(converted_results, f, indent=2, ensure_ascii=False)
        
        print(f"分析结果已保存到: {save_path}")

def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description='Standalone Span Analyzer')
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录路径')
    parser.add_argument('--dataset_name', type=str, default='unknown', help='数据集名称')
    parser.add_argument('--file_pattern', type=str, default='*.pt', help='文件匹配模式')
    parser.add_argument('--max_files', type=int, default=None, help='最大处理文件数')
    parser.add_argument('--max_event_spans', type=int, default=20, help='最大事件span数')
    parser.add_argument('--span_width_threshold', type=float, default=0.5, help='span宽度阈值')
    parser.add_argument('--save_results', type=str, default=None, help='结果保存路径')
    parser.add_argument('--save_plot', type=str, default=None, help='图表保存路径')
    parser.add_argument('--device', type=str, default='cuda', help='计算设备')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = StandaloneSpanAnalyzer(
        max_event_spans=args.max_event_spans,
        span_width_threshold=args.span_width_threshold,
        device=args.device
    )
    
    # 分析数据集
    summary = analyzer.analyze_directory(
        data_dir=args.data_dir,
        dataset_name=args.dataset_name,
        file_pattern=args.file_pattern,
        max_files=args.max_files
    )
    
    if summary:
        # 绘制图表
        analyzer.plot_statistics(args.dataset_name, save_path=args.save_plot)
        
        # 保存结果
        if args.save_results:
            analyzer.save_results(args.save_results)

if __name__ == "__main__":
    main()

# 使用示例代码
def example_usage():
    """使用示例"""
    
    # 示例1: 分析单个目录
    analyzer = StandaloneSpanAnalyzer(
        max_event_spans=20,
        span_width_threshold=0.5,
        device='cuda'
    )
    
    # 分析QVHighlights数据集
    qvhighlights_summary = analyzer.analyze_directory(
        data_dir="/path/to/qvhighlights/features",
        dataset_name="qvhighlights",
        file_pattern="*.pt",
        max_files=1000  # 限制处理1000个文件进行快速测试
    )
    
    # 分析其他数据集
    other_summary = analyzer.analyze_directory(
        data_dir="/path/to/other_dataset/features", 
        dataset_name="other_dataset",
        file_pattern="*.npy"
    )
    
    # 对比数据集
    analyzer.compare_datasets(["qvhighlights", "other_dataset"])
    
    # 绘制QVHighlights的统计图
    analyzer.plot_statistics("qvhighlights", save_path="qvhighlights_analysis.png")
    
    # 保存所有结果
    analyzer.save_results("span_analysis_results.json")
    
    # 示例2: 分析文件列表
    file_list = [
        "/path/to/video1_features.pt",
        "/path/to/video2_features.pt",
        "/path/to/video3_features.npy"
    ]
    
    analyzer.analyze_file_list(file_list, "custom_dataset")

# 快速分析脚本
def quick_analysis(data_dir, dataset_name="test", max_files=100):
    """
    快速分析脚本 - 适合初次探索
    
    Args:
        data_dir: 特征文件目录
        dataset_name: 数据集名称
        max_files: 最大处理文件数
    """
    print(f"快速分析: {dataset_name}")
    print(f"数据目录: {data_dir}")
    
    analyzer = StandaloneSpanAnalyzer(
        max_event_spans=20,
        span_width_threshold=0.5
    )
    
    # 尝试不同的文件格式
    patterns = ["*.pt", "*.pth", "*.npy", "*.npz"]
    
    for pattern in patterns:
        print(f"\n尝试文件模式: {pattern}")
        summary = analyzer.analyze_directory(
            data_dir=data_dir,
            dataset_name=f"{dataset_name}_{pattern.replace('*', '').replace('.', '')}",
            file_pattern=pattern,
            max_files=max_files
        )
        
        if summary and summary['total_videos'] > 0:
            print(f"✅ 成功处理 {summary['total_videos']} 个文件")
            break
        else:
            print(f"❌ 没有找到匹配的文件")
    
    # 如果有结果，显示图表
    if analyzer.results:
        dataset_names = list(analyzer.results.keys())
        if dataset_names:
            analyzer.plot_statistics(dataset_names[0])
            return analyzer
    
    print("没有成功分析任何文件")
    return None

# 批量分析多个数据集
def batch_analysis(dataset_configs):
    """
    批量分析多个数据集
    
    Args:
        dataset_configs: 数据集配置列表
            例如: [
                {
                    'name': 'qvhighlights',
                    'data_dir': '/path/to/qvhighlights',
                    'pattern': '*.pt',
                    'max_files': 1000
                },
                {
                    'name': 'other_dataset',
                    'data_dir': '/path/to/other',
                    'pattern': '*.npy',
                    'max_files': 500
                }
            ]
    """
    analyzer = StandaloneSpanAnalyzer(max_event_spans=20, span_width_threshold=0.5)
    
    successful_datasets = []
    
    for config in dataset_configs:
        print(f"\n处理数据集: {config['name']}")
        
        summary = analyzer.analyze_directory(
            data_dir=config['data_dir'],
            dataset_name=config['name'],
            file_pattern=config.get('pattern', '*.pt'),
            max_files=config.get('max_files', None)
        )
        
        if summary and summary['total_videos'] > 0:
            successful_datasets.append(config['name'])
    
    if len(successful_datasets) > 1:
        print(f"\n对比分析 {len(successful_datasets)} 个数据集")
        analyzer.compare_datasets(successful_datasets)
    
    # 为每个成功的数据集生成图表
    for dataset_name in successful_datasets:
        analyzer.plot_statistics(
            dataset_name, 
            save_path=f"{dataset_name}_span_analysis.png"
        )
    
    # 保存所有结果
    analyzer.save_results("batch_analysis_results.json")
    
    return analyzer

# 数据格式转换辅助函数
def convert_features_format(input_dir, output_dir, input_format="npy", output_format="pt"):
    """
    转换特征文件格式
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录  
        input_format: 输入格式 (npy, pt, npz)
        output_format: 输出格式 (npy, pt)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    input_files = glob.glob(os.path.join(input_dir, f"*.{input_format}"))
    
    for input_file in tqdm(input_files, desc="转换文件格式"):
        try:
            # 加载数据
            if input_format == "npy":
                data = np.load(input_file)
            elif input_format == "pt" or input_format == "pth":
                data = torch.load(input_file, map_location='cpu')
                if isinstance(data, torch.Tensor):
                    data = data.numpy()
            elif input_format == "npz":
                npz_data = np.load(input_file)
                data = npz_data['features']  # 假设特征存储在'features'键下
            
            # 保存数据
            filename = os.path.basename(input_file)
            name_without_ext = os.path.splitext(filename)[0]
            
            if output_format == "npy":
                output_file = os.path.join(output_dir, f"{name_without_ext}.npy")
                np.save(output_file, data)
            elif output_format == "pt":
                output_file = os.path.join(output_dir, f"{name_without_ext}.pt")
                torch.save(torch.from_numpy(data), output_file)
                
        except Exception as e:
            print(f"转换失败 {input_file}: {e}")

# 生成配置文件模板
def generate_config_template(save_path="analysis_config.json"):
    """生成分析配置文件模板"""
    
    template = {
        "datasets": [
            {
                "name": "qvhighlights_train",
                "data_dir": "/path/to/qvhighlights/train/features",
                "pattern": "*.pt",
                "max_files": 1000,
                "description": "QVHighlights训练集"
            },
            {
                "name": "qvhighlights_val", 
                "data_dir": "/path/to/qvhighlights/val/features",
                "pattern": "*.pt",
                "max_files": 500,
                "description": "QVHighlights验证集"
            },
            {
                "name": "other_dataset",
                "data_dir": "/path/to/other/features",
                "pattern": "*.npy", 
                "max_files": null,
                "description": "其他数据集"
            }
        ],
        "analyzer_config": {
            "max_event_spans": 20,
            "span_width_threshold": 0.5,
            "device": "cuda"
        },
        "output_config": {
            "save_results": "span_analysis_results.json",
            "save_plots": true,
            "plot_dir": "plots"
        }
    }
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(template, f, indent=2, ensure_ascii=False)
    
    print(f"配置文件模板已保存到: {save_path}")
    print("请修改配置文件中的路径和参数，然后使用 run_from_config() 函数")

def run_from_config(config_path="analysis_config.json"):
    """从配置文件运行分析"""
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 创建分析器
    analyzer_config = config.get('analyzer_config', {})
    analyzer = StandaloneSpanAnalyzer(**analyzer_config)
    
    # 处理每个数据集
    successful_datasets = []
    for dataset_config in config['datasets']:
        print(f"\n处理数据集: {dataset_config['name']}")
        
        summary = analyzer.analyze_directory(
            data_dir=dataset_config['data_dir'],
            dataset_name=dataset_config['name'],
            file_pattern=dataset_config.get('pattern', '*.pt'),
            max_files=dataset_config.get('max_files', None)
        )
        
        if summary and summary['total_videos'] > 0:
            successful_datasets.append(dataset_config['name'])
    
    # 对比分析
    if len(successful_datasets) > 1:
        analyzer.compare_datasets(successful_datasets)
    
    # 输出结果
    output_config = config.get('output_config', {})
    
    if output_config.get('save_results'):
        analyzer.save_results(output_config['save_results'])
    
    if output_config.get('save_plots'):
        plot_dir = output_config.get('plot_dir', 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        for dataset_name in successful_datasets:
            plot_path = os.path.join(plot_dir, f"{dataset_name}_analysis.png")
            analyzer.plot_statistics(dataset_name, save_path=plot_path)
    
    return analyzer

# 打印使用说明
def print_usage():
    """打印使用说明"""
    
    usage_text = """
    🎯 独立Span分析工具使用指南
    
    📁 支持的文件格式:
       - PyTorch张量文件: *.pt, *.pth
       - NumPy数组文件: *.npy
       - NumPy压缩文件: *.npz (特征需存储在'features'键下)
    
    📊 期望的数据格式:
       - 每个文件包含一个视频的特征
       - 特征形状: [seq_len, hidden_dim]
       - 例如: [100, 256] 表示100帧，每帧256维特征
    
    🚀 快速开始:
    
    # 方法1: 命令行使用
    python span_analyzer.py --data_dir /path/to/features --dataset_name qvhighlights --file_pattern "*.pt"
    
    # 方法2: Python代码使用
    from span_analyzer import quick_analysis
    analyzer = quick_analysis("/path/to/features", "qvhighlights", max_files=100)
    
    # 方法3: 配置文件使用
    from span_analyzer import generate_config_template, run_from_config
    generate_config_template("my_config.json")  # 生成配置模板
    # 编辑配置文件...
    run_from_config("my_config.json")  # 运行分析
    
    📈 输出结果:
       - 控制台打印统计摘要
       - 可视化图表 (span分布、累积分布等)
       - JSON格式的详细结果
       - 推荐的max_event_spans和semantic_weight配置
    
    💡 推荐工作流程:
       1. 使用quick_analysis()快速探索小样本
       2. 根据结果调整参数
       3. 使用完整数据集进行分析
       4. 对比多个数据集
       5. 根据推荐配置优化模型
    """
    
    print(usage_text)

if __name__ == "__main__":
    # 如果没有命令行参数，打印使用说明
    import sys
    if len(sys.argv) == 1:
        print_usage()
    else:
        main()