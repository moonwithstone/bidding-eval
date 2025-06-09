import numpy as np
import pandas as pd
from scipy import stats
import random
from typing import List, Tuple, Dict
import matplotlib
import matplotlib.pyplot as plt
import os
from datetime import datetime
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading

# 强制指定macOS自带的中文字体，解决中文乱码
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class BiddingEvaluationModel:
    """六随机五区间投标评价模型（严格按图片公式实现）"""
    
    def __init__(self):
        self.bids = []
        self.weights = {}
        self.average_price = 0
        self.base_price = 0
        self.max_price = 0
        self.scores = {}
        self.valid_bids = []
        self.valid_indices = []
        
    def filter_valid_bids(self, bids: List[float], max_price: float) -> Tuple[List[float], List[int]]:
        """筛选有效报价，剔除低于最高限价20%的报价"""
        threshold = max_price * 0.8
        valid_bids = []
        valid_indices = []
        for idx, bid in enumerate(bids):
            if bid >= threshold:
                valid_bids.append(bid)
                valid_indices.append(idx)
        return valid_bids, valid_indices

    def remove_extremes(self, bids: List[float]) -> List[float]:
        """去除最低10%和最高15%的报价，四舍五入取整"""
        n = len(bids)
        if n < 10:
            return bids[:]
        sorted_bids = sorted(bids)
        low = int(round(n * 0.10))
        high = int(round(n * 0.15))
        remain = n - low - high
        if remain < 3:
            return sorted_bids
        return sorted_bids[low:n-high]

    def get_final_valid_bids(self, bids: List[float], max_price: float) -> Tuple[List[float], List[int]]:
        """综合筛选有效报价，返回最终有效报价及其原始索引"""
        valid_bids, valid_indices = self.filter_valid_bids(bids, max_price)
        W = len(valid_bids)
        if W < 3:
            return valid_bids, valid_indices
        elif W >= 10:
            sorted_pairs = sorted(zip(valid_bids, valid_indices))
            sorted_bids = [x[0] for x in sorted_pairs]
            sorted_indices = [x[1] for x in sorted_pairs]
            n = W
            low = int(round(n * 0.10))
            high = int(round(n * 0.15))
            remain = n - low - high
            if remain < 3:
                return sorted_bids, sorted_indices
            final_bids = sorted_bids[low:n-high]
            final_indices = sorted_indices[low:n-high]
            return final_bids, final_indices
        else:
            return valid_bids, valid_indices

    def calculate_quantiles(self, prices: List[float], n: int) -> Tuple[float, float]:
        """计算四分位数Q1和Q3，严格按插值法"""
        sorted_prices = sorted(prices)
        theta = 0.25 * (n + 1)
        gamma = 0.75 * (n + 1)
        def get_quantile(pos):
            if pos == int(pos):
                return sorted_prices[int(pos) - 1]
            i = int(np.floor(pos))
            t = pos - i
            if i <= 0:
                return sorted_prices[0]
            elif i >= n:
                return sorted_prices[-1]
            else:
                return sorted_prices[i-1] + t * (sorted_prices[i] - sorted_prices[i-1])
        q1 = get_quantile(theta)
        q3 = get_quantile(gamma)
        return q1, q3

    def calculate_average_price(self, prices: List[float], weights: Dict[str, float]) -> float:
        """计算平均价C，严格按图片公式"""
        n = len(prices)
        mu = sum(prices) / n
        sorted_prices = sorted(prices)
        if n % 2 == 0:
            median = (sorted_prices[n//2 - 1] + sorted_prices[n//2]) / 2
        else:
            median = sorted_prices[n//2]
        if n >= 5:
            q1, q3 = self.calculate_quantiles(prices, n)
            quartile_avg = (q1 + q3) / 2
        else:
            quartile_avg = 0
        geometric_mean = np.prod(prices) ** (1/n)
        if n >= 5:
            average_price = (weights['a'] * mu + 
                             weights['b'] * median + 
                             weights['c'] * quartile_avg + 
                             weights['d'] * geometric_mean)
        else:
            average_price = (weights['e'] * mu + 
                             weights['f'] * median + 
                             weights['g'] * geometric_mean)
        return average_price

    def calculate_base_price(self, max_price: float, average_price: float, c1: float = 0.5, r: float = 0.05) -> float:
        """计算评标基准价T，严格按图片公式"""
        return ((max_price * c1) + (average_price * (1 - c1))) * (1 - r)

    def calculate_deviation_rate(self, bid_price: float, base_price: float) -> float:
        """计算偏差率Di，严格按图片公式"""
        return (bid_price - base_price) / base_price * 100

    def calculate_score(self, deviation_rate: float) -> float:
        """五区间赋分，严格按图片公式（已修正）"""
        d = deviation_rate
        if d <= -10:
            return 80 - 1 * (abs(d) - 10)
        elif -10 < d < 0:
            return 100 - 2 * abs(d)
        elif 0 <= d <= 5:
            return 100 - 4 * d
        elif 5 < d <= 10:
            return 80 - 3 * (d - 5)
        else:  # d > 10
            return 65 - 2 * (d - 10)

    def generate_random_weights(self) -> Dict[str, float]:
        """生成满足约束条件的六个随机权重，严格按图片公式"""
        while True:
            u = [round(random.uniform(1.00, 9.00), 2) for _ in range(7)]
            s1 = u[0] + u[1] + u[2] + u[3]
            s2 = u[4] + u[5] + u[6]
            a = u[0] / s1
            b = u[1] / s1
            c = u[2] / s1
            d = u[3] / s1
            e = u[4] / s2
            f = u[5] / s2
            g = u[6] / s2
            if (abs(a + b + c + d - 1.0) < 1e-8 and abs(e + f + g - 1.0) < 1e-8 and
                0.30 <= a <= 0.45 and 0.05 <= b <= 0.20 and 0.05 <= c <= 0.20 and 0.30 <= d <= 0.45 and
                0.35 <= e <= 0.50 and 0.05 <= f <= 0.30 and 0.35 <= g <= 0.50):
                return {'a': a, 'b': b, 'c': c, 'd': d, 'e': e, 'f': f, 'g': g}

    def evaluate_bids(self, bids: List[float], max_price: float, companies: List[str] = None, 
                     c1: float = 0.5, r: float = 0.05, progress_callback=None) -> Dict:
        """完整的投标评价流程，严格按图片模型"""
        self.bids = bids
        self.max_price = max_price
        
        # 如果没有提供公司名称，使用默认名称
        if companies is None:
            companies = [f'投标人{i+1}' for i in range(len(bids))]
        
        if progress_callback:
            progress_callback("步骤1：开始有效报价筛选...")
        
        # 1. 有效报价筛选
        valid_bids, valid_indices = self.get_final_valid_bids(bids, max_price)
        self.valid_bids = valid_bids
        self.valid_indices = valid_indices
        
        if progress_callback:
            progress_callback(f"步骤1完成：筛选出 {len(valid_bids)} 个有效报价")
            progress_callback(f"有效报价索引: {valid_indices}")
            progress_callback(f"有效报价: {[f'{bid:,.2f}' for bid in valid_bids]}")
        
        # 2. 生成随机权重
        if progress_callback:
            progress_callback("步骤2：生成随机权重...")
        
        self.weights = self.generate_random_weights()
        
        if progress_callback:
            progress_callback("步骤2完成：随机权重生成")
            for key, value in self.weights.items():
                progress_callback(f"权重 {key}: {value:.4f}")
        
        # 3. 计算平均价
        if progress_callback:
            progress_callback("步骤3：计算平均价...")
        
        self.average_price = self.calculate_average_price(valid_bids, self.weights)
        
        if progress_callback:
            n = len(valid_bids)
            mu = sum(valid_bids) / n
            sorted_prices = sorted(valid_bids)
            if n % 2 == 0:
                median = (sorted_prices[n//2 - 1] + sorted_prices[n//2]) / 2
            else:
                median = sorted_prices[n//2]
            geometric_mean = np.prod(valid_bids) ** (1/n)
            
            progress_callback(f"算术平均数: {mu:,.2f}")
            progress_callback(f"中位数: {median:,.2f}")
            progress_callback(f"几何平均数: {geometric_mean:,.2f}")
            
            if n >= 5:
                q1, q3 = self.calculate_quantiles(valid_bids, n)
                quartile_avg = (q1 + q3) / 2
                progress_callback(f"Q1: {q1:,.2f}")
                progress_callback(f"Q3: {q3:,.2f}")
                progress_callback(f"四分位平均数: {quartile_avg:,.2f}")
                progress_callback(f"平均价计算公式: C = {self.weights['a']:.4f}×{mu:,.2f} + {self.weights['b']:.4f}×{median:,.2f} + {self.weights['c']:.4f}×{quartile_avg:,.2f} + {self.weights['d']:.4f}×{geometric_mean:,.2f}")
            else:
                progress_callback(f"平均价计算公式: C = {self.weights['e']:.4f}×{mu:,.2f} + {self.weights['f']:.4f}×{median:,.2f} + {self.weights['g']:.4f}×{geometric_mean:,.2f}")
            
            progress_callback(f"步骤3完成：平均价 C = {self.average_price:,.2f}")
        
        # 4. 计算评标基准价
        if progress_callback:
            progress_callback("步骤4：计算评标基准价...")
        
        self.base_price = self.calculate_base_price(max_price, self.average_price, c1, r)
        
        if progress_callback:
            progress_callback(f"评标基准价计算公式: T = ({max_price:,.2f}×{c1} + {self.average_price:,.2f}×{1-c1})×{1-r}")
            progress_callback(f"步骤4完成：评标基准价 T = {self.base_price:,.2f}")
        
        # 5. 计算各投标单位得分
        if progress_callback:
            progress_callback("步骤5：计算各投标单位得分...")
        
        results = []
        for i, bid in enumerate(bids):
            company_name = companies[i] if i < len(companies) else f'投标人{i+1}'
            if i in valid_indices:
                deviation_rate = self.calculate_deviation_rate(bid, self.base_price)
                score = self.calculate_score(deviation_rate)
                if progress_callback:
                    progress_callback(f"{company_name}: 报价{bid:,.2f}, 偏差率{deviation_rate:.2f}%, 得分{score:.2f}")
            else:
                deviation_rate = None
                score = None
                if progress_callback:
                    progress_callback(f"{company_name}: 报价{bid:,.2f}, 无效报价")
            
            results.append({
                '投标单位': company_name,
                '投标报价': bid,
                '偏差率(%)': round(deviation_rate, 2) if deviation_rate is not None else '',
                '得分': round(score, 2) if score is not None else '无效'
            })
        
        # 按得分降序排列（无效报价排最后）
        results.sort(key=lambda x: (x['得分'] if isinstance(x['得分'], (int, float)) else -999), reverse=True)
        
        if progress_callback:
            progress_callback("步骤5完成：所有投标单位得分计算完毕")
            progress_callback("评价流程全部完成！")
        
        return {
            '随机权重': self.weights,
            '平均价': round(self.average_price, 2),
            '评标基准价': round(self.base_price, 2),
            '评价结果': results
        }
    
    def import_data_from_excel(self, file_path: str) -> Tuple[List[float], float, Dict]:
        """从Excel文件导入投标数据"""
        try:
            df = pd.read_excel(file_path)
            df = df.dropna().drop_duplicates()
            
            # 自动识别列名（支持中英文）
            bid_cols = ['投标报价', '报价', 'bid_price', 'price', '金额']
            company_cols = ['投标单位', '公司名称', 'company', 'bidder', '单位名称']
            max_price_cols = ['最高限价', '控制价', 'max_price', 'control_price']
            
            # 查找投标报价列
            bid_col = None
            for col in df.columns:
                if any(keyword in str(col).lower() for keyword in [c.lower() for c in bid_cols]):
                    bid_col = col
                    break
            
            if bid_col is None:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    bid_col = numeric_cols[0]
                else:
                    raise ValueError("未找到投标报价数据列")
            
            bids = df[bid_col].tolist()
            
            # 查找公司名称列
            company_col = None
            for col in df.columns:
                if any(keyword in str(col).lower() for keyword in [c.lower() for c in company_cols]):
                    company_col = col
                    break
            
            if company_col:
                companies = df[company_col].tolist()
            else:
                companies = [f"投标人{i+1}" for i in range(len(bids))]
            
            # 查找最高限价
            max_price = None
            for col in df.columns:
                if any(keyword in str(col).lower() for keyword in [c.lower() for c in max_price_cols]):
                    max_price = df[col].iloc[0] if not df[col].empty else None
                    break
            
            if max_price is None:
                max_price = max(bids) * 1.1
            
            extra_info = {
                'companies': companies,
                'data_source': file_path,
                'import_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_bidders': len(bids)
            }
            
            return bids, max_price, extra_info
            
        except Exception as e:
            raise Exception(f"导入Excel文件失败: {str(e)}")


class BiddingEvaluationGUI:
    """投标评价工具GUI界面"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("投标评价工具 - 六随机五区间评价模型")
        self.root.geometry("1000x700")
        
        self.model = BiddingEvaluationModel()
        self.results = None
        self.extra_info = None
        
        self.setup_ui()
    
    def setup_ui(self):
        """设置UI界面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 文件选择区域
        file_frame = ttk.LabelFrame(main_frame, text="数据导入", padding="10")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(file_frame, text="选择Excel文件:").grid(row=0, column=0, sticky=tk.W)
        self.file_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path_var, width=50).grid(row=0, column=1, padx=(5, 5))
        ttk.Button(file_frame, text="浏览", command=self.browse_file).grid(row=0, column=2)
        ttk.Button(file_frame, text="开始评价", command=self.start_evaluation).grid(row=0, column=3, padx=(10, 0))
        
        # 参数设置区域
        param_frame = ttk.LabelFrame(main_frame, text="参数设置", padding="10")
        param_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(param_frame, text="c1 (最高限价权重):").grid(row=0, column=0, sticky=tk.W)
        self.c1_var = tk.DoubleVar(value=0.5)
        ttk.Entry(param_frame, textvariable=self.c1_var, width=10).grid(row=0, column=1, padx=(5, 20))
        
        ttk.Label(param_frame, text="r (下浮率):").grid(row=0, column=2, sticky=tk.W)
        self.r_var = tk.DoubleVar(value=0.05)
        ttk.Entry(param_frame, textvariable=self.r_var, width=10).grid(row=0, column=3, padx=(5, 0))
        
        # 计算过程显示区域
        process_frame = ttk.LabelFrame(main_frame, text="计算过程", padding="10")
        process_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        self.process_text = scrolledtext.ScrolledText(process_frame, width=60, height=20)
        self.process_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 结果显示区域
        result_frame = ttk.LabelFrame(main_frame, text="评价结果", padding="10")
        result_frame.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # 创建表格
        columns = ('排名', '投标单位', '投标报价', '偏差率(%)', '得分')
        self.result_tree = ttk.Treeview(result_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.result_tree.heading(col, text=col)
            self.result_tree.column(col, width=100, anchor='center')
        
        self.result_tree.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 滚动条
        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.result_tree.yview)
        scrollbar.grid(row=0, column=2, sticky=(tk.N, tk.S))
        self.result_tree.configure(yscrollcommand=scrollbar.set)
        
        # 导出按钮
        export_frame = ttk.Frame(result_frame)
        export_frame.grid(row=1, column=0, columnspan=2, pady=(10, 0))
        
        ttk.Button(export_frame, text="导出Excel", command=self.export_excel).grid(row=0, column=0, padx=(0, 5))
        # ttk.Button(export_frame, text="导出JSON", command=self.export_json).grid(row=0, column=1, padx=(5, 0))
        
        # 配置权重
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        process_frame.columnconfigure(0, weight=1)
        process_frame.rowconfigure(0, weight=1)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
    
    def browse_file(self):
        """浏览文件"""
        file_path = filedialog.askopenfilename(
            title="选择Excel文件",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        if file_path:
            self.file_path_var.set(file_path)
    
    def update_process(self, message):
        """更新计算过程显示"""
        self.process_text.insert(tk.END, message + "\n")
        self.process_text.see(tk.END)
        self.root.update()
    
    def start_evaluation(self):
        """开始评价流程"""
        if not self.file_path_var.get():
            messagebox.showerror("错误", "请先选择Excel文件")
            return
        
        # 清空之前的结果
        self.process_text.delete(1.0, tk.END)
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
        
        def evaluation_thread():
            try:
                self.update_process("开始导入Excel数据...")
                
                # 导入数据
                bids, max_price, extra_info = self.model.import_data_from_excel(self.file_path_var.get())
                self.extra_info = extra_info
                
                self.update_process(f"数据导入完成！")
                self.update_process(f"投标单位数量: {len(bids)}")
                self.update_process(f"投标单位: {extra_info['companies']}")
                self.update_process(f"最高限价: {max_price:,.2f}")
                self.update_process(f"投标报价: {[f'{bid:,.2f}' for bid in bids]}")
                self.update_process("-" * 50)
                
                # 开始评价
                self.results = self.model.evaluate_bids(
                    bids, max_price, 
                    extra_info['companies'],  # 传入公司名称列表
                    self.c1_var.get(), 
                    self.r_var.get(),
                    self.update_process
                )
                
                # 显示结果
                self.root.after(0, self.display_results)
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("错误", str(e)))
        
        # 在后台线程中运行评价
        thread = threading.Thread(target=evaluation_thread)
        thread.daemon = True
        thread.start()
    
    def display_results(self):
        """显示评价结果"""
        if not self.results:
            return
        
        # 在过程文本中显示汇总信息
        self.update_process("=" * 50)
        self.update_process(f"最终结果汇总:")
        self.update_process(f"平均价: {self.results['平均价']:,.2f}")
        self.update_process(f"评标基准价: {self.results['评标基准价']:,.2f}")
        
        # 在表格中显示结果
        for idx, result in enumerate(self.results['评价结果'], 1):
            self.result_tree.insert('', 'end', values=(
                idx,
                result['投标单位'],
                f"{result['投标报价']:,.2f}",
                result['偏差率(%)'],
                result['得分']
            ))
    
    def export_excel(self):
        """导出到Excel"""
        if not self.results:
            messagebox.showwarning("警告", "没有可导出的结果")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="保存Excel文件",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # 创建DataFrame
                df = pd.DataFrame(self.results['评价结果'])
                
                # 创建Excel写入器
                with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                    # 写入主要结果
                    df.to_excel(writer, sheet_name='评价结果', index=False)
                    
                    # 写入参数信息
                    params_data = {
                        '参数': ['平均价', '评标基准价', '最高限价', 'c1参数', 'r参数'],
                        '数值': [
                            self.results['平均价'],
                            self.results['评标基准价'],
                            self.model.max_price,
                            self.c1_var.get(),
                            self.r_var.get()
                        ]
                    }
                    params_df = pd.DataFrame(params_data)
                    params_df.to_excel(writer, sheet_name='参数信息', index=False)
                    
                    # 写入权重信息
                    weights_data = {
                        '权重': list(self.results['随机权重'].keys()),
                        '数值': list(self.results['随机权重'].values())
                    }
                    weights_df = pd.DataFrame(weights_data)
                    weights_df.to_excel(writer, sheet_name='随机权重', index=False)
                
                messagebox.showinfo("成功", f"结果已导出到: {file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"导出失败: {str(e)}")
    
    def export_json(self):
        """导出到JSON"""
        if not self.results:
            messagebox.showwarning("警告", "没有可导出的结果")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="保存JSON文件",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                export_data = {
                    '评价结果': self.results,
                    '参数设置': {
                        'c1': self.c1_var.get(),
                        'r': self.r_var.get(),
                        '最高限价': self.model.max_price
                    },
                    '导出信息': {
                        '导出时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        '数据来源': self.extra_info.get('data_source', '') if self.extra_info else ''
                    }
                }
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)
                
                messagebox.showinfo("成功", f"结果已导出到: {file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"导出失败: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = BiddingEvaluationGUI(root)
    root.mainloop()