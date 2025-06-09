# 六随机五区间报价格分模型

## 环境准备

1. **创建并激活虚拟环境**（首次使用时已自动创建）

```bash
# 进入项目目录
cd /Users/wangyue/Documents/temp/接单/待确认/6-5/6-5 六随机五区间模型

# 激活虚拟环境（每次新开终端都要先激活）
source venv/bin/activate
```

2. **安装依赖库**（如已安装可跳过）

```bash
pip install -r requirements.txt
```

## 如何运行程序

### 方式一：直接运行 main.py

```bash
python3 main.py
```

### 方式二：分析指定Excel文件（如 test.xlsx）

在 main.py 中添加如下代码或在交互式环境中运行：

```python
from main import BiddingEvaluationModel
model = BiddingEvaluationModel()
model.auto_process_file('test.xlsx')
```

## 输入文件格式要求

- 支持 Excel（.xlsx, .xls）和 CSV 文件。
- 文件需包含以下三列：
  - 序号（可选）
  - 公司名称（或"投标单位""company""bidder""单位名称"等同义表头）
  - 报价（或"投标报价""bid_price""price""金额"等同义表头）
- 若无"最高限价"列，程序会自动用最大报价的1.1倍作为最高限价。

## 输出内容

- 详细Excel报告，允许导出

---
如有疑问请联系开发者。 
