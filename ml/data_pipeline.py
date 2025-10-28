import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# --------------------------
# 配置日志（便于调试清洗过程）
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],  # 日志输出到终端
)
logger = logging.getLogger(__name__)

# --------------------------
# 路径配置（适配你的项目结构）
# --------------------------
# 原始数据路径（相对于项目根目录的data/raw）
RAW_DATA_DIR = Path("data/raw")
TRAIN_RAW_PATH = RAW_DATA_DIR / "train_raw.csv"
TEST_RAW_PATH = RAW_DATA_DIR / "test_raw.csv"

# 清洗后数据输出路径（相对于项目根目录的data/processed）
PROCESSED_DATA_DIR = Path("data/processed")
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)  # 自动创建文件夹
TRAIN_CLEAN_PATH = PROCESSED_DATA_DIR / "train_clean.csv"
TEST_CLEAN_PATH = PROCESSED_DATA_DIR / "test_clean.csv"


# --------------------------
# 核心：数据清洗函数
# --------------------------
def load_raw_data():
    """加载原始数据"""
    try:
        train = pd.read_csv(TRAIN_RAW_PATH)
        test = pd.read_csv(TEST_RAW_PATH)
        logger.info(f"成功加载原始数据：训练集{train.shape}，测试集{test.shape}")
        return train, test
    except FileNotFoundError as e:
        logger.error(f"原始数据文件不存在：{e}")
        raise  # 终止程序，需先确保data/raw有文件


def handle_missing_values(df):
    """处理缺失值：数值型特征用中位数填充（稳健性更好）"""
    # 检查缺失值
    missing = df.isnull().sum()
    if missing.sum() > 0:
        logger.warning(f"检测到缺失值：\n{missing[missing > 0]}")
        # 对数值列（feature1, feature2）用中位数填充
        for col in ["feature1", "feature2"]:
            df[col].fillna(df[col].median(), inplace=True)
        logger.info("已用中位数填充缺失值")
    else:
        logger.info("未检测到缺失值")
    return df


def remove_outliers(df):
    """删除异常值：用IQR方法检测数值特征的异常值"""
    # 仅对特征列（feature1, feature2）处理
    for col in ["feature1", "feature2"]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # 记录清洗前的数量
        before = df.shape[0]
        # 过滤异常值（保留在范围内的数据）
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        # 日志输出清洗结果
        after = df.shape[0]
        logger.info(f"删除{col}的异常值：{before - after}条（剩余{after}条）")
    return df


def clean_data():
    """完整清洗流程：加载→处理缺失值→删除异常值→保存"""
    # 1. 加载原始数据
    train_raw, test_raw = load_raw_data()

    # 2. 清洗训练集（保留目标列target）
    train_clean = handle_missing_values(train_raw.copy())
    train_clean = remove_outliers(train_clean)

    # 3. 清洗测试集（测试集不应该删除样本，可用训练集的边界处理异常值）
    test_clean = handle_missing_values(test_raw.copy())
    # 注意：测试集通常不删除样本（避免数据泄露），这里用训练集的IQR边界进行截断
    for col in ["feature1", "feature2"]:
        q1 = train_raw[col].quantile(0.25)  # 用训练集的分位数
        q3 = train_raw[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        # 截断异常值（而非删除）
        test_clean[col] = test_clean[col].clip(lower=lower_bound, upper=upper_bound)
        logger.info(f"测试集{col}异常值已截断（使用训练集边界）")

    # 4. 保存清洗后的数据
    train_clean.to_csv(TRAIN_CLEAN_PATH, index=False, encoding="utf-8")
    test_clean.to_csv(TEST_CLEAN_PATH, index=False, encoding="utf-8")
    logger.info(f"清洗后数据已保存至：{PROCESSED_DATA_DIR}")
    logger.info(f"清洗后规模：训练集{train_clean.shape}，测试集{test_clean.shape}")


# --------------------------
# 执行清洗流程
# --------------------------
if __name__ == "__main__":
    logger.info("===== 开始数据清洗流程 =====")
    clean_data()
    logger.info("===== 数据清洗流程完成 =====")
