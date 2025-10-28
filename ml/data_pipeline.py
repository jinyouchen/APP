import pandas as pd
import logging
import os
from pathlib import Path
# 新增：模型训练与保存相关依赖
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from joblib import dump  # 用于保存模型（与之前加载逻辑一致）


# --------------------------
# 1. 配置日志（便于调试全流程）
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# --------------------------
# 2. 路径配置（整合数据路径+模型保存路径）
# --------------------------
# 数据路径（保持原有逻辑）
RAW_DATA_DIR = Path("data/raw")
TRAIN_RAW_PATH = RAW_DATA_DIR / "train_raw.csv"
TEST_RAW_PATH = RAW_DATA_DIR / "test_raw.csv"

PROCESSED_DATA_DIR = Path("data/processed")
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_CLEAN_PATH = PROCESSED_DATA_DIR / "train_clean.csv"
TEST_CLEAN_PATH = PROCESSED_DATA_DIR / "test_clean.csv"

# 新增：模型保存路径（用户指定的绝对路径）
MODEL_SAVE_PATH = r"C:\Users\32583\Desktop\APP\data\model\model.pkl"
# 确保模型保存目录存在（避免“目录不存在”导致保存失败）
MODEL_SAVE_DIR = os.path.dirname(MODEL_SAVE_PATH)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


# --------------------------
# 3. 数据清洗模块（保持原有逻辑，优化列名兼容性）
# --------------------------
def load_raw_data():
    """加载原始数据"""
    try:
        train = pd.read_csv(TRAIN_RAW_PATH)
        test = pd.read_csv(TEST_RAW_PATH)
        # 校验必要列是否存在（避免后续报错）
        required_train_cols = ["feature1", "feature2", "target"]
        required_test_cols = ["feature1", "feature2"]
        if not all(col in train.columns for col in required_train_cols):
            raise ValueError(f"训练集缺少必要列！需包含：{required_train_cols}")
        if not all(col in test.columns for col in required_test_cols):
            raise ValueError(f"测试集缺少必要列！需包含：{required_test_cols}")
        
        logger.info(f"成功加载原始数据：训练集{train.shape}，测试集{test.shape}")
        return train, test
    except FileNotFoundError as e:
        logger.error(f"原始数据文件不存在：{e}")
        raise
    except ValueError as e:
        logger.error(f"数据格式错误：{e}")
        raise


def handle_missing_values(df, numeric_cols=["feature1", "feature2"]):
    """处理缺失值：数值列用中位数填充"""
    missing = df.isnull().sum()
    if missing.sum() > 0:
        logger.warning(f"检测到缺失值：\n{missing[missing > 0]}")
        for col in numeric_cols:
            if col in df.columns:
                df[col].fillna(df[col].median(), inplace=True)
                logger.info(f"用中位数填充{col}的缺失值")
            else:
                logger.warning(f"列{col}不存在，跳过缺失值处理")
    else:
        logger.info("未检测到缺失值")
    return df


def remove_outliers(df, numeric_cols=["feature1", "feature2"]):
    """删除训练集异常值（IQR方法）"""
    valid_cols = [col for col in numeric_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    if not valid_cols:
        logger.warning("无有效数值列，跳过异常值处理")
        return df
    
    for col in valid_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        before = df.shape[0]
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        after = df.shape[0]
        logger.info(f"删除{col}异常值：{before - after}条（剩余{after}条）")
    return df


def clean_data():
    """完整数据清洗流程"""
    # 加载原始数据
    train_raw, test_raw = load_raw_data()
    
    # 清洗训练集（删除异常值）
    train_clean = handle_missing_values(train_raw.copy())
    train_clean = remove_outliers(train_clean)
    
    # 清洗测试集（截断异常值，避免数据泄露）
    test_clean = handle_missing_values(test_raw.copy())
    for col in ["feature1", "feature2"]:
        if col in train_raw.columns and col in test_clean.columns:
            q1 = train_raw[col].quantile(0.25)
            q3 = train_raw[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            test_clean[col] = test_clean[col].clip(lower=lower_bound, upper=upper_bound)
            logger.info(f"测试集{col}异常值截断（训练集边界：[{lower_bound:.2f}, {upper_bound:.2f}]）")
    
    # 保存清洗后数据
    train_clean.to_csv(TRAIN_CLEAN_PATH, index=False, encoding="utf-8")
    test_clean.to_csv(TEST_CLEAN_PATH, index=False, encoding="utf-8")
    logger.info(f"清洗后数据保存至：{PROCESSED_DATA_DIR}")
    logger.info(f"清洗后规模：训练集{train_clean.shape}，测试集{test_clean.shape}")
    
    return train_clean, test_clean  # 返回清洗后数据，用于后续模型训练


# --------------------------
# 4. 新增：模型训练与保存模块
# --------------------------
def train_and_save_model(train_clean):
    """用清洗后的训练数据训练模型，并保存到指定路径"""
    # 拆分特征（X）和目标（y）
    X_train = train_clean[["feature1", "feature2"]]
    y_train = train_clean["target"]
    logger.info(f"模型训练数据规模：特征{X_train.shape}，目标{y_train.shape}")
    
    # 训练模型（示例：逻辑回归，可替换为其他模型）
    model = LogisticRegression(random_state=42)  # 固定随机种子，确保可复现
    model.fit(X_train, y_train)
    logger.info("模型训练完成（逻辑回归）")
    
    # 简单评估训练效果（可选，用于日志跟踪）
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    logger.info(f"训练集评估：准确率={train_accuracy:.4f}，F1分数={train_f1:.4f}")
    
    # 保存模型到指定路径
    dump(model, MODEL_SAVE_PATH)  # 用joblib保存，与后续加载逻辑（joblib.load）匹配
    logger.info(f"模型已保存到指定路径：{MODEL_SAVE_PATH}")
    logger.info(f"模型文件大小：{os.path.getsize(MODEL_SAVE_PATH) / 1024:.2f} KB")
    
    return model


# --------------------------
# 5. 主流程：数据清洗→模型训练→保存
# --------------------------
if __name__ == "__main__":
    logger.info("===== 开始端到端流程（数据清洗→模型训练→保存） =====")
    try:
        # 步骤1：执行数据清洗
        train_clean, test_clean = clean_data()
        
        # 步骤2：训练模型并保存到指定路径
        trained_model = train_and_save_model(train_clean)
        
        logger.info("===== 端到端流程全部完成 =====")
    except Exception as e:
        logger.error(f"流程执行失败：{str(e)}", exc_info=True)
        raise