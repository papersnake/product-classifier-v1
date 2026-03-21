import os
import xlwings as xw
import numpy as np
import jieba as jb
import joblib
import pandas as pd
from price_calculator import AutoSalePrice
import preprocessing

# 导入PyTorch预测器（可选，如果不可用则跳过）
PYTORCH_AVAILABLE = False
TextCNNPredictor = None

try:
    from pytorch_predictor import TextCNNPredictor as _TextCNNPredictor
    TextCNNPredictor = _TextCNNPredictor
    PYTORCH_AVAILABLE = True
except ImportError as e:
    print(f"PyTorch预测器导入失败: {e}")
    print("PyTorch相关功能将不可用")
    PYTORCH_AVAILABLE = False

# vectorizer = joblib.load(r"E:\data\python\myproject\Tfidf_vect.pkl")
# clf = joblib.load(r"E:\data\python\myproject\clf_line_model.pkl")
# cat_id_df = pd.read_csv("E:\data\python\myproject\cat_id_df.csv", dtype={
#                       "cat_id": 'Int64', '中类': str})
# cat_to_id = dict(cat_id_df.values)
# id_to_cat = dict(cat_id_df[['cat_id', '中类']].values)

# 加载sklearn模型
skmode_path = os.path.join(os.path.dirname(
    __file__), "data", "Tfidf_min_vect.pkl")
vectorizer = joblib.load(skmode_path)
classifier_path = os.path.join(os.path.dirname(
    __file__), "data", "clf_min_model.pkl")
clf = joblib.load(classifier_path)
cat_id_df_path = os.path.join(os.path.dirname(
    __file__), "data", "cat_small_id_df.csv")
cat_id_df = pd.read_csv(cat_id_df_path, dtype={
                        "cat_id": 'Int64', 'ItemClsCode': str})
# cat_to_id = dict(cat_id_df.values)
id_to_cat = dict(cat_id_df[['cat_id', 'ItemClsCode']].values)
# print(id_to_cat)

# PyTorch预测器实例（懒加载）
_pytorch_predictor = None


def get_info(cat_id):
    print(cat_id)
    return cat_id_df.loc[cat_id_df['cat_id'] == cat_id][['ItemClsCode', 'ClassName']].values


# 使用统一的预处理模块
def preprocess(cells):
    """
    预处理函数（包装器，调用统一的预处理模块）
    注意：cells应包含ItemCode, ItemName, unit, MainSupcode列，可能包含helper列
    """
    return preprocessing.preprocess_single(cells)


def main():
    wb = xw.Book.caller()
    sheet = wb.sheets[0]
    if sheet["A1"].value == "Hello xlwings!":
        sheet["A1"].value = "Bye xlwings!"
    else:
        sheet["A1"].value = "Hello xlwings!"


@xw.func
def hello(name):
    return f"Hello {name}!"


@xw.func
def Py_AutoSalePrice(price):
    """Returns twice the sum of the two arguments"""
    result = AutoSalePrice(price)
    return result


@xw.func
def jb_cut(str):
    format_sec = " ".join([w for w in list(jb.cut(str))])
    print(format_sec)
    # return format_sec
    pred_cat_id = clf.predict(vectorizer.transform([format_sec]))
    print(pred_cat_id[0])
    return id_to_cat[pred_cat_id[0]]


@xw.func
def myPredict(sec):
    format_sec = " ".join([w for w in list(jb.cut(sec))])
    print(format_sec)
    # return format_sec
    pred_cat_id = clf.predict(vectorizer.transform([format_sec]))
    print(pred_cat_id[0])
    return id_to_cat[pred_cat_id[0]]


@xw.func
@xw.arg('data', pd.DataFrame, index=False, header=False, dtype="object")
# @xw.ret(index=False)
@xw.ret(expand='table')
def col_Predict(data):
    # data=data.rename(columns={0: "ItemCode", 1: "ItemName",3:"unit",9:"MainSupcode"})

    data = data.rename(
        columns={0: "ItemCode", 1: "ItemName", 2: "helper", 3: "unit", 9: "MainSupcode"})
    # print(data.columns)
    int_cols = [col for col in data.columns if isinstance(col, int)]
    print(int_cols)
    data = data.drop(columns=int_cols, errors='ignore')
    data['MainSupcode'] = data['MainSupcode'].astype('int').astype('object')
    # data = data.loc[:, ~data.columns.str.match('\d')]
    # row_str =" ".join([str(w) for w in data])
    # dataset = pd.DataFrame({'ItemCode': data[:,0], 'ItemName': data[:,1]})
    data.loc[:, 'preprocessed'] = data[['ItemCode', 'ItemName',
                                        'helper', 'unit', 'MainSupcode']].apply(preprocess, axis=1)
    print(data.info())
    print(data['preprocessed'])
    pred_cat_id = clf.predict(vectorizer.transform(data['preprocessed']))
    print(pred_cat_id[0])
    print(get_info(pred_cat_id[0]))
    return get_info(pred_cat_id[0])
# return id_to_cat[pred_cat_id[0]]


@xw.func
@xw.ret(expand='table')
def dynamic_array(r, c):
    return np.random.randn(int(r), int(c))


# ============================================================================
# PyTorch TextCNN预测函数
# ============================================================================

def get_pytorch_predictor():
    """获取PyTorch预测器实例（懒加载）"""
    global _pytorch_predictor

    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch预测器不可用，请确保已安装PyTorch和相关依赖")

    # 确保TextCNNPredictor已导入
    assert TextCNNPredictor is not None, "TextCNNPredictor未正确导入"

    if _pytorch_predictor is None:
        try:
            # 使用与PyTorch训练一致的类别映射
            model_dir = os.path.join(os.path.dirname(__file__), "data")
            _pytorch_predictor = TextCNNPredictor(
                model_path=os.path.join(model_dir, "textcnn_final.pth"),
                vocab_path=os.path.join(model_dir, "vocab.pkl"),
                embedding_matrix_path=os.path.join(
                    model_dir, "embedding_matrix.npy"),
                cat_mapping_path=os.path.join(
                    model_dir, "cat_id_mapping.csv"),  # 使用PyTorch训练的映射文件
                max_seq_len=11
            )
            print("PyTorch预测器初始化成功")
        except Exception as e:
            print(f"PyTorch预测器初始化失败: {e}")
            raise

    return _pytorch_predictor


@xw.func
def PyTorch_Predict(sec):
    """
    使用PyTorch TextCNN模型进行预测

    Parameters:
    sec: 产品描述文本（字符串）或包含ItemCode, ItemName, unit, MainSupcode的pandas Series

    Returns:
    str: 预测的ItemClsCode
    """
    try:
        predictor = get_pytorch_predictor()

        # 预处理文本
        import pandas as pd
        if isinstance(sec, pd.Series):
            # 如果是Series，使用统一的预处理函数
            format_sec = preprocess(sec)
            print(f"预处理文本（从Series）: {format_sec[:50]}...")
        else:
            # 如果是字符串，假设是已分词的文本或原始文本
            # 如果字符串包含空格，假设已分词；否则用jieba分词
            if ' ' in str(sec):
                format_sec = str(sec)
            else:
                format_sec = " ".join([w for w in list(jb.cut(str(sec)))])
            print(f"预处理文本: {format_sec[:50]}...")

        # 预测
        pred_cat_id, pred_name = predictor.predict_single(
            format_sec, return_prob=False)
        print(f"预测类别ID: {pred_cat_id}")
        print(f"预测类别名称: {pred_name}")

        # 返回ItemClsCode - 使用PyTorch预测器的映射
        # 加载cat_id_mapping.csv以获取ItemClsCode
        # 保证ItemClsCode是字符串类型以避免Excel显示问题
        pytorch_cat_mapping = pd.read_csv(
            "data/cat_id_mapping.csv", encoding='utf-8-sig', dtype={'cat_id': 'Int64', 'ItemClsCode': str})
        result = pytorch_cat_mapping.loc[pytorch_cat_mapping['cat_id']
                                         == pred_cat_id, 'ItemClsCode'].values
        if len(result) > 0:
            return result[0]
        else:
            # 如果找不到映射，返回类别ID作为字符串
            return str(pred_cat_id)

    except Exception as e:
        print(f"PyTorch预测错误: {e}")
        return f"错误: {str(e)}"


@xw.func
def PyTorch_Predict_WithName(sec):
    """
    使用PyTorch TextCNN模型进行预测，返回类别名称

    Parameters:
    sec: 产品描述文本

    Returns:
    str: 预测的类别名称
    """
    try:
        predictor = get_pytorch_predictor()

        # 预处理文本
        format_sec = " ".join([w for w in list(jb.cut(sec))])

        # 预测
        pred_cat_id, pred_name = predictor.predict_single(
            format_sec, return_prob=False)  # type: ignore

        return pred_name

    except Exception as e:
        print(f"PyTorch预测错误: {e}")
        return f"错误: {str(e)}"


@xw.func
@xw.arg('data', pd.DataFrame, index=False, header=False, dtype="object")
@xw.ret(expand='table')
def PyTorch_col_Predict(data):
    """
    使用PyTorch TextCNN模型进行批量预测

    Parameters:
    data: 包含产品数据的DataFrame，列顺序为：
          0: ItemCode, 1: ItemName, 2: helper, 3: unit, 9: MainSupcode

    Returns:
    pandas.DataFrame: 包含ItemClsCode和ClassName的预测结果
    """
    try:
        predictor = get_pytorch_predictor()

        # 数据预处理（与col_Predict保持一致）
        data = data.rename(
            columns={0: "ItemCode", 1: "ItemName", 2: "helper", 3: "unit", 9: "MainSupcode"})

        int_cols = [col for col in data.columns if isinstance(col, int)]
        data = data.drop(columns=int_cols, errors='ignore')
        data['MainSupcode'] = data['MainSupcode'].astype(
            'int').astype('object')

        # 预处理文本
        data.loc[:, 'preprocessed'] = data[['ItemCode', 'ItemName',
                                            'helper', 'unit', 'MainSupcode']].apply(preprocess, axis=1)

        print(f"批量预测数据形状: {data.shape}")
        print(f"预处理文本示例: {data['preprocessed'].iloc[0][:50]}...")

        # 批量预测
        predictions = predictor.predict_batch(data['preprocessed'].tolist())

        # 准备结果
        results = []
        for pred_cat_id, pred_name in predictions:
            # 查找ItemClsCode
            item_cls_code = cat_id_df.loc[
                cat_id_df['cat_id'] == pred_cat_id, 'ItemClsCode'
            ].values

            if len(item_cls_code) > 0:
                cls_code = item_cls_code[0]
                cls_name = pred_name
            else:
                cls_code = str(pred_cat_id)
                cls_name = pred_name

            results.append([cls_code, cls_name])

        # 转换为DataFrame返回
        result_df = pd.DataFrame(results, columns=['ItemClsCode', 'ClassName'])
        print(f"预测结果示例: {result_df.iloc[0]}")

        return result_df.values

    except Exception as e:
        print(f"PyTorch批量预测错误: {e}")
        import traceback
        traceback.print_exc()

        # 返回错误信息
        error_df = pd.DataFrame([['ERROR', str(e)[:50]]], columns=[
                                'ItemClsCode', 'ClassName'])
        return error_df


@xw.func
def Compare_Predictions(sec):
    """
    比较sklearn和PyTorch模型的预测结果

    Parameters:
    sec: 产品描述文本（字符串）或包含ItemCode, ItemName, unit, MainSupcode的pandas Series

    Returns:
    str: 比较结果字符串
    """
    try:
        import pandas as pd

        # 预处理文本
        if isinstance(sec, pd.Series):
            # 如果是Series，使用统一的预处理函数
            format_sec = preprocess(sec)
            print(f"比较函数预处理文本（从Series）: {format_sec[:50]}...")
        else:
            # 如果是字符串，假设是已分词的文本或原始文本
            if ' ' in str(sec):
                format_sec = str(sec)
            else:
                format_sec = " ".join([w for w in list(jb.cut(str(sec)))])
            print(f"比较函数预处理文本: {format_sec[:50]}...")

        # sklearn预测
        sklearn_pred_id = clf.predict(vectorizer.transform([format_sec]))[0]
        sklearn_result = id_to_cat.get(
            sklearn_pred_id, f"未知({sklearn_pred_id})")

        # PyTorch预测
        if PYTORCH_AVAILABLE:
            try:
                predictor = get_pytorch_predictor()
                pytorch_pred_id, pytorch_pred_name = predictor.predict_single(
                    format_sec, return_prob=False)  # type: ignore

                # 获取ItemClsCode - 使用PyTorch预测器的映射
                pytorch_cat_mapping = pd.read_csv(
                    "data/cat_id_mapping.csv", encoding='utf-8-sig', dtype={'cat_id': 'Int64', 'ItemClsCode': str})
                pytorch_result = pytorch_cat_mapping.loc[
                    pytorch_cat_mapping['cat_id'] == pytorch_pred_id, 'ItemClsCode'
                ].str.strip().values
                pytorch_result = pytorch_result[0] if len(
                    pytorch_result) > 0 else str(pytorch_pred_id)

                pytorch_result = pytorch_result

                comparison = f"sklearn: {sklearn_result} | PyTorch: {pytorch_result}"

                if sklearn_result == pytorch_result:
                    comparison += " (一致)"
                else:
                    comparison += " (不一致)"

                return comparison

            except Exception as e:
                return f"sklearn: {sklearn_result} | PyTorch: 错误({str(e)[:30]})"
        else:
            return f"sklearn: {sklearn_result} | PyTorch: 不可用"

    except Exception as e:
        return f"比较错误: {str(e)}"


if __name__ == "__main__":
    xw.Book("myproject.xlsm").set_mock_caller()
    print(joblib.__version__)

    # 测试PyTorch预测器
    if PYTORCH_AVAILABLE:
        try:
            predictor = get_pytorch_predictor()
            print("PyTorch预测器测试成功")

            # 测试单个预测
            test_text = "测试商品"
            pred_id, pred_name = predictor.predict_single(
                test_text, return_prob=False)  # type: ignore
            print(f"测试预测: 文本='{test_text}' -> ID={pred_id}, 名称={pred_name}")
        except Exception as e:
            print(f"PyTorch预测器测试失败: {e}")

    main()
