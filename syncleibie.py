'''
Author: papersnake cctv5cn@gmail.com
Date: 2025-10-21 10:33:20
LastEditors: papersnake cctv5cn@gmail.com
LastEditTime: 2025-10-21 10:50:54
Description: This script connects to a SQL Server database to retrieve and save item classification data.

Copyright (c) 2025 by ${git_name_email}, All Rights Reserved.
'''
import pandas as pd
import logging
from pathlib import Path
from contextlib import contextmanager
from sqlalchemy import create_engine
import urllib.parse

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 数据库配置
DB_CONFIG = {
    'server': '192.168.8.2',
    'database': 'panda',
    'username': 'sa',
    'password': 'changqi',
    'driver': 'ODBC Driver 17 for SQL Server'
}

# 确保输出目录存在
DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)


def create_db_engine():
    """创建SQLAlchemy引擎"""
    params = urllib.parse.quote_plus(
        f'DRIVER={DB_CONFIG["driver"]};'  # 移除多余的大括号
        f'SERVER={DB_CONFIG["server"]};'
        f'DATABASE={DB_CONFIG["database"]};'
        f'UID={DB_CONFIG["username"]};'
        f'PWD={DB_CONFIG["password"]}'
    )
    logging.info(f"Creating database engine with params: {params}")
    return create_engine(f'mssql+pyodbc:///?odbc_connect={params}')


@contextmanager
def create_connection():
    """创建数据库连接的上下文管理器"""
    engine = None
    try:
        engine = create_db_engine()
        with engine.connect() as conn:
            yield conn
    except Exception as e:
        logging.error(f"数据库连接错误: {str(e)}")
        raise
    finally:
        if engine:
            engine.dispose()


def save_dataframe(df: pd.DataFrame, filename: str):
    """安全保存DataFrame到CSV文件"""
    try:
        output_path = DATA_DIR / filename
        df.to_csv(output_path, encoding='utf-8', index=False)
        logging.info(f"成功保存文件到: {output_path}")
    except Exception as e:
        logging.error(f"保存文件失败: {str(e)}")
        raise


def main():
    try:
        with create_connection() as conn:
            # 生成中类信息
            queryItemCls = """
                SELECT itemclscode, ClassName
                FROM gs_item_class
                WHERE len(itemclscode)=4
                AND itemclscode not like '23%'
            """
            queryALLItemCls = """
                SELECT itemclscode, ClassName
                FROM gs_item_class
                WHERE itemclscode not like '23%'
            """

            leibei = pd.read_sql_query(queryItemCls, conn)
            Allleibie = pd.read_sql_query(queryALLItemCls, conn)

            # 数据清理
            leibei['itemclscode'] = leibei['itemclscode'].str.strip()
            Allleibie['itemclscode'] = Allleibie['itemclscode'].str.strip()

            # 保存文件
            save_dataframe(leibei, 'leibei.csv')
            save_dataframe(Allleibie, 'allleibei.csv')

            logging.info("中类信息获取完成")

    except Exception as e:
        logging.error(f"程序执行失败: {str(e)}")
        raise


if __name__ == '__main__':
    main()
