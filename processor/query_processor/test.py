#!/usr/bin/env python3
"""
MongoDB 连接检测脚本
功能：尝试连接指定的 MongoDB 实例，执行 ping 命令，输出连接状态。
用法：可直接运行，或作为模块导入使用 check_mongo_connection() 函数。
"""

import os
import sys
import pymongo
from pymongo.errors import ConnectionFailure, OperationFailure, ConfigurationError

# --------------------------- 配置区域 ---------------------------
# 修改以下变量以匹配你的 MongoDB 连接信息
MONGO_HOST = os.getenv("MONGO_HOST", "localhost")
MONGO_PORT = int(os.getenv("MONGO_PORT", "27017"))
MONGO_USER = os.getenv("MONGO_USER", "")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD", "")
MONGO_AUTH_DB = os.getenv("MONGO_AUTH_DB", "admin")
MONGO_SSL = os.getenv("MONGO_SSL", "false").lower() in ("true", "1", "yes")
# 连接超时时间（秒）
CONNECT_TIMEOUT = int(os.getenv("MONGO_TIMEOUT", "5"))
# -------------------------------------------------------------


def check_mongo_connection(
    host: str = MONGO_HOST,
    port: int = MONGO_PORT,
    username: str = MONGO_USER,
    password: str = MONGO_PASSWORD,
    auth_db: str = MONGO_AUTH_DB,
    ssl: bool = MONGO_SSL,
    timeout: int = CONNECT_TIMEOUT,
) -> bool:
    """
    检测 MongoDB 连接是否正常。
    返回 True 表示连接成功，False 表示失败。
    """
    client = None
    try:
        # 构建连接 URI
        if username and password:
            uri = f"mongodb://{username}:{password}@{host}:{port}/?authSource={auth_db}"
        else:
            uri = f"mongodb://{host}:{port}/"

        # 创建客户端，设置超时
        client = pymongo.MongoClient(
            uri,
            serverSelectionTimeoutMS=timeout * 1000,  # 转换为毫秒
            ssl=ssl,
        )

        # 执行 ping 命令检测连接
        client.admin.command("ping")
        print(f"✅ MongoDB 连接成功！服务地址：{host}:{port}")
        return True

    except ConnectionFailure as e:
        print(f"❌ 连接失败（网络/认证/服务不可用）：{e}")
    except OperationFailure as e:
        print(f"❌ 操作失败（权限/数据库错误）：{e}")
    except ConfigurationError as e:
        print(f"❌ 配置错误（URI/SSL 等）：{e}")
    except Exception as e:
        print(f"❌ 未知错误：{e}")
    finally:
        if client:
            client.close()

    return False


def main():
    """命令行入口"""
    success = check_mongo_connection()
    # 返回退出码：0 成功，1 失败
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()