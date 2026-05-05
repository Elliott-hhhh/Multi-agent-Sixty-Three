import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# 使用绝对导入
from backend.tools import set_reminder
from backend.reminder_scheduler import get_global_scheduler

print("=" * 60)
print("测试多提醒调度功能")
print("=" * 60)

# 设置多个提醒（模拟14:00和14:05两个提醒）
from datetime import datetime, timedelta

now = datetime.now()
reminder1_time = (now + timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M")
reminder2_time = (now + timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M")

print(f"\n当前时间: {now.strftime('%Y-%m-%d %H:%M')}")
print(f"提醒1时间: {reminder1_time}")
print(f"提醒2时间: {reminder2_time}\n")

# 添加第一个提醒
result1 = set_reminder.invoke({"content": "喝水", "remind_time": reminder1_time})
print(f"提醒1: {result1}\n")

# 添加第二个提醒
result2 = set_reminder.invoke({"content": "吃药", "remind_time": reminder2_time})
print(f"提醒2: {result2}\n")

# 获取全局调度器并等待所有任务完成
scheduler = get_global_scheduler()
scheduler.wait_until_all_done()

print("\n" + "=" * 60)
print("所有提醒已处理完毕，程序自动退出")
print("=" * 60)
