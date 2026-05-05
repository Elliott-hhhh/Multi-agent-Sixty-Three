"""
全局提醒调度器 - 统一管理所有提醒任务
=====================================
解决多提醒并发、自动退出等问题
"""

import threading
import time
from datetime import datetime
from typing import Optional, Callable, Dict, List
import heapq


class ReminderTask:
    """单个提醒任务"""
    def __init__(self, task_id: str, remind_time: datetime, title: str, content: str, 
                 openid: str, callback: Optional[Callable] = None):
        self.task_id = task_id
        self.remind_time = remind_time
        self.title = title
        self.content = content
        self.openid = openid
        self.callback = callback
        self.status = "pending"  # pending, sent, failed
        self.result = None
    
    def __lt__(self, other):
        """用于优先队列排序"""
        return self.remind_time < other.remind_time


class GlobalReminderScheduler:
    """
    全局提醒调度器（单例模式）
    
    特性：
    1. 统一管理所有提醒任务
    2. 单线程调度，避免多线程冲突
    3. 支持任务完成自动退出
    4. 支持并发发送多个提醒
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._tasks: Dict[str, ReminderTask] = {}
        self._task_queue: List[ReminderTask] = []
        self._queue_lock = threading.Lock()
        self._scheduler_thread: Optional[threading.Thread] = None
        self._running = False
        self._stop_event = threading.Event()
        self._wechat_client = None
        self._initialized = True
        self._active_tasks_count = 0
        self._tasks_count_lock = threading.Lock()
    
    def set_wechat_client(self, client):
        """设置微信客户端"""
        self._wechat_client = client
    
    def add_reminder(self, task_id: str, remind_time: datetime, title: str, 
                     content: str, openid: str, 
                     callback: Optional[Callable] = None) -> dict:
        """
        添加提醒任务
        
        Returns:
            {"success": bool, "message": str, "task_id": str}
        """
        # 检查时间是否已过
        now = datetime.now()
        if remind_time <= now:
            if (now - remind_time).total_seconds() < 60:
                # 给1分钟缓冲
                remind_time = now + __import__('datetime').timedelta(seconds=10)
            else:
                return {
                    "success": False,
                    "message": f"提醒时间 {remind_time.strftime('%Y-%m-%d %H:%M')} 已过",
                    "task_id": task_id
                }
        
        task = ReminderTask(task_id, remind_time, title, content, openid, callback)
        
        with self._queue_lock:
            self._tasks[task_id] = task
            heapq.heappush(self._task_queue, task)
            self._active_tasks_count += 1
        
        # 启动调度器（如果未启动）
        self._start_scheduler()
        
        wait_seconds = (remind_time - now).total_seconds()
        return {
            "success": True,
            "message": f"提醒已添加！将在 {remind_time.strftime('%Y-%m-%d %H:%M')}（约 {self._format_wait(wait_seconds)} 后）发送",
            "task_id": task_id
        }
    
    def _start_scheduler(self):
        """启动调度线程"""
        if self._running:
            return
        
        with self._lock:
            if not self._running:
                self._running = True
                self._stop_event.clear()
                self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=False)
                self._scheduler_thread.start()
                print(f"[全局调度器] 已启动")
    
    def _scheduler_loop(self):
        """调度器主循环"""
        while self._running and not self._stop_event.is_set():
            now = datetime.now()
            tasks_to_send = []
            
            # 检查是否有到期的任务
            with self._queue_lock:
                while self._task_queue and self._task_queue[0].remind_time <= now:
                    task = heapq.heappop(self._task_queue)
                    if task.status == "pending":
                        tasks_to_send.append(task)
            
            # 并发发送到期提醒
            if tasks_to_send:
                for task in tasks_to_send:
                    threading.Thread(
                        target=self._send_reminder,
                        args=(task,),
                        daemon=True
                    ).start()
            
            # 检查是否所有任务都已完成
            with self._queue_lock:
                if self._active_tasks_count == 0 and not self._task_queue:
                    print("[全局调度器] 所有提醒任务已完成，准备退出...")
                    self._running = False
                    break
            
            # 等待一小段时间再检查
            time.sleep(1)
        
        print("[全局调度器] 已停止")
    
    def _send_reminder(self, task: ReminderTask):
        """发送单个提醒"""
        if not self._wechat_client:
            task.status = "failed"
            task.result = "微信客户端未配置"
            self._decrement_active_count()
            return
        
        try:
            result = self._wechat_client.send_reminder(
                openid=task.openid,
                title=task.title,
                content=task.content,
                remind_time=task.remind_time.strftime("%Y-%m-%d %H:%M")
            )
            
            if result.get("errcode") == 0:
                task.status = "sent"
                task.result = "发送成功"
                print(f"[提醒发送] ✅ 「{task.title}」已成功发送到微信")
            else:
                task.status = "failed"
                errcode = result.get("errcode")
                errmsg = result.get("errmsg", "未知错误")
                task.result = f"[{errcode}] {errmsg}"
                print(f"[提醒发送] ❌ 「{task.title}」发送失败: [{errcode}] {errmsg}")
                if errcode == 45047:
                    print(f"[提醒发送] 💡 提示：48小时交互窗口已过期")
            
            if task.callback:
                task.callback(task.status == "sent", task.result)
                
        except Exception as e:
            task.status = "failed"
            task.result = str(e)
            print(f"[提醒发送] ❌ 「{task.title}」发送异常: {e}")
        
        finally:
            self._decrement_active_count()
    
    def _decrement_active_count(self):
        """减少活跃任务计数"""
        with self._tasks_count_lock:
            self._active_tasks_count -= 1
    
    def wait_until_all_done(self, timeout: Optional[float] = None):
        """
        等待所有提醒任务完成
        
        Args:
            timeout: 最大等待时间（秒），None表示无限等待
        """
        if not self._running:
            return
        
        print(f"\n⏳ 等待所有提醒发送完成...")
        start_time = time.time()
        
        try:
            while self._running:
                with self._queue_lock:
                    if self._active_tasks_count == 0 and not self._task_queue:
                        break
                
                if timeout and (time.time() - start_time) > timeout:
                    print("⚠️ 等待超时")
                    break
                
                time.sleep(0.5)
        
        except KeyboardInterrupt:
            print("\n👋 用户中断等待")
        
        self.stop()
    
    def stop(self):
        """停止调度器"""
        self._running = False
        self._stop_event.set()
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            self._scheduler_thread.join(timeout=5)
    
    def get_pending_tasks(self) -> List[dict]:
        """获取待处理的提醒任务"""
        with self._queue_lock:
            return [
                {
                    "task_id": t.task_id,
                    "title": t.title,
                    "remind_time": t.remind_time.strftime("%Y-%m-%d %H:%M"),
                    "status": t.status
                }
                for t in self._task_queue if t.status == "pending"
            ]
    
    @staticmethod
    def _format_wait(seconds: float) -> str:
        """格式化等待时间"""
        if seconds < 60:
            return f"{int(seconds)}秒"
        elif seconds < 3600:
            return f"{int(seconds / 60)}分钟"
        elif seconds < 86400:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}小时{minutes}分钟"
        else:
            days = int(seconds // 86400)
            hours = int((seconds % 86400) // 3600)
            return f"{days}天{hours}小时"


# 全局调度器实例
_global_scheduler = GlobalReminderScheduler()


def get_global_scheduler() -> GlobalReminderScheduler:
    """获取全局调度器实例"""
    return _global_scheduler
