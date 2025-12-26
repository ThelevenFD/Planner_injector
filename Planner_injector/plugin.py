from dataclasses import dataclass
import time
from typing import Tuple, TypedDict, Optional, Dict, List
import httpx
import threading
from src.chat.brain_chat.brain_planner import BrainPlanner
from src.chat.utils.utils import get_chat_type_and_target_info
from src.common.data_models.database_data_model import DatabaseMessages
from src.common.data_models.info_data_model import TargetPersonInfo
from src.plugin_system import (
    ActionInfo,
    BaseEventHandler,
    BasePlugin,
    ConfigField,
    EventType,
    MaiMessages,
    register_plugin,
    BaseCommand
)
from src.common.logger import get_logger
from src.plugin_system.base.component_types import PythonDependency

logger = get_logger("planner_injector")
is_enabled = None
allow_user_debug = None
@dataclass
class UserInfo:
    userid: str
    impression: int
    attitude: str

class UserInfoCacheEntry(TypedDict):
    user_info: UserInfo
    stamp: float

class UserInfoCacheManager:
    """
    一个线程安全的、用于管理用户信息的缓存类。
    封装了缓存的读、写、过期删除等全部逻辑。
    """
    def __init__(self, ttl: int = 3600):
        self._cache: Dict[str, UserInfoCacheEntry] = {}
        self._lock = threading.Lock()
        self.ttl = ttl

    def get(self, user_id: str) -> Optional[UserInfo]:
        """
        从缓存中获取用户信息。如果信息不存在或已过期，则返回 None。
        """
        with self._lock:
            cache_entry = self._cache.get(user_id)
            if not cache_entry:
                return None

            if time.time() - cache_entry["stamp"] > self.ttl:
                # 缓存已过期，删除并返回 None
                del self._cache[user_id]
                return None
            
            return cache_entry["user_info"]

    def set(self, user_id: str, impression: int, attitude: str):
        """
        将用户信息存入缓存。
        """
        with self._lock:
            user_info = UserInfo(userid=user_id, impression=impression, attitude=attitude)
            self._cache[user_id] = UserInfoCacheEntry(user_info=user_info, stamp=time.time())

# 创建一个全局唯一的缓存管理器实例
user_cache_manager = UserInfoCacheManager()

class DebugCommand(BaseCommand):
    command_name = "debug"
    command_description = "测试命令，用于调试"
    command_pattern = r"/debug( (?P<chatid>\w+))?"


    async def execute(self):
        if not allow_user_debug or not is_enabled:
            return False, "该功能已关闭", 2
        stream_id = self.matched_groups.get("chatid")
        qid = self.message.message_info.user_info.user_id
        if not stream_id:
            stream_id = self.message.chat_stream.stream_id
        is_group_chat, _ = get_chat_type_and_target_info(stream_id)
        if not is_group_chat:
            impression = user_cache_manager.get(qid)
            await self.send_text(f"impression:{impression}",storage_message=False)
        await self.send_text(f"is_group_chat:{is_group_chat}",storage_message=False)
        return True, f"你输入的ID是：{stream_id}", 2


class UserInfoGet(BaseEventHandler):
    event_type = EventType.ON_MESSAGE
    handler_name = "user_info_get"
    handler_description = "获取用户好感度"
    weight = 900

    def __init__(self):
        super().__init__()

    async def post_api(self, user_id: str) -> Tuple[int, str]:
        """
        通过 API 获取用户信息，并增加了更完善的异常处理。
        """
        async with httpx.AsyncClient() as client:
            url = f"{self.get_config('api.url').rstrip('/')}/get_info/{user_id}"
            timeout = self.get_config("api.timeout")
            try:
                response = await client.post(url, timeout=timeout)
                response.raise_for_status()  # 检查 HTTP 错误 (如 4xx, 5xx)
                data = response.json()
                # 确保关键字段存在
                impression = data.get("impression", 0)
                attitude = data.get("attitude", "一般")
                logger.debug(f"API 请求成功，用户 {user_id} 的好感度为 {impression}")
                return impression, attitude
            except httpx.TimeoutException:
                logger.error(f"API 请求超时: {url}")
                return 0, "一般"
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                logger.error(f"API 请求失败: {e}")
                return 0, "一般"
            except Exception as e: # 捕获 JSON 解析等其他潜在错误
                logger.error(f"处理 API 响应时发生未知错误: {e}")
                return 0, "一般"

    async def execute(self, message: MaiMessages | None):
        if not is_enabled:
            return False, True, None, None, None
        base_info = message.message_base_info
        userId = str(base_info.get("user_id"))

        # 使用重构后的缓存管理器
        if user_cache_manager.get(userId) is not None:
            return True, True, None, None, None

        impression, attitude = await self.post_api(userId)
        
        # 使用重构后的缓存管理器
        user_cache_manager.set(userId, impression, attitude)
        
        return True, True, None, None, None

_original_func = None

async def _patch_planner(
    self: BrainPlanner,
    chat_target_info: Optional["TargetPersonInfo"],
    current_available_actions: Dict[str, ActionInfo],
    message_id_list: List[Tuple[str, "DatabaseMessages"]],
    chat_content_block: str = "",
    interest: str = "",
    prompt_key: str = "brain_planner_prompt_react"
) -> tuple[str, List[Tuple[str, "DatabaseMessages"]]]:
    global _original_func
    result = await _original_func(
        self=self,
        chat_target_info=chat_target_info,
        current_available_actions=current_available_actions,
        message_id_list=message_id_list,
        chat_content_block=chat_content_block,
        interest=interest,
        prompt_key=prompt_key
    ) # pyright: ignore[reportOptionalCall]
    prompt, message_id_list_result = result
    
    # 使用重构后的缓存管理器
    user_info = user_cache_manager.get(str(chat_target_info.user_id))

    if user_info:
        favorability_prompt = f"\n你对当前用户的好感度是{user_info.impression}，态度是{user_info.attitude}，好感度越高，选择reply的概率越大。好感度>50则有75%的概率reply。"
        prompt += favorability_prompt
        logger.info(f"成功为{self.chat_id[:5]}...注入好感度提示")
    return prompt, message_id_list_result

def patch():
    try:
        time.sleep(3)
        global _original_func
        _original_func = BrainPlanner.build_planner_prompt
        BrainPlanner.build_planner_prompt = _patch_planner
        logger.info("注入好感度提示 (planner patch)")
        if _original_func != BrainPlanner.build_planner_prompt:
            logger.info("成功注入好感度提示 (planner patch)")
        else:
            logger.error("注入失败 (planner patch)")
    except Exception as e:
        logger.error(f"注入失败: {e}")

# 启动 patching 线程
_thread = threading.Thread(target=patch, daemon=True)
_thread.start()

@register_plugin
class Plugin(BasePlugin):
    plugin_name = "planner_injector"
    enable_plugin = True
    dependencies = []
    python_dependencies = [PythonDependency("httpx", ">=0.28.1")]
    config_file_name = "config.toml"
    config_schema: dict = {
        "plugin": {
            "name": ConfigField(type=str, default="planner_injector", description="插件名称"),
            "config_version": ConfigField(type=str, default="1.0.2", description="配置版本(不要修改 除非你知道自己在干什么)"),
            "enabled": ConfigField(type=bool, default=True, description="是否启用插件"),
            "user_debug": ConfigField(type=bool, default=False, description="允许用户获取好感度信息"),
        },
        "api": {
            "url": ConfigField(type=str, default="http://url.to.your.zhenxun", description="与真寻连接的api地址"),
            "timeout": ConfigField(type=int, default=10, description="超时时间(s)"),
        }
    }
    description = "把回复概率与好感度挂钩"

    def __init__(self, **kwargs):
        global is_enabled, allow_user_debug
        super().__init__(**kwargs)
        plugin_name = self.get_config("plugin.name")
        logger.info(f"插件{plugin_name}已加载")
        is_enabled = self.get_config("plugin.enabled")
        allow_user_debug = self.get_config("plugin.user_debug")

    def get_plugin_components(self):
        return [
            (UserInfoGet.get_handler_info(), UserInfoGet),
            (DebugCommand.get_command_info(), DebugCommand)
        ]
