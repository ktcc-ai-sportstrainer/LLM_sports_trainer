import logging
import json
from datetime import datetime
from typing import Any, Dict, Optional
import os

class SystemLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # 基本ロガーの設定
        self.logger = logging.getLogger("SwingCoachSystem")
        # 既存のハンドラをクリア
        self.logger.handlers.clear()  
        self.logger.setLevel(logging.INFO)
        
        # ファイルハンドラの設定
        log_file = os.path.join(log_dir, f"system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # コンソールハンドラの設定
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # フォーマッターの設定
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # ハンドラの追加
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # エージェントごとの詳細ログファイル
        self.agent_loggers = {}

    def setup_agent_logger(self, agent_name: str) -> None:
        """エージェントごとの詳細ログ設定"""
        agent_logger = logging.getLogger(f"SwingCoachSystem.{agent_name}")
        agent_logger.setLevel(logging.DEBUG)
        
        # エージェント固有のログファイル
        log_file = os.path.join(self.log_dir, f"{agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        agent_logger.addHandler(handler)
        self.agent_loggers[agent_name] = agent_logger

    def log_info(self, message: str, agent: Optional[str] = None) -> None:
        """情報ログの記録"""
        if agent and agent in self.agent_loggers:
            self.agent_loggers[agent].info(message)
        self.logger.info(message)

    def log_error(self, message: str, agent: Optional[str] = None) -> None:
        """エラーログの記録"""
        if agent and agent in self.agent_loggers:
            self.agent_loggers[agent].error(message)
        self.logger.error(message)

    def log_warning(self, message: str, agent: Optional[str] = None) -> None:
        """警告ログの記録"""
        if agent and agent in self.agent_loggers:
            self.agent_loggers[agent].warning(message)
        self.logger.warning(message)

    def log_debug(self, message: str, agent: Optional[str] = None) -> None:
        """デバッグログの記録"""
        if agent and agent in self.agent_loggers:
            self.agent_loggers[agent].debug(message)
        self.logger.debug(message)

    def log_agent_input(self, agent: str, input_data: Dict[str, Any]) -> None:
        """エージェントの入力データを記録"""
        if agent in self.agent_loggers:
            self.agent_loggers[agent].debug(
                f"Input data:\n{json.dumps(input_data, indent=2, ensure_ascii=False)}"
            )

    def log_agent_output(self, agent: str, output_data: Dict[str, Any]) -> None:
        """エージェントの出力データを記録"""
        if agent in self.agent_loggers:
            self.agent_loggers[agent].debug(
                f"Output data:\n{json.dumps(output_data, indent=2, ensure_ascii=False)}"
            )

    def log_execution_time(self, agent: str, execution_time: float) -> None:
        """エージェントの実行時間を記録"""
        if agent in self.agent_loggers:
            self.agent_loggers[agent].info(f"Execution time: {execution_time:.2f} seconds")

    def log_state_change(self, state_name: str, old_value: Any, new_value: Any) -> None:
        """システム状態の変更を記録"""
        self.logger.debug(
            f"State change - {state_name}: {old_value} -> {new_value}"
        )

    def log_error_details(
        self,
        error: Exception,
        agent: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """エラーの詳細を記録"""
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.now().isoformat(),
            "context": context or {}
        }
        
        error_log = json.dumps(error_info, indent=2, ensure_ascii=False)
        
        if agent and agent in self.agent_loggers:
            self.agent_loggers[agent].error(f"Error details:\n{error_log}")
        self.logger.error(f"Error details:\n{error_log}")

    def export_logs(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """全ログファイルをエクスポート"""
        if output_dir is None:
            output_dir = self.log_dir
        
        log_files = {}
        
        # システムログ
        system_logs = [f for f in os.listdir(self.log_dir) if f.startswith("system_")]
        if system_logs:
            latest_system_log = max(system_logs)
            log_files["system"] = os.path.join(self.log_dir, latest_system_log)
        
        # エージェントログ
        for agent in self.agent_loggers:
            agent_logs = [f for f in os.listdir(self.log_dir) if f.startswith(f"{agent}_")]
            if agent_logs:
                latest_agent_log = max(agent_logs)
                log_files[agent] = os.path.join(self.log_dir, latest_agent_log)
        
        return log_files