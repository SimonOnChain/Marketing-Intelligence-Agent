"""Persistent chat history management for Streamlit app."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Default storage location
DEFAULT_HISTORY_DIR = Path(__file__).parent.parent.parent / "data" / "chat_history"


class ChatHistoryManager:
    """Manages persistent chat history with file-based storage."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or DEFAULT_HISTORY_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _get_session_file(self, session_id: str) -> Path:
        """Get the file path for a session."""
        return self.storage_dir / f"{session_id}.json"

    def save_session(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """Save chat messages for a session."""
        session_file = self._get_session_file(session_id)
        data = {
            "session_id": session_id,
            "updated_at": datetime.now().isoformat(),
            "messages": messages,
        }
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Load chat messages for a session."""
        session_file = self._get_session_file(session_id)
        if not session_file.exists():
            return []
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("messages", [])
        except (json.JSONDecodeError, IOError):
            return []

    def delete_session(self, session_id: str) -> bool:
        """Delete a session's chat history."""
        session_file = self._get_session_file(session_id)
        if session_file.exists():
            session_file.unlink()
            return True
        return False

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all saved sessions with metadata."""
        sessions = []
        for session_file in self.storage_dir.glob("*.json"):
            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    messages = data.get("messages", [])
                    user_msgs = [m for m in messages if m.get("role") == "user"]
                    sessions.append({
                        "session_id": data.get("session_id", session_file.stem),
                        "updated_at": data.get("updated_at"),
                        "message_count": len(messages),
                        "query_count": len(user_msgs),
                        "preview": user_msgs[0].get("query", "")[:50] if user_msgs else "",
                    })
            except (json.JSONDecodeError, IOError):
                continue
        # Sort by most recent first
        sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return sessions

    def export_session_json(self, session_id: str) -> Optional[str]:
        """Export a session's chat history as JSON string."""
        messages = self.load_session(session_id)
        if not messages:
            return None
        export_data = {
            "session_id": session_id,
            "exported_at": datetime.now().isoformat(),
            "messages": messages,
        }
        return json.dumps(export_data, indent=2, ensure_ascii=False)

    def export_session_csv(self, session_id: str) -> Optional[str]:
        """Export a session's chat history as CSV string."""
        messages = self.load_session(session_id)
        if not messages:
            return None

        import csv
        from io import StringIO

        output = StringIO()
        fieldnames = ["role", "query", "answer", "agents_used", "execution_time", "cost", "timestamp"]
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for msg in messages:
            row = {
                "role": msg.get("role", ""),
                "query": msg.get("query", ""),
                "answer": msg.get("answer", ""),
                "agents_used": ", ".join(msg.get("agents_used", [])),
                "execution_time": msg.get("execution_time", ""),
                "cost": msg.get("cost", ""),
                "timestamp": msg.get("timestamp", ""),
            }
            writer.writerow(row)

        return output.getvalue()

    def add_message(
        self,
        session_id: str,
        role: str,
        query: Optional[str] = None,
        answer: Optional[str] = None,
        agents_used: Optional[List[str]] = None,
        execution_time: Optional[float] = None,
        cost: Optional[float] = None,
        sources: Optional[List[Dict]] = None,
    ) -> None:
        """Add a single message to a session."""
        messages = self.load_session(session_id)
        message = {
            "role": role,
            "timestamp": datetime.now().isoformat(),
        }
        if query:
            message["query"] = query
        if answer:
            message["answer"] = answer
        if agents_used:
            message["agents_used"] = agents_used
        if execution_time is not None:
            message["execution_time"] = execution_time
        if cost is not None:
            message["cost"] = cost
        if sources:
            message["sources"] = sources

        messages.append(message)
        self.save_session(session_id, messages)


# Singleton instance
_history_manager: Optional[ChatHistoryManager] = None


def get_history_manager() -> ChatHistoryManager:
    """Get or create the singleton history manager."""
    global _history_manager
    if _history_manager is None:
        _history_manager = ChatHistoryManager()
    return _history_manager
