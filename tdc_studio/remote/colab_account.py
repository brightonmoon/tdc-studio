"""Colab CLI Multi-Account Switcher and Manager for Windows and Unix."""

import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


class ColabAccountManager:
    """Manages multi-account OAuth tokens for google-colab-cli.

    In environments like Windows, google-colab-cli stores user OAuth credentials
    at ~/.config/colab-cli/token.json. When GPU quotas are exceeded (e.g.
    TooManyAssignmentsError / Precondition Failed), switching between registered
    Google accounts allows uninterrupted training workflows.
    """

    def __init__(self, config_dir: Optional[Path] = None):
        if config_dir is None:
            self.config_dir = Path.home() / ".config" / "colab-cli"
        else:
            self.config_dir = Path(config_dir)

        self.token_path = self.config_dir / "token.json"
        self.config_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _compute_hash(file_path: Path) -> str:
        """Compute MD5 hash of a file."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    def get_active_account(self) -> Optional[str]:
        """Detect the name of the currently active Colab account.

        Compares the hash of token.json against existing token_<name>.json files.
        """
        if not self.token_path.is_file():
            return None

        try:
            active_hash = self._compute_hash(self.token_path)
        except OSError:
            return None

        backup_files = list(self.config_dir.glob("token_*.json"))
        for b in backup_files:
            # Skip temporary or lock files
            if b.name.endswith(".tmp_bak"):
                continue
            try:
                b_hash = self._compute_hash(b)
                if b_hash == active_hash:
                    # Strip 'token_' prefix and '.json' suffix
                    return b.stem.removeprefix("token_")
            except OSError:
                continue

        return "[Unknown / Unsaved Account]"

    def list_accounts(self) -> List[Dict[str, Any]]:
        """List all saved accounts and their active status."""
        active_account = self.get_active_account()
        accounts: List[Dict[str, Any]] = []

        backup_files = sorted(self.config_dir.glob("token_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for b in backup_files:
            if b.name.endswith(".tmp_bak"):
                continue
            name = b.stem.removeprefix("token_")
            is_active = (name == active_account)
            accounts.append({
                "name": name,
                "is_active": is_active,
                "path": str(b),
                "mtime": b.stat().st_mtime,
            })

        return accounts

    def save_account(self, account_name: str) -> Path:
        """Save currently active token.json as token_<account_name>.json."""
        if not account_name:
            raise ValueError("Account name must not be empty.")

        if not self.token_path.is_file():
            raise FileNotFoundError(
                f"No active token.json found at {self.token_path}. Run a colab login/session first."
            )

        target = self.config_dir / f"token_{account_name}.json"
        shutil.copy2(self.token_path, target)
        return target

    def use_account(self, account_name: str) -> bool:
        """Switch the active credentials to token_<account_name>.json."""
        if not account_name:
            raise ValueError("Account name must not be empty.")

        target = self.config_dir / f"token_{account_name}.json"
        if not target.is_file():
            available = [a["name"] for a in self.list_accounts()]
            raise FileNotFoundError(
                f"Account credentials for '{account_name}' not found at {target}. "
                f"Available accounts: {available}"
            )

        shutil.copy2(target, self.token_path)
        return True

    def delete_account(self, account_name: str) -> bool:
        """Delete credentials file for account_name."""
        if not account_name:
            raise ValueError("Account name must not be empty.")

        target = self.config_dir / f"token_{account_name}.json"
        if not target.is_file():
            raise FileNotFoundError(f"Account '{account_name}' not found.")

        target.unlink()
        return True

    def new_account(self, account_name: str, colab_binary: str = "colab") -> bool:
        """Initiate OAuth login for a new account and save as token_<account_name>.json.

        Temporarily backs up the active token, prompts interactive login,
        saves the resulting token, and restores or activates the new token.
        """
        if not account_name:
            raise ValueError("Account name must not be empty.")

        target = self.config_dir / f"token_{account_name}.json"
        temp_backup = self.config_dir / "token.json.tmp_bak"

        # 1. Back up active token if present
        had_token = False
        if self.token_path.is_file():
            shutil.move(str(self.token_path), str(temp_backup))
            had_token = True

        try:
            print(f"[ColabAccountManager] Starting OAuth flow for '{account_name}'...")
            print("Please follow browser prompts to complete authentication.")

            # Execute `colab sessions` to trigger OAuth login flow
            subprocess.run([colab_binary, "sessions"], check=False)

            if self.token_path.is_file():
                shutil.copy2(self.token_path, target)
                print(f"[ColabAccountManager] Successfully authenticated and saved account '{account_name}'.")
                return True
            else:
                print("[ColabAccountManager] Authentication failed or was aborted.")
                # Restore previous token if login failed
                if had_token and temp_backup.is_file():
                    shutil.move(str(temp_backup), str(self.token_path))
                return False
        finally:
            # Clean up temp backup if still exists
            if temp_backup.is_file():
                temp_backup.unlink()

    def rotate_to_next_account(self, exclude: Optional[List[str]] = None) -> Optional[str]:
        """Automatically switch to the next available account.

        Useful when quota errors (e.g. TooManyAssignmentsError) occur.
        Returns the new account name, or None if no other accounts exist.
        """
        active = self.get_active_account()
        exclude_set = set(exclude or [])
        if active:
            exclude_set.add(active)

        available_accounts = [
            acc["name"] for acc in self.list_accounts() if acc["name"] not in exclude_set
        ]

        if not available_accounts:
            return None

        next_acc = available_accounts[0]
        self.use_account(next_acc)
        return next_acc
