"""Unit tests for ColabAccountManager multi-account switching."""

import pytest

from tdc_studio.remote.colab_account import ColabAccountManager


def test_colab_account_manager_lifecycle(tmp_path):
    # Setup temporary config directory
    config_dir = tmp_path / ".config" / "colab-cli"
    mgr = ColabAccountManager(config_dir=config_dir)

    # 1. No active account initially
    assert mgr.get_active_account() is None
    assert mgr.list_accounts() == []

    # 2. Create an initial active token
    token_data_a = b'{"access_token": "token_a_content"}'
    mgr.token_path.write_bytes(token_data_a)

    # Initially unsaved
    assert mgr.get_active_account() == "[Unknown / Unsaved Account]"

    # 3. Save as account 'acc_a'
    saved_path = mgr.save_account("acc_a")
    assert saved_path.is_file()
    assert mgr.get_active_account() == "acc_a"

    accounts = mgr.list_accounts()
    assert len(accounts) == 1
    assert accounts[0]["name"] == "acc_a"
    assert accounts[0]["is_active"] is True

    # 4. Create and save account 'acc_b'
    token_data_b = b'{"access_token": "token_b_content"}'
    mgr.token_path.write_bytes(token_data_b)
    mgr.save_account("acc_b")

    assert mgr.get_active_account() == "acc_b"
    accounts = mgr.list_accounts()
    assert len(accounts) == 2

    # 5. Switch back to account 'acc_a'
    success = mgr.use_account("acc_a")
    assert success is True
    assert mgr.get_active_account() == "acc_a"
    assert mgr.token_path.read_bytes() == token_data_a

    # 6. Automatic rotation to next available account
    next_acc = mgr.rotate_to_next_account()
    assert next_acc == "acc_b"
    assert mgr.get_active_account() == "acc_b"

    # Rotating again when all other accounts visited returns None or cycles
    third_acc = mgr.rotate_to_next_account(exclude=["acc_a"])
    assert third_acc is None

    # 7. Delete account
    mgr.delete_account("acc_a")
    remaining = [a["name"] for a in mgr.list_accounts()]
    assert "acc_a" not in remaining
    assert "acc_b" in remaining


def test_colab_account_manager_errors(tmp_path):
    mgr = ColabAccountManager(config_dir=tmp_path)

    # Save with no token
    with pytest.raises(FileNotFoundError):
        mgr.save_account("missing")

    # Use non-existent account
    with pytest.raises(FileNotFoundError):
        mgr.use_account("nonexistent")

    # Delete non-existent account
    with pytest.raises(FileNotFoundError):
        mgr.delete_account("nonexistent")

    # Empty names
    with pytest.raises(ValueError):
        mgr.save_account("")
    with pytest.raises(ValueError):
        mgr.use_account("")
    with pytest.raises(ValueError):
        mgr.delete_account("")
