import logging
from datetime import datetime
from pathlib import Path
from time import time
from typing import Optional

import appdirs
from git import Repo
import dvc.api


logger = logging.getLogger()


def get_cloned_rev(repo: str, rev: str = "master") -> Repo:
    path = appdirs.user_cache_dir(
        appname='bohr-ui', appauthor='giganticode', version='0.1'
    )
    host = 'https://github.com/'
    if not repo.startswith('https://github.com/'):
        raise AssertionError(repo)

    path_to_repo = Path(path) / 'github-cache' / repo[len(host):] / rev
    if not path_to_repo.exists():
        return Repo.clone_from(repo, path_to_repo, depth=1, b=rev)
    else:
        return Repo(path_to_repo)


def is_update_needed(git_revision: Repo) -> bool:
    fetch_head_file = Path(git_revision.working_tree_dir) / ".git" / "FETCH_HEAD"
    if not fetch_head_file.exists():
        return True

    last_modification = fetch_head_file.stat().st_mtime
    updated_sec_ago = time() - last_modification
    logger.debug(
        f"Repo {git_revision} last attempt to pull {datetime.fromtimestamp(last_modification)}"
    )
    return updated_sec_ago > 300


def update(
        repo: Repo
) -> None:
    logger.info("Updating the repo... ")
    repo.remotes.origin.pull()


def get_path_to_revision(
        remote_url: str, rev: str, force_update: bool = False
) -> Optional[Path]:
    old_revision: Repo = get_cloned_rev(remote_url, rev)
    if is_update_needed(old_revision) or force_update:
        if force_update:
            logger.debug("Forcing refresh ...")
        update(old_revision)
    else:
        logger.info(f"Pass `--force-refresh` to refresh the repository.")
    return old_revision.working_tree_dir
