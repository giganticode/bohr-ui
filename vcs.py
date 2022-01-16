import logging
from datetime import datetime
from pathlib import Path
from time import time
from typing import Optional

import appdirs
from git import Repo
import dvc.api


logger = logging.getLogger()


def get_cloned_rev(repo: str, rev: str = "master", commit: Optional[str] = None) -> Repo:
    path = appdirs.user_cache_dir(
        appname='bohr-ui', appauthor='giganticode', version='0.1'
    )
    host = 'https://github.com/'
    if not repo.startswith('https://github.com/'):
        raise AssertionError(repo)

    path_to_repo = Path(path) / 'github-cache' / repo[len(host):] / rev
    if commit is not None:
        path_to_repo = path_to_repo / commit
    if not path_to_repo.exists():
        repo = Repo.clone_from(repo, path_to_repo, depth=1, b=rev)
        if commit is not None:
            repo.git.checkout(commit)
        return repo
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
        remote_url: str, rev: str, force_update: bool = False, commit: Optional[str] = None
) -> Optional[Path]:
    old_revision: Repo = get_cloned_rev(remote_url, rev, commit)
    if force_update or is_update_needed(old_revision):
        if force_update:
            logger.debug("Forcing refresh ...")
        update(old_revision)
    return Path(old_revision.working_tree_dir)
