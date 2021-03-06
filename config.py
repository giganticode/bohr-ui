from dataclasses import dataclass
from typing import Dict, Optional

from bohrlabels.core import Label
from bohrlabels.labels import CommitLabel
from frozendict import frozendict

map = {
    'levin_files': 'levin (files)',
    'berger_files': 'berger (files)',
    'manual_labels.herzig': 'herzig',
    'commits_200k_files': '200k (files)',
    'commits_200k_files_no_merges': '200k (files) - no merges',
    'mauczka_files' : 'mauczka (files)',
    'idan_files' : 'idan (files)',
    'bohr.herzig_train': 'herzig (train)',
    'bohr.herzig_eval': 'herzig (eval)',
}

def get_mnemonic_for_dataset(dataset_name):
    return map[dataset_name]


# pull these data from bugginess-workdir
datasets_with_labels = {
    'bugginess': {
        'commits_200k_files': None,
        'bohr.herzig_train': lambda c: (CommitLabel.BugFix if c.raw_data['manual_labels']['herzig']['CLASSIFIED'] == 'BUG' else CommitLabel.NonBugFix),
        'bohr.herzig_eval': lambda c: (CommitLabel.BugFix if c.raw_data['manual_labels']['herzig']['CLASSIFIED'] == 'BUG' else CommitLabel.NonBugFix),
        'idan_files': lambda c: (CommitLabel.BugFix if c.raw_data['idan/0_1']['Is_Corrective'] else CommitLabel.NonBugFix),
        'levin_files': lambda c: (CommitLabel.BugFix if c.raw_data['manual_labels']['levin']['bug'] == 1 else CommitLabel.NonBugFix),
        'berger_files': lambda c: (CommitLabel.BugFix if c.raw_data['manual_labels']['berger']['bug'] == 1 else CommitLabel.NonBugFix),
        'manual_labels.herzig': lambda c: (CommitLabel.BugFix if c.raw_data['manual_labels']['herzig']['CLASSIFIED'] == 'BUG' else CommitLabel.NonBugFix),
        'mauczka_files': lambda c: (CommitLabel.BugFix if c.raw_data['manual_labels']['mauczka']['hl_corrective'] == 1 else CommitLabel.NonBugFix),
    },
    'refactoring': {
        'manual_labels.herzig': lambda c: (CommitLabel.Refactoring if c.raw_data['manual_labels']['herzig']['CLASSIFIED'] == 'REFACTORING' else CommitLabel.CommitLabel & ~CommitLabel.Refactoring),
    }
}

datasets_without_labels_bugginess = [d for d in map if d not in datasets_with_labels['bugginess'].keys()]


@dataclass(frozen=True)
class Task:
    name: str
    label_obj: Label
    label: str
    label_model_label_to_int: Dict[int, int]
    transformer_label_to_int: Optional[Dict[str, int]] = None


bugginess_task = Task('bugginess', CommitLabel.BugFix, 'prob_BugFix', frozendict({32752: 0, 15: 1}), frozendict({'NonBugFix': 0, 'BugFix': 1}))
refactoring_task = Task('refactoring', CommitLabel.Refactoring, 'prob_Refactoring', frozendict({32511: 0, 256: 1}))

all_tasks = [bugginess_task, refactoring_task]

label_models_metadata_refactoring = [
    {
        'name': 'zero_model',
        'label_source': 'zeros',
        'issues': 'without issues',
        'model': 'label model',
        'train_dataset': 'herzig_train',
        'task': refactoring_task,
    },
    {
        'name': 'random_model',
        'label_source': 'random',
        'issues': 'without issues',
        'model': 'label model',
        'train_dataset': 'herzig_train',
        'task': refactoring_task,
    },
    {
        'name': 'refactoring_no_ref_heuristics',
        'label_source': 'bugginess',
        'issues': 'without issues',
        'model': 'label model',
        'train_dataset': 'herzig_train',
        'task': refactoring_task,
    },
    {
        'name': 'refactoring_few_ref_heuristics',
        'label_source': 'a few keywords',
        'issues': 'without issues',
        'model': 'label model',
        'train_dataset': 'herzig_train',
        'task': refactoring_task,
    },
]


label_models_metadata = [
    {
        'name': 'zero_model',
        'label_source': 'zeros',
        'issues': 'without issues',
        'model': 'label model',
        'train_dataset': 'commits_200k_files',
        'task': bugginess_task,
    },
    {
        'name': 'random_model',
        'label_source': 'random',
        'issues': 'without issues',
        'model': 'label model',
        'train_dataset': 'commits_200k_files',
        'task': bugginess_task,
    },
    {
        'name': 'gitcproc',
        'label_source': 'gitcproc',
        'issues': 'without issues',
        'model': 'label model',
        'train_dataset': 'commits_200k_files',
        'task': bugginess_task,
    },
    {
        'name': 'gitcproc_orig',
        'label_source': 'gitcproc_orig',
        'issues': 'without issues',
        'model': 'label model',
        'train_dataset': 'commits_200k_files',
        'task': bugginess_task,
    },
    {
        'name': 'all_heuristics_without_issues',
        'label_source': 'all heuristics',
        'issues': 'without issues',
        'model': 'label model',
        'train_dataset': 'commits_200k_files',
        'task': bugginess_task,
    },
    {
        'name': 'all_heuristics_with_issues',
        'label_source': 'all heuristics',
        'issues': 'with issues',
        'model': 'label model',
        'train_dataset': 'commits_200k_files',
        'task': bugginess_task,
    },
    {
        'name': 'only_keywords',
        'label_source': 'keywords',
        'issues': 'with issues',
        'model': 'label model',
        'train_dataset': 'commits_200k_files',
        'task': bugginess_task,
    },
    {
        'name': 'only_message_and_label_keywords',
        'label_source': 'keywords',
        'issues': 'issue labels',
        'model': 'label model',
        'train_dataset': 'commits_200k_files',
        'task': bugginess_task,
    },
    {
        'name': 'only_message_keywords',
        'label_source': 'keywords',
        'issues': 'without issues',
        'model': 'label model',
        'train_dataset': 'commits_200k_files',
        'task': bugginess_task,
    },
    {
        'name': 'all_heuristics_without_issues_berger',
        'label_source': 'all heuristics',
        'issues': 'without issues',
        'model': 'label model',
        'train_dataset': 'berger_train',
        'task': bugginess_task,
    },
    {
        'name': 'all_heuristics_without_issues_herzig',
        'label_source': 'all heuristics',
        'issues': 'without issues',
        'model': 'label model',
        'train_dataset': 'herzig_train',
        'task': bugginess_task,
    },
    {
        'name': 'all_heuristics_without_issues_levin',
        'label_source': 'all heuristics',
        'issues': 'without issues',
        'model': 'label model',
        'train_dataset': 'levin_train',
        'task': bugginess_task,
    },
    {
        'name': 'all_heuristics_without_issues_important_projects',
        'label_source': 'all heuristics',
        'issues': 'without issues',
        'model': 'label model',
        'train_dataset': 'levin_berger_herzig_train',
        'task': bugginess_task,
    },
    {
        'name': 'all_heuristics_without_issues_orig200k',
        'label_source': 'all heuristics',
        'issues': 'without issues',
        'model': 'label model',
        'train_dataset': 'commits_200k',
        'task': bugginess_task,
    },
    {
        'name': 'only_message_keywords_berger',
        'label_source': 'keywords',
        'issues': 'without issues',
        'model': 'label model',
        'train_dataset': 'berger_train',
        'task': bugginess_task,
    },
    {
        'name': 'only_message_keywords_herzig',
        'label_source': 'keywords',
        'issues': 'without issues',
        'model': 'label model',
        'train_dataset': 'herzig_train',
        'task': bugginess_task,
    },
    {
        'name': 'only_message_keywords_levin',
        'label_source': 'keywords',
        'issues': 'without issues',
        'model': 'label model',
        'train_dataset': 'levin_train',
        'task': bugginess_task,
    },
    {
        'name': 'only_message_keywords_important_projects',
        'label_source': 'keywords',
        'issues': 'without issues',
        'model': 'label model',
        'train_dataset': 'levin_berger_herzig_train',
        'task': bugginess_task,
    },
]
