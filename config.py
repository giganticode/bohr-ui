from bohrlabels.labels import CommitLabel

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


label_models_metadata_refactoring = [
    {
        'name': 'zero_model',
        'label_source': 'zeros',
        'issues': 'without issues',
        'model': 'label model',
        'train_dataset': 'herzig_train',
        'task': 'refactoring',
    },
    {
        'name': 'random_model',
        'label_source': 'random',
        'issues': 'without issues',
        'model': 'label model',
        'train_dataset': 'herzig_train',
        'task': 'refactoring',
    },
    {
        'name': 'refactoring_no_ref_heuristics',
        'label_source': 'bugginess',
        'issues': 'without issues',
        'model': 'label model',
        'train_dataset': 'herzig_train',
        'task': 'refactoring',
    },
    {
        'name': 'refactoring_few_ref_heuristics',
        'label_source': 'a few keywords',
        'issues': 'without issues',
        'model': 'label model',
        'train_dataset': 'herzig_train',
        'task': 'refactoring',
    },
]

label_models_metadata = [
    {
        'name': 'zero_model',
        'label_source': 'zeros',
        'issues': 'without issues',
        'model': 'label model',
        'train_dataset': 'commits_200k_files',
        'task': 'bugginess',
    },
    {
        'name': 'random_model',
        'label_source': 'random',
        'issues': 'without issues',
        'model': 'label model',
        'train_dataset': 'commits_200k_files',
        'task': 'bugginess',
    },
    {
        'name': 'gitcproc',
        'label_source': 'gitcproc',
        'issues': 'without issues',
        'model': 'label model',
        'train_dataset': 'commits_200k_files',
        'task': 'bugginess',
    },
    {
        'name': 'gitcproc_orig',
        'label_source': 'gitcproc_orig',
        'issues': 'without issues',
        'model': 'label model',
        'train_dataset': 'commits_200k_files',
        'task': 'bugginess',
    },
    {
        'name': 'all_heuristics_without_issues',
        'label_source': 'all heuristics',
        'issues': 'without issues',
        'model': 'label model',
        'train_dataset': 'commits_200k_files',
        'task': 'bugginess',
    },
    {
        'name': 'all_heuristics_with_issues',
        'label_source': 'all heuristics',
        'issues': 'with issues',
        'model': 'label model',
        'train_dataset': 'commits_200k_files',
        'task': 'bugginess',
    },
    {
        'name': 'only_keywords',
        'label_source': 'keywords',
        'issues': 'with issues',
        'model': 'label model',
        'train_dataset': 'commits_200k_files',
        'task': 'bugginess',
    },
    {
        'name': 'only_message_and_label_keywords',
        'label_source': 'keywords',
        'issues': 'issue labels',
        'model': 'label model',
        'train_dataset': 'commits_200k_files',
        'task': 'bugginess',
    },
    {
        'name': 'only_message_keywords',
        'label_source': 'keywords',
        'issues': 'without issues',
        'model': 'label model',
        'train_dataset': 'commits_200k_files',
        'task': 'bugginess',
    },
]
