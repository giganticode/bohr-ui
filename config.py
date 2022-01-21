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


datasets_with_labels = [
    'commits_200k_files',
    'levin_files',
    'berger_files',
    'manual_labels.herzig',
    'mauczka_files',
    'bohr.herzig_train',
    'bohr.herzig_eval',
    'idan_files',
]

datasets_without_labels = [d for d in map if d not in datasets_with_labels]




label_models_metadata = [
    {
        'name': 'gitcproc',
        'label_source': 'gitcproc',
        'issues': 'without issues',
        'model': 'label model',
        'train_dataset': 'commits_200k_files',
    },
    {
        'name': 'gitcproc_orig',
        'label_source': 'gitcproc_orig',
        'issues': 'without issues',
        'model': 'label model',
        'train_dataset': 'commits_200k_files',
    },
    {
        'name': 'all_heuristics_without_issues',
        'label_source': 'all heuristics',
        'issues': 'without issues',
        'model': 'label model',
        'train_dataset': 'commits_200k_files',
    },
    {
        'name': 'all_heuristics_with_issues',
        'label_source': 'all heuristics',
        'issues': 'with issues',
        'model': 'label model',
        'train_dataset': 'commits_200k_files',
    },
    {
        'name': 'only_keywords',
        'label_source': 'keywords',
        'issues': 'with issues',
        'model': 'label model',
        'train_dataset': 'commits_200k_files',
    },
    {
        'name': 'only_message_keywords',
        'label_source': 'keywords',
        'issues': 'without issues',
        'model': 'label model',
        'train_dataset': 'commits_200k_files',
    },
]
