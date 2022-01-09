def get_mnemonic_for_dataset(dataset_name):
    map = {
        'levin_files': 'levin (files)',
        'berger_files': 'berger (files)',
        'manual_labels.herzig': 'herzig',
        'commits_200k_files': '200k (files)',
        # 'commits_200k_files_no_merges': '200k (files) - no merges',
        'mauczka_files' : 'mauczka (files)',
        'bohr.herzig_train': 'herzig (train)',
        'bohr.herzig_eval': 'herzig (eval)',
    }
    return map[dataset_name]


datasets_with_labels = [
    'levin_files',
    'berger_files',
    'manual_labels.herzig',
    'mauczka_files',
    'bohr.herzig_train',
    'bohr.herzig_eval',
]

datasets_without_labels = [
    'commits_200k_files',
]


predefined_models = {
    'gitcproc': {
        'label_source': 'gitcproc',
        'issues': 'without issues',
        'model': 'label model',
    },
    'gitcproc_orig': {
        'label_source': 'gitcproc_orig',
        'issues': 'without issues',
        'model': 'label model',
    },
    'all_heuristics_without_issues': {
        'label_source': 'all heuristics',
        'issues': 'without issues',
        'model': 'label model',
    },
    'all_heuristics_with_issues': {
        'label_source': 'all heuristics',
        'issues': 'with issues',
        'model': 'label model',
    },
    'only_keywords': {
        'label_source': 'keywords',
        'issues': 'with issues',
        'model': 'label model',
    },
    'only_message_keywords': {
        'label_source': 'keywords',
        'issues': 'without issues',
        'model': 'label model',
    },
    'all_heuristics_with_issues_message_and_change': {
        'trained_on': 'message and change',
        'label_source': 'all heuristics',
        'issues': 'with issues',
        'model': 'transformer',
    },
    'all_heuristics_with_issues_only_change': {
        'trained_on': 'only change',
        'label_source': 'all heuristics',
        'issues': 'with issues',
        'model': 'transformer',
    },
    'all_heuristics_with_issues_only_message': {
        'trained_on': 'only message',
        'label_source': 'all heuristics',
        'issues': 'with issues',
        'model': 'transformer',
    },
    'all_heuristics_without_issues_message_and_change': {
        'trained_on': 'message and change',
        'label_source': 'all heuristics',
        'issues': 'without issues',
        'model': 'transformer',
    },
    # 'all_heuristics_without_issues_no_merge_only_message': {
    #     'trained_on': 'only message',
    #     'label_source': 'all heuristics',
    #     'issues': 'without issues',
    #     'model': 'transformer',
    # },
    'all_heuristics_without_issues_only_message': {
        'trained_on': 'only message',
        'label_source': 'all heuristics',
        'issues': 'without issues',
        'model': 'transformer',
    },
    'gitcproc_only_change': {
        'trained_on': 'only change',
        'label_source': 'gitcproc',
        'issues': 'without issues',
        'model': 'transformer',
    },
    'gitcproc_only_message': {
        'trained_on': 'only message',
        'label_source': 'gitcproc',
        'issues': 'without issues',
        'model': 'transformer',
    },
    'only_message_keywords_message_and_change': {
        'trained_on': 'message and change',
        'label_source': 'keywords',
        'issues': 'without issues',
        'model': 'transformer',
    },
    # 'only_message_keywords_no_merge_only_change': {
    #     'trained_on': 'only change',
    #     'label_source': 'keywords',
    #     'issues': 'without issues',
    #     'model': 'transformer',
    # },
    # 'only_message_keywords_no_merge_only_message': {
    #     'trained_on': 'only message',
    #     'label_source': 'keywords',
    #     'issues': 'without issues',
    #     'model': 'transformer',
    # },
    'only_message_keywords_only_change': {
        'trained_on': 'only change',
        'label_source': 'keywords',
        'issues': 'without issues',
        'model': 'transformer',
    },
    'only_message_keywords_only_message': {
        'trained_on': 'only message',
        'label_source': 'keywords',
        'issues': 'without issues',
        'model': 'transformer',
    },
}