dataset_id_to_mnemonic = {
    'levin_files': 'levin (files)',
    'berger_files': 'berger (files)',
    'manual_labels.herzig': 'herzig',
    'commits_200k_files': '200k (files)',
    # 'commits_200k_files_no_merges': '200k (files) - no merges',
    'mauczka_files' : 'mauczka (files)',
    'bohr.herzig_train': 'herzig (train)',
    'bohr.herzig_eval': 'herzig (eval)',
    'bohr_200k_small_changes': '200k (small diffs)',
    'bohr_200k_large_changes': '200k (large diffs)',
    'levin_small_changes': 'levin (small diffs)',
    'levin_large_changes': 'levin (large diffs)',
    'berger_small_changes': 'berger (small diffs)',
    'berger_large_changes': 'berger (large diffs)',
}


dataset_mnemonic_to_id = {mn: id for id, mn in dataset_id_to_mnemonic.items()}


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