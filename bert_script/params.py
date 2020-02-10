BERT_PRAMS = {
    'vocab_file': './efs/sp/wiki-ja.txt',
    'model_file': './efs/sp/wiki-ja.model',
    'bert_config_file': './efs/config.json',
    'init_checkpoint': './efs/model/model.ckpt-1400000',
    'do_lower_case': False,
    'layers': "-1,-2,-3,-4",
    'max_seq_length': 512,
    'batch_size': 32,
    'use_tpu': False,
    'master': None,
    'num_tpu_cores': 8,
    'use_one_hot_embeddings': False,
}