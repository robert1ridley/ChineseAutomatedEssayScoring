class Configs:
    DROPOUT = 0.5
    CNN_FILTERS = 100
    CNN_KERNEL_SIZE = 5
    LSTM_UNITS = 100
    EMBEDDING_DIM = 300
    PRETRAINED_EMBEDDING = True
    DEBUG = False
    EMBEDDING_PATH = 'embeddings/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5'
    VOCAB_SIZE = 20000
    DATA_PATH = 'leleketang_combined_clean.tsv'
    EPOCHS = 50
    BATCH_SIZE = 10
    MODEL_OUTPUT_PATH = 'trained_models/'
    AGE_GROUPS = ['初一']


class OrgConfigs:
    DROPOUT = 0.5
    CNN_FILTERS = 100
    CNN_KERNEL_SIZE = 5
    LSTM_UNITS = 100
    EMBEDDING_DIM = 300
    PRETRAINED_EMBEDDING = True
    DEBUG = False
    EMBEDDING_PATH = 'embeddings/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5'
    VOCAB_SIZE = 20000
    DATA_PATH = 'organization_data.tsv'
    EPOCHS = 50
    BATCH_SIZE = 10
    MODEL_OUTPUT_PATH = 'trained_models/'
