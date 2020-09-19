import numpy as np
import argparse
import os

# parameters
W2V_MODEL = None
INFERSENT_MODEL = None
SETTING = {
    'w2v': None,
    'infersent': None,
    'verbose': 2
}

# --- UTILITIES --- #

LOG_LEVELS = ['ERROR', 'WARN', 'INFO', 'DEBUG']
def LOG(tag, content, verbose=SETTING['verbose']):
    log_level = -1
    for idx, level in enumerate(LOG_LEVELS):
        if tag.upper() == level:
            log_level = idx
            break
    if verbose >= log_level:
        print(f' [ {tag.upper()} ] {content} ')

def ensure_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        LOG('info', f'create directories: {dir_path}')

def read_filelines(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()!='']
    return lines


# --- SHARED USAGES--- #

# @param location: embedding dir
# @param target: word or bundle string you want to search
def check_emb_exists(location, target):
    path = os.path.join(location, target.replace(' ', '_') + '.npy')
    if os.path.isfile(path):
        LOG('debug', f'embedding exist: {path}')
        return path
    return None

# @param emb: np array
# store embeddings as `location/emb.npy`
def save_emb(location, target, emb):
    ensure_exists(location)
    path = os.path.join(location, target.replace(' ', '_'))
    np.save(path, emb)
    LOG('debug', f'embedding saved: {path+".npy"}')
    return path + '.npy'    # add numpy ext

def load_w2v_model(model_path):
    LOG('info', f'loading w2v model: {model_path}...')
    from gensim.models import KeyedVectors
    model = KeyedVectors.load(model_path)
    LOG('info', f'w2v model loaded')
    return model

def load_infersent_model(model_path, config):
    LOG('info', f'loading infersent model: {model_path}...')
    import torch
    from infersent import InferSent
    model = InferSent(config)
    model.load_state_dict(torch.load(model_path))
    LOG('info', f'infersent model loaded')
    return model


# --- WORD EMBEDDING ---#

def extract_word_emb(word, model=None):
    global W2V_MODEL 
    if not model:
        if not W2V_MODEL:
            W2V_MODEL = load_w2v_model(SETTING['w2v'])
        model = W2V_MODEL
    try:
        return model[word]
    except:
        LOG('warn', f'embedding "{word}"" not found in model!')
        return None

def get_word_emb(word, model=None, emb_dir=None):
    if emb_dir:
        path = check_emb_exists(emb_dir, word)
        if path:
            return np.load(path, allow_pickle=True)
    emb = extract_word_emb(word, model)
    if emb_dir:
        save_emb(emb_dir, word, emb)
    return emb

def get_all_word_emb(words, model=None, emb_dir=None):
    embs = []
    for word in words:
        embs.append(get_word_emb(word, model, emb_dir))
    return embs


# --- BUNDLE EMBEDDING --- #

# @param bundles: list of bundle
# @ret: all bundle embedding's np array
def get_batches(bundles, emb_dir, w2v_model=None):
    # load beginning-of-sent and end-of-sent embedding
    emb_bos = np.load(os.path.join(emb_dir, 'bos.npy'))
    emb_eos = np.load(os.path.join(emb_dir, 'eos.npy'))
    
    # extract embeddings
    max_len = 0
    embeddings, lengths, bundle_list, fail_list = [], [], [], []
    for bundle in bundles:
        words = bundle.split(' ')
        embs = []
        embs.append(emb_bos)
        embs.extend(get_all_word_emb(words, model=w2v_model, emb_dir=emb_dir))
        embs.append(emb_eos)
        
        # store info
        embeddings.append(embs)
        lengths.append(len(embs))
        bundle_list.append(bundle)
        max_len = len(embs) if len(embs) > max_len else max_len
    
    # generate batches
    batches = np.zeros((max_len, len(embeddings), embeddings[0][0].shape[0]))
    for i in range(len(embeddings)):
        for j in range(len(embeddings[i])):
            batches[j][i][:] = embeddings[i][j]
    return batches, np.array(lengths), bundle_list, fail_list

def get_bundles_embedding(bundles, infersent_model, w2v_model, emb_dir, output_dir=None):
    import torch
    config = {'bsize': 64, 
              'word_emb_dim': 300, 
              'enc_lstm_dim': 2048,
              'pool_type': 'max', 
              'dpout_model': 0.0, 
              'version': 2}
    model = load_infersent_model(SETTING['infersent'], config)

    LOG('info', 'generating batches...')
    embeddings, lengths, bundle_list, fail_list = get_batches(bundles, w2v_model=w2v_model, emb_dir=emb_dir)
    LOG('debug', f'finished. Get {len(bundle_list)} bundles.')
    
    LOG('info', 'processing bundles...')
    bundles_emb = []
    for idx in range(0, embeddings.shape[1], config['bsize']):
        batch = torch.FloatTensor(embeddings[:,idx:idx+config['bsize'], :])
        length = lengths[idx:idx+config['bsize']]
        with torch.no_grad():
            emb = model.forward((batch, length)).data.cpu().numpy()
        bundles_emb.append(emb)
    bundles_emb = np.vstack(bundles_emb)

    if output_dir:
        LOG('info', f'writing output to {output_dir}...')
        ensure_exists(output_dir)
        for (bundle, emb) in zip(bundle_list, bundles_emb):
            out_path = os.path.join(output_dir, bundle.replace(' ', '_'))
            np.save(out_path, emb)
    return bundles_emb

def log_args(args):
    print()
    print('Get setting: ')
    print(f"  + input: {args.input}")
    print(f"  + w2v model: {args.word2vec_model}")
    print(f"  + infersent model: {args.infersent_model}")
    print(f"  + word embedding dir: {args.word_embedding}")
    print(f"  + bundle embedding dir: {args.bundle_embedding}")
    print(f"  + verbose level: {args.verbose}")
    print('')
    return

if __name__ == '__main__':
    def parse_args():
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('input', help="bundles for extracting bundles")
        parser.add_argument('-w2v', '--word2vec-model', help="word2vec's kv model")
        parser.add_argument('-if', '--infersent-model', help="infersent's pkl model")
        parser.add_argument('-wd', '--word-embedding', help="word embedding folder")
        parser.add_argument('-bd', '--bundle-embedding', help="bundle embedding folder")
        parser.add_argument('-v', '--verbose', type=int, default=1, help="verbose level. 0=only error, 3=debugging")
        return parser.parse_args()
    args = parse_args()
    log_args(args)
    
    # set parameters
    if args.word2vec_model:
        SETTING['w2v'] = args.word2vec_model
    if args.infersent_model:
        SETTING['infersent'] = args.infersent_model
    if args.verbose:
        SETTING['verbose'] = args.verbose

    # get input
    if os.path.isfile(args.input):
        targets = read_filelines(args.input)
    else:
        targets = [args.input]
    LOG('info', f'get #{len(targets)} targets')

    # load models
    # w2v_model = load_w2v_model(args.word2vec_model)
    # sent_model = load_infersent_model(args.infersent_model)

    ret = get_bundles_embedding(targets, infersent_model=None, w2v_model=None, emb_dir=args.word_embedding, output_dir=args.bundle_embedding)
    print(len(ret))   # sanity check

    exit(0)
    for target in targets:
        for word in target.split(' '):
            ret = get_word_emb(word, emb_dir=args.word_embedding)
            print(f'ret shape: {ret.shape}')


