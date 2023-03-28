from configs import get_config
from lj_dataloader import get_loader
from utils.vocab import Vocab,to_var,OOVDict
from lj_solver import *
import pickle
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    config = get_config(mode='train')
    val_config = get_config(mode='validation')
    test_config = get_config(mode='test')
    print(config)
    with open(os.path.join(config.save_path, 'config.txt'), 'w') as f:
        print(config, file=f)

    train_data_loader = get_loader(all_data=load_pickle(config.sentences_path),
                              batch_size=config.batch_size)
    eval_data_loader = get_loader(all_data=load_pickle(test_config.sentences_path),
                             batch_size=config.eval_batch_size, shuffle=False)
    test_data_loader = get_loader(all_data=load_pickle(test_config.sentences_path),
                                  batch_size=config.eval_batch_size, shuffle=False)
    solver=RLSolver
    solver = solver(config, train_data_loader, eval_data_loader,test_data_loader,is_train=True)
    solver.build_generate()
    # solver.build_emotion_classifier()
    if config.test==True:
        solver.test()
    else:
        solver.train()
        # solver.test()
        #solver.evaluate()
