import os
import sys
import argparse
import datetime
import shutil

import h5py
import torch
import torch.optim as optim

from DataLoader_DIW import DataLoader as DataLoader_DIW
from DataLoader import DataLoader


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', default='ReD', help='model file definition')
    parser.add_argument('-bs',default=4, type=int, help='batch size')
    parser.add_argument('-it', default=100000, type=int, help='Iterations')
    parser.add_argument('-lt', default=10, type=int, help='Loss file saving refresh interval (seconds)')
    parser.add_argument('-mt', default=3000 , type=int, help='Model saving interval (iterations)')
    parser.add_argument('-et', default=1000 , type= int, help='Model evaluation interval (iterations)')
    parser.add_argument('-lr', default=1e-3 , type= float, help='Learning rate')
    parser.add_argument('-t_depth_file', default='', help='Training file for relative depth')
    parser.add_argument('-v_depth_file', default='' , help='Validation file for relative depth')
    parser.add_argument('-rundir', default='' , help='Running directory')
    parser.add_argument('-ep', default=10 , type=int , help='Epochs')
    parser.add_argument('-start_from', default='' , help='Start from previous model')
    parser.add_argument('-diw', default=False , type=bool , help='Is training on DIW dataset')
    parser.add_argument('-optim', default='RMSprop', help='choose the optimizer')
    args = parser.parse_args()
    return args


def save_loss_accuracy(args, g_model, t_loss, t_WKDR, v_loss, v_WKDR):
    _v_loss_tensor = torch.Tensor(v_loss)
    _t_loss_tensor = torch.Tensor(t_loss)
    _v_WKDR_tensor = torch.Tensor(v_WKDR)
    _t_WKDR_tensor = torch.Tensor(t_WKDR)

    _full_filename = os.path.join(args.rundir, 'loss_accuracy_record_period' + str(g_model.period) + '.h5')
    if os.path.isfile(_full_filename):
        os.remove(_full_filename)

    myFile = h5py.File(_full_filename, 'w')
    myFile.create_dataset('t_loss', data=_t_loss_tensor.numpy())
    myFile.create_dataset('v_loss', data=_v_loss_tensor.numpy())
    myFile.create_dataset('t_WKDR', data=_t_WKDR_tensor.numpy())
    myFile.create_dataset('v_WKDR', data=_v_WKDR_tensor.numpy())
    myFile.close()


def save_model(model, directory, current_iter, config):
    model.config = config
    torch.save(model, directory+'/model_period'+str(model.period)+'_'+str(current_iter)+'.pth')


def save_best_model(model, directory, config, iteration):
    model.config = config
    model.iter = iteration
    torch.save(model, os.path.join(directory,'Best_model_period'+str(model.period)+'.pth'))


if __name__ == '__main__':
    # --- arguments ---
    args = parseArgs()

    # --- dataloader ---
    train_depth_path = None
    valid_depth_path = None
    folderpath = '/home/jim/Documents/NYU/data/'
    # arguments check
    if args.t_depth_file != '':
        train_depth_path = folderpath + args.t_depth_file
    if args.v_depth_file != '':
        valid_depth_path = folderpath + args.v_depth_file
    # error hint
    if train_depth_path is None:
        print("Error: Missing training file for depth!")
        sys.exit(1)
    if valid_depth_path is None:
        print("Error: Missing validation file for depth!")
        sys.exit(1)
    # dataloader
    if args.diw:
        train_loader = DataLoader_DIW(train_depth_path, folderpath+"train_labels/", symbol=args.m)
        valid_loader = DataLoader_DIW(valid_depth_path, folderpath+"val_labels/", symbol=args.m)
        from validation_crit.validate_crit_DIW import *
    else:
        train_loader = DataLoader(train_depth_path, folderpath+"train_labels/", symbol=args.m)
        valid_loader = DataLoader(valid_depth_path, folderpath+"val_labels/", symbol=args.m)
        from validation_crit.validate_crit1 import *
    
    if args.it == 0:
        args.it = int(args.ep * (train_loader.n_relative_depth_sample)/args.bs)

    # Run path
    jobid = os.getenv('PBS_JOBID')
    job_name = os.getenv('PBS_JOBNAME')
    if args.rundir == '':
        if jobid == '' or jobid is None:
            jobid = 'debug'
        else:
            jobid = jobid.split('%.')[0]
        args.rundir = os.path.join('../results/', args.m, str(job_name))
    if not os.path.exists(args.rundir):
        os.mkdir(args.rundir)
    torch.save(args, args.rundir+'/args.pth')

    # --- Model and criterion ---
    # if args.m == "hourglass":
    #     from models.hourglass import *
    # elif args.m == "ReD":
    #     from models.ReDWeb_network import *
    from models.hourglass import *
    from models.ReDWeb_network import get_criterion
    config = {}
    if args.start_from != '':
        print(os.path.join(args.rundir, args.start_from))
        g_model = torch.load(os.path.join(args.rundir , args.start_from))
        if g_model.period is None:
            g_model.period = 1
        g_model.period += 1
        config = g_model.config
    else:
        g_model = Model()
        g_model.period = 1
    config['learningRate'] = args.lr

    if get_criterion is None:
        print("Error: no criterion specified!!!!!!!")
        sys.exit(1)

    g_criterion = get_criterion()
    g_model = g_model.cuda()
    if args.optim == 'RMSprop':
        optimizer = optim.RMSprop(g_model.parameters(), lr=args.lr) #optimizer
        print('Using RMSprop')
    elif args.optim == 'Adam':
        optimizer = optim.Adam(g_model.parameters(), lr=args.lr)
        print('Using Adam')

    # --- Run Iteration ---
    best_valist_set_error_rate = 1.0
    train_loss = []
    train_WKDR = []
    valid_loss = []
    valid_WKDR = []
    lfile = open(args.rundir+'/training_loss_period'+str(g_model.period)+'.txt', 'w')
    total_loss = 0.0
    for i in range(0,args.it):
        # one step
        batch_input, batch_target = train_loader.load_next_batch(args.bs)
        optimizer.zero_grad()
        batch_output = g_model(batch_input)
        batch_loss = g_criterion.forward(batch_output, batch_target)
        batch_loss.backward()
        optimizer.step()
        total_loss += batch_loss.item()
        if i % 20 == 0:
            t_now = datetime.datetime.now()
            t = t_now.strftime("%Y-%m-%d %H:%M:%S")
            print(t)
        print(('iteration {}, loss = {}'.format(i, batch_loss.item())))
        lfile.write('iteration {}, loss = {}\n'.format(i, batch_loss.item()))

        if i % args.mt == 0 and i!=0:
            print('Saving model at iteration {}...'.format(i))
            save_model(g_model, args.rundir, i, config)

        if i % args.et == 0:
            print('Evaluatng at iteration {}'.format(i))
            train_eval_loss, train_eval_WKDR = evaluate(train_loader, g_model, g_criterion, 100) #TODO
            valid_eval_loss, valid_eval_WKDR = evaluate(valid_loader, g_model, g_criterion, 100)
            print("train_eval_loss:",train_eval_loss, "; train_eval_WKDR:" ,train_eval_WKDR)
            print("valid_eval_loss:", valid_eval_loss, "; valid_eval_WKDR:", valid_eval_WKDR)

            train_loss.append(batch_loss.item())
            valid_loss.append(valid_eval_loss)
            train_WKDR.append(train_eval_WKDR)
            valid_WKDR.append(valid_eval_WKDR)

            save_loss_accuracy(args, g_model, train_loss, train_WKDR, valid_loss, valid_WKDR)

            if best_valist_set_error_rate > valid_eval_WKDR:
                best_valist_set_error_rate = valid_eval_WKDR
                save_best_model(g_model, args.rundir, config, i)
