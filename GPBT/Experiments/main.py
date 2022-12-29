# original coder : https://github.com/D-X-Y/ResNeXt-DenseNet
# added simpnet model 
from __future__ import division

import os, sys, pdb, shutil, time, random, datetime
import argparse
import torch
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time
import copy
import models
from tensorboardX import SummaryWriter

model_names = sorted(name for name in models.__dict__
  if name.islower() and not name.startswith("__")
  and callable(models.__dict__[name]))

#print('models : ',model_names)
parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR or ImageNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data_path', type=str, help='Path to dataset')
parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10'], help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--arch', metavar='ARCH', default='resnet18', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnext29_8_64)')
# Optimization options
parser.add_argument('--epochs', type=int, default=700, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
parser.add_argument('--learning_rate', type=float, default=0.1, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.90, help='Momentum.')
parser.add_argument('--decay', type=float, default=0.002, help='Weight decay (L2 penalty).')
#parser.add_argument('--schedule', type=int, nargs='+', default=[100, 190, 306, 390, 440, 540], help='Decrease learning rate at these epochs.')
#parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
# Checkpoints
parser.add_argument('--print_freq', default=200, type=int, metavar='N', help='print frequency (default: 200)')
parser.add_argument('--save_path', type=str, default='./', help='Folder to save checkpoints and log.')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
# random seed
parser.add_argument('--manualSeed', type=int, help='manual seed')
args = parser.parse_args(['./data/cifar.python', '--dataset=cifar10', '--arch=simplenet' 
,'--save_path=./snapshots/simplenet', '--epochs=540', '--batch_size=256' ,'--workers=2' ,'--save_path=./snapshots/simplenet']
)
args.use_cuda = args.ngpu>0 and torch.cuda.is_available()

if args.manualSeed is None:
  args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.cuda.manual_seed(args.manualSeed)
if args.use_cuda:
  torch.cuda.manual_seed_all(args.manualSeed)
#speeds things a bit more  
cudnn.benchmark = True
# torch.backends.cudnn.enabled = False
# torch.backends.cudnn.deterministic = True
#asd


class TrainCIFAR():
  def __init__(self, config):
    self.args = vars(copy.deepcopy(args))
    for key, value in config.items():
        self.args[key] = value
    # Init logger
    if not os.path.isdir(args.save_path):
      os.makedirs(args.save_path)
    # used for file names, etc 
    
    self.time_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')   
    
    log = open(os.path.join(args.save_path, 'log_seed_{0}_{1}.txt'.format(args.manualSeed, self.time_stamp)), 'w')

    print_log('save path : {}'.format(args.save_path),log)
    state = {k: v for k, v in self.args.items()}
    print_log(state,log)
    print_log("Random Seed: {}".format(args.manualSeed),log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')),log)
    print_log("torch  version : {}".format(torch.__version__),log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()),log)
    # Init dataset
    if not os.path.isdir(args.data_path):
      os.makedirs(args.data_path)


    if args.dataset == 'cifar10':
      mean = [x / 255 for x in [125.3, 123.0, 113.9]]
      std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif args.dataset == 'cifar100':
      mean = [x / 255 for x in [129.3, 124.1, 112.4]]
      std = [x / 255 for x in [68.2, 65.4, 70.4]]
    else:
      assert False, "Unknow dataset : {}".format(args.dataset)



 #   writer = SummaryWriter()


    #   # Data transforms
    # mean = [0.5071, 0.4867, 0.4408]
    # std = [0.2675, 0.2565, 0.2761]


    train_transform = transforms.Compose(
      [ transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(), transforms.ToTensor(),
      transforms.Normalize(mean, std)])
      #[transforms.CenterCrop(32), transforms.ToTensor(),
      # transforms.Normalize(mean, std)])
      #)
    test_transform = transforms.Compose(
      [transforms.CenterCrop(32), transforms.ToTensor(), transforms.Normalize(mean, std)])


    if args.dataset == 'cifar10':
      train_data = dset.CIFAR10(args.data_path, train=True, transform=train_transform, download=True)
      test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
      num_classes = 10
    elif args.dataset == 'cifar100':
      train_data = dset.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)
      test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
      num_classes = 100
    elif args.dataset == 'imagenet':
      assert False, 'Did not finish imagenet code'
    else:
      assert False, 'Does not support dataset : {}'.format(args.dataset)

    from sklearn.model_selection import train_test_split
    self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=True)
  # test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
  #                       num_workers=args.workers, pin_memory=True)

    nb_test = int((.5) * len(test_data))
    test_dataset, val_dataset = torch.utils.data.dataset.random_split(test_data, [nb_test, nb_test])
    self.test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=True)
    self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=True)
    print_log("=> creating model '{}'".format(args.arch),log)
    # Init model, criterion, and optimizer
    self.net = models.__dict__[args.arch](num_classes,drp = self.args['drp'],eps=self.args['eps_arch'],momentum=self.args['momentum_arch'])
    #torch.save(net, 'net.pth')
    #init_net = torch.load('net.pth')
    #net.load_my_state_dict(init_net.state_dict())
    print_log("=> network :\n {}".format(self.net),log)

    self.net = torch.nn.DataParallel(self.net, device_ids=list(range(args.ngpu)))

    # define loss function (criterion) and optimizer
    self.criterion = torch.nn.CrossEntropyLoss()

    #optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.005, nesterov=False)
    self.optimizer = torch.optim.Adadelta(self.net.parameters(), lr=self.args['lr'], rho=self.args['momentum'], eps=self.args['eps'], # momentum=state['momentum'],
                                      weight_decay=self.args['weight_decay'])

    print_log("=> Seed '{}'".format(args.manualSeed),log)
    print_log("=> dataset mean and std '{} - {}'".format(str(mean), str(std)),log)
    
    states_settings = {
                      'optimizer': self.optimizer.state_dict()
                      }


    print_log("=> optimizer '{}'".format(states_settings),log)
    # 50k,95k,153k,195k,220k 

    if args.use_cuda:
      self.net.cuda()
      self.criterion.cuda()

    self.recorder = RecorderMeter(args.epochs)
    # optionally resume from a checkpoint
    if args.resume:
      if os.path.isfile(args.resume):
        print_log("=> loading checkpoint '{}'".format(args.resume),log)
        checkpoint = torch.load(args.resume)
        self.recorder = checkpoint['recorder']
        args.start_epoch = checkpoint['epoch']
        self.net.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print_log("=> loaded checkpoint '{}' (epoch {})" .format(args.resume, checkpoint['epoch']),log)
      else:
        print_log("=> no checkpoint found at '{}'".format(args.resume),log)
    else:
      print_log("=> did not use any checkpoint for {} model".format(args.arch),log)

    if args.evaluate:
      print("if we go in there we are wasting time (args.evaluate=true)")
      validate(self.test_loader, self.net, self.criterion,log)
      return
    self.i = 0
    log.close()


  def adapt(self, config):

    self_copy = copy.deepcopy(self)
    for key, value in config.items():
        self_copy.args[key] = value


    self_copy.net.module.adapt(self_copy.args['drp'],self_copy.args['eps_arch'],self_copy.args['momentum_arch'])



    for param_group in self_copy.optimizer.param_groups:
        param_group['lr'] = self_copy.args['lr']
        param_group['rho'] = self_copy.args['momentum']
        param_group['eps'] = self_copy.args['eps']
        param_group['weight_decay'] = self_copy.args['weight_decay']



    return self_copy
    

  def train1(self):
    log = open(os.path.join(args.save_path, 'log_seed_{0}_{1}.txt'.format(args.manualSeed, self.time_stamp)), 'a')

    # train for one epoch
    train_acc, train_los = train(self.train_loader, self.net, self.criterion, self.optimizer, self.i,log)
    self.i+=1
    log.close()
    return train_acc, train_los

  def val1(self):
      log = open(os.path.join(args.save_path, 'log_seed_{0}_{1}.txt'.format(args.manualSeed, self.time_stamp)), 'a')

      val_acc,   val_los   = validate(self.val_loader, self.net, self.criterion,log)
      log.close()
      return val_acc,val_los

  def test1(self):
      log = open(os.path.join(args.save_path, 'log_seed_{0}_{1}.txt'.format(args.manualSeed, self.time_stamp)), 'a')

      val_acc,   val_los   = validate(self.test_loader, self.net, self.criterion,log)
      log.close()
      return val_acc,val_los

  def step(self):
      log = open(os.path.join(args.save_path, 'log_seed_{0}_{1}.txt'.format(args.manualSeed, self.time_stamp)), 'a')

      start_time = time.time()
      epoch_time = AverageMeter() 
      #current_learning_rate = adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule)
      #current_learning_rate = float(self.scheduler.get_last_lr()[-1])
      #print('lr:',current_learning_rate)

      #self.scheduler.step()

      #adjust_learning_rate(optimizer, epoch)


      need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs-self.i))
      need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

      print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:.6f}]'.format(time_string(), self.i, args.epochs, need_time, self.args['lr']) \
                  + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(self.recorder.max_accuracy(False), 100-self.recorder.max_accuracy(False)),log)

      train_acc, train_los = self.train1()
      val_acc,val_los = self.val1()
      is_best = self.recorder.update(self.i-1, train_los, train_acc, val_los, val_acc)

    #  save_checkpoint({
    #    'epoch': self.i,
    #    'arch': args.arch,
    #    'state_dict': self.net.state_dict(),
    #    'recorder': self.recorder,
    #    'optimizer' : self.optimizer.state_dict(),
    #  }, is_best, args.save_path, 'checkpoint_{0}.pth.tar'.format(self.time_stamp), self.time_stamp)

      # measure elapsed time
      epoch_time.update(time.time() - start_time)
      start_time = time.time()
      #self.recorder.plot_curve( os.path.join(args.save_path, 'training_plot_{0}_{1}.png'.format(args.manualSeed, self.time_stamp)) ) 
      log.close()       
      return train_acc, train_los,val_acc,val_los





    # Main loop


   # for epoch in range(args.start_epoch, args.epochs):



   #   if epoch == 180:
   #       save_checkpoint({
   #         'epoch': epoch ,
   #         'arch': args.arch,
   #         'state_dict': net.state_dict(),
   #         'recorder': recorder,
   #         'optimizer' : optimizer.state_dict(),
   #       }, False, args.save_path, 'checkpoint_{0}_{1}.pth.tar'.format(epoch, time_stamp), time_stamp)



 #   writer.close()
 #   log.close()

  # train function (forward, backward, update)
def train(train_loader, model, criterion, optimizer, epoch, log):
  batch_time = AverageMeter()
  data_time = AverageMeter()
  losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()
  # switch to train mode
  model.train()

  end = time.time()
  for i, (input, target) in enumerate(train_loader):
    if i>0:
          break
    # measure data loading time
    data_time.update(time.time() - end)
    if args.use_cuda:
  #   target = target.cuda(async=True)
      input = input.cuda()
    input_var = torch.autograd.Variable(input)
    target_var = torch.autograd.Variable(target)
    # compute output
    output = model(input_var)
    loss = criterion(output, target_var)

    # measure accuracy and record loss
    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    losses.update(loss.data.item(), input.size(0))
    top1.update(prec1.item(), input.size(0))
    top5.update(prec5.item(), input.size(0))

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if i % args.print_freq == 0:
      print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
            'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
            'Loss {loss.val:.4f} ({loss.avg:.4f})   '
            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
            epoch, i, len(train_loader), batch_time=batch_time,
            data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log)
  print_log('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg), log)
  return top1.avg, losses.avg

def validate(val_loader, model, criterion, log):
  losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()

  # switch to evaluate mode
  model.eval()
  with torch.no_grad():
    for i, (input, target) in enumerate(val_loader):
      if i>0:
        break
      if args.use_cuda:
      #  target = target.cuda(async=True)
        input = input.cuda()
     # input_var = torch.autograd.Variable(input, volatile=True)
     # target_var = torch.autograd.Variable(target, volatile=True)

      # compute output
      output = model(input)
      loss = criterion(output, target)

      # measure accuracy and record loss
      prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
      losses.update(loss.data.item(), input.size(0))
      top1.update(prec1.item(), input.size(0))
      top5.update(prec5.item(), input.size(0))

    print_log('  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg), log)

  return top1.avg, losses.avg

def extract_features(val_loader, model, criterion, log):
  losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()

  # switch to evaluate mode
  model.eval()

  for i, (input, target) in enumerate(val_loader):
    if args.use_cuda:
    #  target = target.cuda(async=True)
      input = input.cuda()
    input_var = torch.autograd.Variable(input, volatile=True)
    target_var = torch.autograd.Variable(target, volatile=True)

    # compute output
    output, features = model([input_var])

    pdb.set_trace()

    loss = criterion(output, target_var)

    # measure accuracy and record loss
    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    losses.update(loss.data.item(), input.size(0))
    top1.update(prec1.item(), input.size(0))
    top5.update(prec5.item(), input.size(0))

  print_log('  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg), log)

  return top1.avg, losses.avg

def print_log(print_string, log):
  print("{}".format(print_string))
  log.write('{}\n'.format(print_string))
  log.flush()

def save_checkpoint(state, is_best, save_path, filename, timestamp=''):
  filename = os.path.join(save_path, filename)
  #torch.save(state, filename)
  if is_best:
    bestname = os.path.join(save_path, 'model_best_{0}.pth.tar'.format(timestamp))
    shutil.copyfile(filename, bestname)

#def adjust_learning_rate(optimizer, epoch, gammas, schedule):
#  """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#  lr = args.learning_rate
#  assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
#  for (gamma, step) in zip(gammas, schedule):
#    if (epoch >= step):
#      lr = lr * gamma
#    else:
#      break
#  for param_group in optimizer.param_groups:
#    param_group['lr'] = lr
#  return lr

# def adjust_learning_rate(optimizer, epoch):
#     """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
#     lr = args.learning_rate * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
#     # log to TensorBoard
#     # if args.tensorboard:
#     #     log_value('learning_rate', lr, epoch)
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
  """Computes the precision@k for the specified values of k"""
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0 / batch_size))
  return res
