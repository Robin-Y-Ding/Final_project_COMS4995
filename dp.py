import torch
import torch.nn.functional as F
from argument import parser, print_args
import torchvision as tv
from utils import makedirs, create_logger,save_model,evaluate
from model import ModelA
from fast_gradient_sign_untarget import FastGradientSignUntargeted, project
import os
from torch.utils.data import DataLoader
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as dsets
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Trainer():
    def __init__(self,args,logger,attack):
        self.args = args
        self.logger = logger 
        self.attack = attack 
    

    def defensive_perb(self,model,data_loader):
        args = self.args
        logger = self.logger

        
        for epoch in range(args.max_epoch):
            _iter = args.batch_size

            clean_correct = 0 
            adv_correct = 0
            nor_correct = 0


            for data, label in data_loader:
                data, label = data.cuda(), label.cuda()

                modifier = torch.zeros_like(data,requires_grad=True).cuda()
                optimizer = torch.optim.Adam([modifier],lr=args.learning_rate_for_def)

                for iter in range(1, args.max_iter):
                    optimizer.zero_grad()
                    def_data = data + modifier
                    def_data = def_data.clamp(0, 1)
                    def_output = model(def_data)
                    def_loss = F.cross_entropy(def_output,label)

                    adv_data = self.attack.perturb(def_data,label,'mean',True,False,False)
                    adv_output = model(adv_data)
                    
                    adv_loss = F.cross_entropy(adv_output,label)
                    loss = def_loss + adv_loss
                    loss.backward()

                    optimizer.step()

                
                nor_output = model(data)
                nor_loss = F.cross_entropy(nor_output,label)


                def_output = model(def_data)
                def_loss = F.cross_entropy(def_output,label)
                adv_data = self.attack.perturb(def_data,label,'mean',True,False,False)
                adv_output = model(adv_data)
                adv_loss = F.cross_entropy(adv_output, label)

                def_pred = torch.max(def_output,1)[1]
                adv_pred = torch.max(adv_output,1)[1]
                correct = (adv_pred == label).sum()
                nor_pred = torch.max(nor_output,1)[1]

                clean_correct += (def_pred == label).sum()
                adv_correct += (adv_pred == label).sum()
                nor_correct += (nor_pred == label).sum()

                logger.info('current Adv loss: %.3f, Def loss: %.3f, Total loss: %.3f, normal loss: %.3f' % (adv_loss,def_loss,loss,nor_loss))
                logger.info('%d/%d: current def acc: %.3f' % (_iter/args.batch_size,len(data_loader),float(clean_correct) / _iter))
                logger.info('%d/%d: current adv acc: %.3f' % (_iter/args.batch_size,len(data_loader),float(adv_correct) / _iter))
                logger.info('%d/%d: current normal acc: %.3f' % (_iter/args.batch_size,len(data_loader),float(nor_correct) / _iter))


                merge_data = torch.cat((def_data,data),3)

                if len(data_loader) == 118:
                    torch.save(merge_data.cpu().detach(), 'log/MNIST/linf/data/train/def_train_data_%d.pt' %  (_iter/args.batch_size))
                    torch.save(label.cpu().detach(), 'log/MNIST/linf/label/train/def_train_label_%d.pt' %  (_iter/args.batch_size))
                
                else:
                    torch.save(merge_data.cpu().detach(), 'log/MNIST/linf/data/val/def_test_data_%d.pt' %  (_iter/args.batch_size))
                    torch.save(label.cpu().detach(), 'log/MNIST/linf/label/val/def_test_label_%d.pt' %  (_iter/args.batch_size))
                
                _iter += args.batch_size

                if (_iter/args.batch_size) % args.n_store_image_step == 0 and (_iter/args.batch_size) < len(data_loader):
                    

                    label_np = label.cpu().detach().numpy()

                    np.save(os.path.join(args.log_folder,'linf/label_%d' % (_iter/args.batch_size)),label_np)


                    tv.utils.save_image(adv_data.cpu(), 
                        os.path.join(args.log_folder, 'linf/adv_images_%d.jpg' % (_iter/args.batch_size)), 
                        nrow=16)
                    


                    tv.utils.save_image(data.cpu(), 
                        os.path.join(args.log_folder, 'linf/nor_images_%d.jpg' % (_iter/args.batch_size)), 
                        nrow=16)
                    
                    tv.utils.save_image(def_data.cpu(), 
                        os.path.join(args.log_folder, 'linf/def_images_large_%d.jpg' % (_iter/args.batch_size)), 
                        nrow=16)


            
    def test(self, model, loader, adv_test=True):
        model.eval()
        total_acc = 0.0
        num = 0
        total_adv_acc = 0.0

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            for data, label in loader:
                data, label = data.to(device),label.to(device)
                output = model(data)

                pred = torch.max(output, dim=1)[1]
                te_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                
                total_acc += te_acc
                num += output.shape[0]

                if adv_test:
                    
                    adv_data = self.attack.perturb(data, pred, 'mean', False)

                    adv_output = model(adv_data)

                    adv_pred = torch.max(adv_output, dim=1)[1]
                    adv_acc = evaluate(adv_pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                    total_adv_acc += adv_acc
                else:
                    total_adv_acc = -num

        return total_acc / num , total_adv_acc / num
def main(args):

    save_folder = '%s' % (args.dataset)

    log_folder = os.path.join(args.log_root, save_folder)
    model_folder = os.path.join(args.model_root, save_folder)

    makedirs(log_folder)
    makedirs(model_folder)

    setattr(args, 'log_folder', log_folder)
    setattr(args, 'model_folder', model_folder)

    logger = create_logger(log_folder, 'def_perb', 'info')

    print_args(args, logger)

    model = ModelA()
    if torch.cuda.is_available():
        model.cuda()

    checkpoint = torch.load(args.load_checkpoint)
    model.load_state_dict(checkpoint)


    model.eval()

    attack = FastGradientSignUntargeted(model, 
                                        None,
                                        args.epsilon, 
                                        args.alpha, 
                                        min_val=0, 
                                        max_val=1, 
                                        max_iters=args.k, 
                                        _type=args.perturbation_type)
    



    trainer = Trainer(args, logger, attack)

    train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor())
    # load the dataset from pytorch official MNIST dataset; get the test data first
    test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
    tr_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    # Create the test loader from the test loader
    te_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    nor_acc, adv_acc = trainer.test(model,te_loader,True)
    logger.info('Clean Acc: %.3f, Adv Acc: %.3f' %(nor_acc,adv_acc))

    trainer.defensive_perb(model, tr_loader)

    trainer.defensive_perb(model, te_loader)


if __name__ == '__main__':
    args = parser()

    main(args)


            




    





                





