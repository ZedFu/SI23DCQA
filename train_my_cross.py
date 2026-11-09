import os 
import argparse 
import yaml
import random
import numpy as np 
import torch
from torchvision import transforms

from torch.utils.data import DataLoader
from model import MultiFeatureModel
from datasets import I23DCQA_FeatureDataset
from parse import ParserClass, add_runtime_arguments, load_options

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from scipy.stats import spearmanr, kendalltau, pearsonr
from scipy.optimize import curve_fit

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, \
        y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)
    
    return y_output_logistic


def rank_loss(y_pred, y):
    if y_pred.shape[0] < 2:
        return y_pred.sum() * 0.0
    ranking_loss = torch.nn.functional.relu(
        (y_pred - y_pred.t()) * torch.sign((y.t() - y))
    )
    scale = 1 + torch.max(ranking_loss)
    return (
        torch.sum(ranking_loss) / y_pred.shape[0] / (y_pred.shape[0] - 1) / scale
    ).float()

def plcc_loss(y_pred, y):
    sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
    y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
    sigma, m = torch.std_mean(y, unbiased=False)
    y = (y - m) / (sigma + 1e-8)
    loss0 = torch.nn.functional.mse_loss(y_pred, y) / 4
    rho = torch.mean(y_pred * y)
    loss1 = torch.nn.functional.mse_loss(rho * y_pred, y) / 4
    return ((loss0 + loss1) / 2).float()

class TSystem(L.LightningModule):
    def __init__(self, opc : ParserClass):
        super(TSystem, self).__init__()
        self.opc = opc
        self.model = MultiFeatureModel()
        self.best_srcc_o, self.best_klcc_o, self.best_plcc_o = -1, -1, -1
        self.best_srcc_c, self.best_klcc_c, self.best_plcc_c = -1, -1, -1
        self.best_srcc_s, self.best_klcc_s, self.best_plcc_s = -1, -1, -1

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opc.op_lr)
        return [optimizer]

    def decode_batch(self,batch):
        input_image = batch['input_image']
        frames = batch['frames']
        transformed_feature = batch['transformed_feature']
        patches = batch['patches']
        tensor3d = batch['tensor3d']
        tensorco = batch['tensorco']
        overall = batch['overall'].float()
        color = batch['color'].float()
        shape = batch['shape'].float()
        return input_image, frames, transformed_feature, patches, tensor3d, tensorco, overall, color, shape

    def loss(self, y_pred, y):
        mse_loss = torch.mean((y_pred-y)**2)
        p_loss, r_loss = plcc_loss(y_pred, y), rank_loss(y_pred, y)
        loss = p_loss + 0.3 * r_loss + 0.1 * mse_loss
        return loss

    def training_step(self,batch, batch_idx):
        input_image, frames, transformed_feature, patches, tensor3d, tensorco, overall, color, shape = self.decode_batch(batch)
        o_l,c_l,s_l = [], [], []
        o,c,s = self.model(input_image,frames,transformed_feature,patches,tensor3d,tensorco)
        o_l.append(o)
        c_l.append(c)
        s_l.append(s)
        o = torch.mean(torch.stack(o_l,dim=-1),dim=-1)
        c = torch.mean(torch.stack(c_l,dim=-1),dim=-1)
        s = torch.mean(torch.stack(s_l,dim=-1),dim=-1)
        o_loss = self.loss(o,overall)
        c_loss = self.loss(c,color)
        s_loss = self.loss(s, shape)
        loss = (o_loss + c_loss + s_loss) / 3
        self.log("train/a_loss",o_loss)
        self.log("train/c_loss",c_loss)
        self.log("train/q_loss",s_loss)
        self.log("train/loss",loss)
        return loss

    def on_test_start(self):
        self.o_p, self.o_l = [], []
        self.c_p, self.c_l = [], []
        self.s_p, self.s_l = [], []
        
    def test_step(self,batch, batch_idx):
        input_image, frames, transformed_feature, patches, tensor3d, tensorco, overall, color, shape = self.decode_batch(batch)
        o_l,c_l,s_l = [], [], []
        o,c,s = self.model(input_image,frames,transformed_feature,patches,tensor3d,tensorco)
        o_l,c_l,s_l = [], [], []
        o_l.append(o)
        c_l.append(c)
        s_l.append(s)
        o = torch.mean(torch.stack(o_l,dim=-1),dim=-1)
        c = torch.mean(torch.stack(c_l,dim=-1),dim=-1)
        s = torch.mean(torch.stack(s_l,dim=-1),dim=-1)
        self.o_p.append(o)
        self.o_l.append(overall)
        self.c_p.append(c)
        self.c_l.append(color)
        self.s_p.append(s)
        self.s_l.append(shape)

    def test_metric(self,y_pred, y):
        y_pred, y = y_pred.cpu().numpy(), y.cpu().numpy()
        y_pred = fit_function(y, y_pred)
        # print(y_pred.shape, y.shape)
        s = spearmanr(y_pred, y)[0]
        k = kendalltau(y_pred, y)[0]
        p = pearsonr(y_pred, y)[0]
        return s, k ,p

    def on_test_end(self):
        o_p = torch.cat(self.o_p,dim=0)
        o_l = torch.cat(self.o_l,dim=0)
        c_p = torch.cat(self.c_p,dim=0)
        c_l = torch.cat(self.c_l,dim=0)
        s_p = torch.cat(self.s_p,dim=0)
        s_l = torch.cat(self.s_l,dim=0)
        o_srcc, o_klcc, o_plcc = self.test_metric(o_p, o_l)
        c_srcc, c_klcc, c_plcc = self.test_metric(c_p, c_l)
        s_srcc, s_klcc, s_plcc = self.test_metric(s_p, s_l)
        o_str = 'overall: SRCC: {}, KRCC: {}, PLCC: {}'.format(o_srcc,o_klcc,o_plcc)
        c_str = 'color: SRCC: {}, KRCC: {}, PLCC: {}'.format(c_srcc,c_klcc,c_plcc)
        s_str = 'shape: SRCC: {}, KRCC: {}, PLCC: {}'.format(s_srcc,s_klcc,s_plcc)
        with open(os.path.join('./logs/',self.opc.exp_name,'test_result.txt'),'a') as f:
            f.write(o_str + '\n' + c_str + '\n' + s_str)
            f.write('\n')

    def on_validation_start(self):
        self.o_p, self.o_l = [], []
        self.c_p, self.c_l = [], []
        self.s_p, self.s_l = [], []
        
    def validation_step(self,batch, batch_idx):
        input_image, frames, transformed_feature, patches, tensor3d, tensorco, overall, color, shape = self.decode_batch(batch)
        o_l,c_l,s_l = [], [], []
        o,c,s = self.model(input_image,frames,transformed_feature,patches,tensor3d,tensorco)
        o_l,c_l,s_l = [], [], []
        o_l.append(o)
        c_l.append(c)
        s_l.append(s)
        o = torch.mean(torch.stack(o_l,dim=-1),dim=-1)
        c = torch.mean(torch.stack(c_l,dim=-1),dim=-1)
        s = torch.mean(torch.stack(s_l,dim=-1),dim=-1)
        self.o_p.append(o)
        self.o_l.append(overall)
        self.c_p.append(c)
        self.c_l.append(color)
        self.s_p.append(s)
        self.s_l.append(shape)

    def on_validation_end(self):
        o_p = torch.cat(self.o_p,dim=0)
        o_l = torch.cat(self.o_l,dim=0)
        c_p = torch.cat(self.c_p,dim=0)
        c_l = torch.cat(self.c_l,dim=0)
        s_p = torch.cat(self.s_p,dim=0)
        s_l = torch.cat(self.s_l,dim=0)
        o_srcc, o_klcc, o_plcc = self.test_metric(o_p, o_l)
        c_srcc, c_klcc, c_plcc = self.test_metric(c_p, c_l)
        s_srcc, s_klcc, s_plcc = self.test_metric(s_p, s_l)
        # if o_srcc > self.best_srcc_o:
        #     self.best_srcc_o, self.best_klcc_o, self.best_plcc_o = o_srcc, o_klcc, o_plcc 
        # if c_srcc > self.best_srcc_c:
        #     self.best_srcc_c, self.best_klcc_c, self.best_plcc_c = c_srcc, c_klcc, c_plcc 
        # if s_srcc > self.best_srcc_s:
        #     self.best_srcc_s, self.best_klcc_s, self.best_plcc_s = s_srcc, s_klcc, s_plcc 
        if o_srcc + o_klcc + o_plcc + c_srcc + c_klcc + c_plcc + s_srcc + s_klcc + s_plcc  \
            > self.best_srcc_o + self.best_klcc_o + self.best_plcc_o + self.best_srcc_c + self.best_klcc_c + self.best_plcc_c +self.best_srcc_s + self.best_klcc_s + self.best_plcc_s:
            self.best_srcc_o, self.best_klcc_o, self.best_plcc_o = o_srcc, o_klcc, o_plcc 
            self.best_srcc_c, self.best_klcc_c, self.best_plcc_c = c_srcc, c_klcc, c_plcc 
            self.best_srcc_s, self.best_klcc_s, self.best_plcc_s = s_srcc, s_klcc, s_plcc 

    def on_fit_end(self):
        o_str = 'overall: SRCC: {}, KLCC: {}, PLCC: {}'.format(self.best_srcc_o, self.best_klcc_o, self.best_plcc_o )
        c_str = 'color: SRCC: {}, KLCC: {}, PLCC: {}'.format(self.best_srcc_c, self.best_klcc_c, self.best_plcc_c )
        s_str = 'shape: SRCC: {}, KLCC: {}, PLCC: {}'.format(self.best_srcc_s, self.best_klcc_s, self.best_plcc_s )
        with open(os.path.join('./logs/',self.opc.exp_name,'test_result.txt'),'a') as f:
            f.write(o_str + '\n' + c_str + '\n' + s_str)
            f.write('\n')


def train_test_split(ann_file, ratio=0.8, seed=42):
    random.seed(seed)
    video_infos = []
    with open(ann_file, "r") as fin:
        for line in fin.readlines():
            line_split = line.strip().split("|")
            image_path, mesh_path, video_path, overall, color, shape = line_split
            overall = float(overall)
            color = float(color)
            shape = float(shape)
            # filename = os.path.join(dataset_path, filename)
            video_infos.append(dict(image_path=image_path,
                                    mesh_path=mesh_path,
                                    video_path=video_path,
                                    overall=overall,
                                    color=color,
                                    shape=shape))
    random.shuffle(video_infos)
    return (
        video_infos[: int(ratio * len(video_infos))],
        video_infos[int(ratio * len(video_infos)) :],
    )

def train_list_generate(ann_file, seed=42):
    random.seed(seed)
    video_infos = []
    with open(ann_file, "r") as fin:
        for line in fin.readlines():
            line_split = line.strip().split("|")
            image_path, mesh_path, video_path, overall, color, shape = line_split
            overall = float(overall)
            color = float(color)
            shape = float(shape)
            # filename = os.path.join(dataset_path, filename)
            video_infos.append(dict(image_path=image_path,
                                    mesh_path=mesh_path,
                                    video_path=video_path,
                                    overall=overall,
                                    color=color,
                                    shape=shape))
    random.shuffle(video_infos)
    return video_infos

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--con', type=str, default='./config-cross/my-ta.yml',help='the config file!')
    add_runtime_arguments(parser)
    args = parser.parse_args()
    opt = load_options(args)

    opc = ParserClass(opt)

    for i in range(opc.dataset_num_split):
        checkpoint_callback = ModelCheckpoint(os.path.join('./logs', opc.exp_name, f'split_{i:02d}'),
                                              filename='{epoch:02d}', save_last=True, save_top_k=0)
        logger = TensorBoardLogger(save_dir='./logs',name=opc.exp_name)

        system = TSystem(opc)
        trainer = L.Trainer(
            callbacks=[checkpoint_callback] if opc.save_checkpoint else [],
            logger=logger,
            max_epochs=opc.train_epochs,
            enable_checkpointing=opc.save_checkpoint,
            accelerator=opc.exp_accelerator,
            devices=opc.exp_device
        )
        transformations_train = transforms.Compose([transforms.Resize(224), transforms.RandomCrop(224), transforms.ToTensor(),\
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
        # train_list, test_list = train_test_split(opc.dataset_ann,ratio = 0.8, seed=42*i)
        train_list = train_list_generate(opc.dataset_train)
        test_list = train_list_generate(opc.dataset_test)
        train_dataset = I23DCQA_FeatureDataset(opc.dataset_root,mode='train',ann_file=train_list,transform=transformations_train,video_width_crop=224,video_height_crop=224)
        test_dataset = I23DCQA_FeatureDataset(opc.dataset_root,mode='test',ann_file=test_list,transform=transformations_train,video_width_crop=224,video_height_crop=224)
    
        trainer.fit(system,train_dataloaders=DataLoader(train_dataset,batch_size=opc.train_bs,num_workers=opc.num_workers),
                    val_dataloaders=DataLoader(test_dataset,batch_size=opc.train_bs,num_workers=opc.num_workers))
        # trainer.test(system,dataloaders=DataLoader(test_dataset,batch_size=opc.train_bs,num_workers=opc.num_workers))


if __name__ == "__main__":
    main()

