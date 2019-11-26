import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, DecoderWithAttention
from frameDataset import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu

# Data parameters
#data_folder = '/tmp/xinyuw3/input_file'  # folder with data files saved by create_input_files.py
data_folder = '.'
data_name = 'finetune1'
# Model parameters
emb_dim = 512  # dimension of word embeddings
role_dim = 512 # dimension of role embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
#cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
print(device)
# Training parameters
start_epoch = 0
epochs = 30  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 3
workers = 4  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 0.001 #4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
print_freq = 1  # print training/validation stats every __ batches
fine_tune_encoder = True  # fine-tune encoder?
checkpoint = None  # path to checkpoint, None if none


def main():
    """
    Training and validation.
    """

    global epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map, role_map
    #print('reading word map')
    # Read word map
    word_map_file = os.path.join(data_folder, 'token2id' + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    #print('reading role map')
    role_map_file = os.path.join(data_folder, 'roles2id' + '.json')
    with open(role_map_file, 'r') as j:
        role_map = json.load(j)
    #print('initializing..')
    # Initialize / load checkpoint
    if checkpoint is None:
        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       role_vocab_size=len(role_map),
                                       role_embed_dim=role_dim,
                                       dropout=dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        #best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)
    #print('creating encoder/decoder..')
    #encoder = nn.DataParallel(encoder,device_ids=[0,1])
    #decoder = nn.DataParallel(decoder,device_ids=[0,1])
    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    #print('creating dataloader..')
    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        FrameDataset(data_folder, 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    # val_loader = torch.utils.data.DataLoader(
    #     FrameDataset(data_folder, 'VAL', transform=transforms.Compose([normalize])),
    #     batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    # Epochs
    for epoch in range(start_epoch, epochs):

        # decay learning rate somehow
        # One epoch's training
        #print('start training')
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)
        print('start validation..')
        # One epoch's validation
        # recent_bleu4 = validate(val_loader=val_loader,
        #                         encoder=encoder,
        #                         decoder=decoder,
        #                         criterion=criterion)


        # Save checkpoint
        # save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
        #                 decoder_optimizer, recent_bleu4, False)


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """
    
    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()
    print('building average meters')
    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()


    # Batches
    for i, (imgs, caps, caplens, roles, all_frames) in enumerate(train_loader):
        data_time.update(time.time() - start)
        print('enumerating loader')
        # Move to GPU, if available
        
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)
        roles = roles.to(device)
        all_frames = all_frames.to(device)
        # Forward prop.
        imgs = encoder(imgs)
        all_losses = []
        all_top5 = []
        #print('enumerate ground truths')
        for j in range(3):
            frames = all_frames[:,j,:].squeeze(1)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, frames, caplens, roles)
        
            targets = caps_sorted

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
            scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # Calculate loss
            cur_loss_buffer = criterion(scores, targets)
            cur_loss_buffer += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
            cur_loss = cur_loss_buffer
            all_losses.append(cur_loss)

            cur_top5 = accuracy(scores, targets, 1)
            all_top5.append(cur_top5)
   
           # decoder_optimizer.zero_grad()
            #if encoder_optimizer is not None:
             #   encoder_optimizer.zero_grad()
            #cur_loss.backward()

            # Clip gradients
            #if grad_clip is not None:
             #   clip_gradient(decoder_optimizer, grad_clip)
            #if encoder_optimizer is not None:
             #   clip_gradient(encoder_optimizer, grad_clip)
            #print('updaitng decoder')
            # Update weights
            #decoder_optimizer.step()
            #if encoder_optimizer is not None:
             #   encoder_optimizer.step()
 
        loss = sum(all_losses)/3
        top5 = max(all_top5) # sum/3


        #print('computing gradients..')
        # Back prop.
        
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
        if encoder_optimizer is not None:
            clip_gradient(encoder_optimizer, grad_clip)
        #print('updaitng decoder')
        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()
        #print('track loss..')
        # Keep track of metrics
        losses.update(loss, sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-1 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))


def validate(val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()


    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, roles, all_frames) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)
            roles = roles.to(device)
            all_frames = all_frames.to(device)

            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)

            all_losses=[]
            all_top5=[]
            for j in range(3):
                frames = all_frames[:, j, :].squeeze(1)
                scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, frames, caplens, roles)
                targets = caps_sorted

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
                scores_copy = scores.clone()
                scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
                targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

                cur_loss = criterion(scores, targets)
                cur_loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
                all_losses.append(cur_loss)

                cur_top5 = accuracy(scores, targets, 1)
                all_top5.append(cur_top5)

            loss = min(all_losses)
            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            #top5 = accuracy(scores, targets, 1)
            top5 = max(all_top5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-1 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))



    return None


if __name__ == '__main__':
    main()
