import torch
import numpy as np
import torch.nn as nn
import time
import torch.nn.functional as F
from biatten import BiAttention
from en_decoder import Encoder, EncoderLayer, DecoderLayer, Decoder, Encoderimage
from util import positional_encodings_like
from random import shuffle
from torch.autograd import Variable


class natorderNet(nn.Module):
    def __init__(self, args):
        super(natorderNet, self).__init__()

        self.emb_dp = args.input_drop_ratio
        self.model_dp = args.drop_ratio

        self.embed_size = args.embed_size
        self.src_embed = nn.Embedding(args.doc_vocab, self.embed_size)

        h_dim = args.d_rnn
        d_mlp = args.d_mlp
        self.dropout = nn.Dropout(p=0.5)
        self.linears = nn.ModuleList([nn.Linear(h_dim, d_mlp),
                                      nn.Linear(h_dim, d_mlp), nn.Linear(h_dim, d_mlp), nn.Linear(h_dim, d_mlp),
                                      nn.Linear(d_mlp, 1)])

        # image encoder
        self.img_enc = Encoderimage(args.img_feature_size)

        # sentence encoder
        self.sen_enc = nn.LSTM(self.embed_size, args.d_rnn // 2, bidirectional=True, batch_first=True)

        selfatt_layer = EncoderLayer(h_dim, 4, 512, args.attdp)
        self.encoder = Encoder(selfatt_layer, args.gnnl)
        selfatt_layer2 = EncoderLayer(h_dim, 4, 512, args.attdp)
        self.encoder2 = Encoder(selfatt_layer2, args.gnnl)

        decodlayer = DecoderLayer(h_dim, 4, 512, args.attdp)
        self.decoder = Decoder(decodlayer, 1)

        self.biatten = BiAttention(h_dim, h_dim, h_dim, 8)

        self.critic = None

    def equip(self, critic):
        self.critic = critic

    def forward(self, imgfeature, captions, lengths, orders, tgt_len, doc_num):
        document_matrix, paramemory, key, document_mask, pos_emb, im_mean, sen_mean = self.encode(imgfeature, captions, lengths, doc_num)

        order = torch.tensor(orders).unsqueeze(0).cuda()
        batch, num = order.size()

        dec_outputs = self.decoder(pos_emb, paramemory, document_mask, document_mask)

        # B qN 1 H
        query = self.linears[0](dec_outputs).unsqueeze(2)
        # B 1 kN H
        key = key.unsqueeze(1)
        # B qN kN H
        e = torch.relu(query + key)
        # B qN kN
        e = self.linears[4](e).squeeze(-1)
        sum_e = e.sum(0).unsqueeze(0)

        # mask already pointed nodes
        pointed_mask = [sum_e.new_zeros(sum_e.size(0), 1, sum_e.size(2)).bool()]#.bool()

        for t in range(1, sum_e.size(1)):
            # B
            tar = order[:, t - 1]
            # B kN
            pm = pointed_mask[-1].clone().detach()
            pm[torch.arange(sum_e.size(0)), :, tar] = 1
            pointed_mask.append(pm)
        # B qN kN
        pointed_mask = torch.cat(pointed_mask, 1)

        pointed_mask_by_target = pointed_mask.new_zeros(pointed_mask.size(0), pointed_mask.size(2))
        target_mask = pointed_mask.new_zeros(pointed_mask.size(0), pointed_mask.size(1))

        for b in range(target_mask.size(0)):
            pointed_mask_by_target[b, :tgt_len[b]] = 1
            target_mask[b, :tgt_len[b]] = 1

        pointed_mask_by_target = pointed_mask_by_target.unsqueeze(1).expand_as(pointed_mask)
        sum_e.masked_fill_(pointed_mask_by_target == 0, -1e9)
        # e.masked_fill_(pointed_mask == 1, -1e9)

        loss_col = 0
        logp2 = F.log_softmax(sum_e, dim=1)
        for i in range(batch):
            len1 = tgt_len[i]
            tar1 = order[i][:len1]
            pre_p1 = logp2[i, :len1, :len1].t()

            m0 = tar1[0]
            col1 = pre_p1[m0].unsqueeze(0)

            for b in range(len1 - 1):
                m = tar1[b + 1]
                tgt_col = pre_p1[m].unsqueeze(0)
                col1 = torch.cat((col1, tgt_col), 0)
            collu = col1

            label = torch.linspace(0, len1 - 1, len1).long().cuda()

            loss2 = nn.NLLLoss()(collu, label)
            loss_col += loss2
        loss_col = loss_col / batch


        e1 = torch.relu(query + key)
        e1 = self.linears[4](e1).squeeze(-1)
        sum_e1 = e1.sum(0).unsqueeze(0)
        sum_e1.masked_fill_(pointed_mask_by_target == 0, -1e9)
        sum_e1.masked_fill_(pointed_mask == 1, -1e9)
        logp1 = F.log_softmax(sum_e1, dim=-1)
        logp1 = logp1.view(-1, logp1.size(-1))
        loss1 = self.critic(logp1, order.contiguous().view(-1))

        target_mask = target_mask.view(-1)
        loss1.masked_fill_(target_mask == 0, 0)

        loss1 = loss1.sum() / order.size(0)

        total = (loss1 + loss_col) / num

        return total, im_mean, sen_mean

    def rnn_enc(self, captions, lengths, doc_num):
        '''
        :param src_and_len:
        :param doc_num: B, each doc has sentences number
        :return: document matirx
        '''
        sorted_len, ix = torch.sort(lengths, descending=True)

        sorted_src = captions[ix]

        # bi-rnn must uses pack, else needs mask
        packed_x = nn.utils.rnn.pack_padded_sequence(sorted_src, sorted_len, batch_first=True)

        x = packed_x.data

        x = self.src_embed(x)

        if self.emb_dp > 0:
            x = F.dropout(x, self.emb_dp, self.training)

        packed_x = nn.utils.rnn.PackedSequence(x, packed_x.batch_sizes)

        # 2 TN H
        states, (hn, _) = self.sen_enc(packed_x)
        # states = nn.utils.rnn.pad_packed_sequence(states, True)

        # TN 2H
        hn = hn.transpose(0, 1).contiguous().view(captions.size(0), -1)

        # hn = hn.squeeze(0)

        _, recovered_ix = torch.sort(ix, descending=False)
        hn = hn[recovered_ix]
        # states = states[recovered_ix]

        # max-pooling
        # hn, _ = states.max(1)

        batch_size = len(doc_num)
        maxdoclen = max(doc_num)
        output = hn.view(batch_size, maxdoclen, -1)

        return output


    def encode(self, imgfeature, captions, lengths, doc_num):
        # get sentence emb and mask
        sentences = self.rnn_enc(captions, lengths, doc_num)
        imags = imgfeature.unsqueeze(0)
        pos_sents = positional_encodings_like(sentences, t=None)
        pos_sent = self.linears[3](pos_sents)

        if self.model_dp > 0:
            sentences = F.dropout(sentences, self.model_dp, self.training)

        sen_mask = sentences.new_zeros(sentences.size(0), sentences.size(1)).bool()
        for i in range(sentences.size(0)):
            sen_mask[i, :sentences.size(1)] = 1
        sen_mask = sen_mask.unsqueeze(1)
        parasen = self.encoder(sentences + pos_sent, sen_mask)

        ima_mask = imags.new_zeros(imags.size(0), imags.size(1)).bool()
        for i in range(imags.size(0)):
            ima_mask[i, :imags.size(1)] = 1
        ima_mask = ima_mask.unsqueeze(1)
        paraim = self.encoder2(imags, ima_mask)

        biattention = self.biatten.forward_all(paraim, parasen)
        biatte = torch.mean(biattention.squeeze(0), dim=0, keepdim=True).squeeze(0)
        biatten = F.softmax(biatte, dim=-1)
        posi_attention = torch.mm(biatten, self.linears[2](pos_sents).squeeze(0))

        pos_emb = positional_encodings_like(imags, t=None)

        para_memory = paraim + posi_attention.unsqueeze(0)

        im_mean = torch.mean(paraim.squeeze(0), dim=0, keepdim=True)
        sen_mean = torch.mean(parasen.squeeze(0), dim=0, keepdim=True)
        key = self.linears[1](posi_attention.unsqueeze(0))

        return imags, para_memory, key, ima_mask, pos_emb, im_mean, sen_mean

    def load_pretrained_emb(self, emb):
        self.src_embed = nn.Embedding.from_pretrained(emb, freeze=False).cuda()


def nat_pointer(args, model, imgfeature, captions, lengths, doc_num):
    document, paramemory, keys, document_mask, pos_emb, im_mean, sen_mean = model.encode(imgfeature, captions, lengths, doc_num)

    dec_output = model.decoder(pos_emb, paramemory, document_mask, document_mask)
    query = model.linears[0](dec_output).unsqueeze(2)

    # B 1 kN H
    key = keys.unsqueeze(1)
    # B qN kN H
    e = torch.relu(query + key)
    # B qN kN
    e = model.linears[4](e).squeeze(-1)
    sum_e = e.sum(0).unsqueeze(0)

    logp = F.softmax(sum_e, dim=-1)
    log_p = logp.squeeze(0)
    mask = torch.zeros_like(log_p).bool()
    bestout = []
    for i in range(sum_e.size(1)):
        best_p = torch.max(log_p[i].unsqueeze(0), 1)[1]
        m = best_p[0].item()
        bestout.append(m)
        mask[:, m] = 1
        log_p.masked_fill_(mask == 1, 0)

    best_output = bestout

    return best_output

def print_params(model):
    print('total parameters:', sum([np.prod(list(p.size())) for p in model.parameters()]))


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """
    Returns cosine similarity between x1 and x2, computed along dim.
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

def get_loss(imgs, caps, margin):
    # sims = torch.Tensor(imgs.size(0),imgs.size(0))
    sims = []
    n_image = imgs.size(0)
    for i in range(n_image):
        img_i = imgs[i]  # .unsqueeze(0) # (dim,)
        img_rep = img_i.repeat(n_image, 1)
        sim_i = cosine_similarity(img_rep, caps)
        sims.append(sim_i.unsqueeze(0))

    sims = torch.cat(sims, 0)  # row: i2t col: t2i

    diagonal = sims.diag().view(imgs.size(0), 1)
    d1 = diagonal.expand_as(sims)
    d2 = diagonal.t().expand_as(sims)
    loss_t2i = (margin + sims - d1).clamp(min=0.).sum()
    #loss_i2t = (margin + sims - d2).clamp(min=0.).sum()
    #losses = (loss_i2t + loss_t2i) / imgs.size(0)
    losses = loss_t2i
    return losses

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def train(args, train_iter, dev, test_real, doc, checkpoint):
    model = natorderNet(args)
    model.cuda()

    model.load_pretrained_emb(doc)

    print_params(model)
    print(model)

    wd = 1e-5

    if args.optimizer == 'Adam':
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=wd)
    elif args.optimizer == 'Adadelta':
        opt = torch.optim.Adadelta(model.parameters(), lr=args.lr, rho=0.95, weight_decay=wd)
    elif args.optimizer == 'AdaGrad':
        opt = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=wd)
    else:
        raise NotImplementedError

    best_score = -np.inf
    best_iter = 0

    # lr_sche = torch.optim.lr_scheduler.ExponentialLR(opt, args.lrdecay)

    criterion = nn.NLLLoss(reduction='none')
    model.equip(criterion)

    start = time.time()
    patience = args.patience

    early_stop = args.early_stop

    for epc in range(args.maximum_steps):

        for iters, (image_stories, captions_set, lengths_set, order_set, tgt_len_set, doc_num_set) in enumerate(train_iter):
            model.train()
            model.zero_grad()
            loss = 0
            ims_mean = []
            sens_mean = []

            t1 = time.time()
            images = to_var(torch.stack(image_stories))
            img_feature = model.img_enc(images)

            for si, data in enumerate(zip(img_feature, captions_set, lengths_set, order_set, tgt_len_set, doc_num_set)):
                imgfeature = data[0]
                captions = to_var(data[1])
                lengths = data[2]
                orders = data[3]
                tgt_len = data[4]
                doc_num = data[5]
                # shuffle exorder
                randid = list(range(len(orders)))
                shuffle(randid)
                imgfeatur = torch.stack([imgfeature[ri] for ri in randid])
                order = [orders[ri] for ri in randid]
                order = list(np.argsort(order))

                loss1, im_mean, sen_mean = model(imgfeatur, captions, lengths, order, tgt_len, doc_num)
                ims_mean.append(im_mean)
                sens_mean.append(sen_mean)
                loss = loss1 + loss

            ims = torch.cat(ims_mean)
            sens = torch.cat(sens_mean)
            loss_triple = get_loss(ims, sens, args.margin)
            loss_triple /= (args.batch_size)
            loss /= (args.batch_size)

            totalloss = loss + args.lamb*loss_triple
            totalloss.backward()

            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            t2 = time.time()

            if iters % 500 == 0:
                print('epc:{} iter:{} point:{:.2f} t:{:.2f}'.format(epc, iters + 1, loss,
                t2 - t1))

        with torch.no_grad():
            print('valid..............')
            score, pmr, ktau = valid_model(args, model, dev)
            print('epc:{}, acc:{:.2%}, best:{:.2%}'.format(epc, score, best_score))

            if score > best_score:
                best_score = score
                best_iter = epc

                print('save best model at epc={}'.format(epc))
                checkpoint = {'model': model.state_dict(),
                              'optim': opt.state_dict(),
                              'args': args,
                              'best_score': best_score}
                torch.save(checkpoint, '{}/{}.best.pt'.format(args.model_path, args.model))

            if early_stop and (epc - best_iter) >= early_stop:
                print('early stop at epc {}'.format(epc))
                break

    print('\n*******Train Done********{}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    minutes = (time.time() - start) // 60
    if minutes < 60:
        print('best:{:.2%}, iter:{}, time:{} mins, lr:{:.1e}, '.format(best_score, best_iter, minutes,
                                                                       opt.param_groups[0]['lr']))
    else:
        hours = minutes / 60
        print('best:{:.2%}, iter:{}, time:{:.1f} hours, lr:{:.1e}, '.format(best_score, best_iter, hours,
                                                                            opt.param_groups[0]['lr']))

    checkpoint = torch.load('{}/{}.best.pt'.format(args.model_path, args.model), map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    with torch.no_grad():
        acc, pmr, ktau = valid_model(args, model, test_real, shuflle_times=1)
        print('test acc:{:.2%} pmr:{:.2%} ktau:{:.2%}'.format(acc, pmr, ktau))


def valid_model(args, model, dev, dev_metrics=None, shuflle_times=1):
    model.eval()
    f = open(args.writetrans, 'w')

    best_acc = []

    truth = []
    predicted = []

    for j, (image_stories, captions_set, lengths_set, order_set, tgt_len_set, doc_num_set) in enumerate(dev):

        images = to_var(torch.stack(image_stories))
        img_feature = model.img_enc(images)

        for si, data in enumerate(zip(img_feature, captions_set, lengths_set, order_set, tgt_len_set, doc_num_set)):
            imgfeature = data[0]
            captions = to_var(data[1])
            lengths = data[2]
            orders = data[3]
            tgt_len = data[4]
            doc_num = data[5]
            tru = torch.tensor(orders).view(-1).tolist()
            truth.append(tru)

            if len(tru) == 1:
                pred = tru
            else:
                pred = nat_pointer(args, model, imgfeature, captions, lengths, doc_num)
            predicted.append(pred)
            print('{}|||{}'.format(' '.join(map(str, pred)), ' '.join(map(str, truth[-1]))), file=f)

    right, total = 0, 0
    pmr_right = 0
    taus = []
    pm_p, pm_r = [], []
    import itertools

    from sklearn.metrics import accuracy_score

    for t, p in zip(truth, predicted):
        if len(p) == 1:
            right += 1
            total += 1
            pmr_right += 1
            taus.append(1)
            continue

        eq = np.equal(t, p)
        right += eq.sum()
        total += len(t)

        pmr_right += eq.all()

        s_t = set([i for i in itertools.combinations(t, 2)])
        s_p = set([i for i in itertools.combinations(p, 2)])
        pm_p.append(len(s_t.intersection(s_p)) / len(s_p))
        pm_r.append(len(s_t.intersection(s_p)) / len(s_t))

        cn_2 = len(p) * (len(p) - 1) / 2
        pairs = len(s_p) - len(s_p.intersection(s_t))
        tau = 1 - 2 * pairs / cn_2
        taus.append(tau)

    # acc = right / total

    acc = accuracy_score(list(itertools.chain.from_iterable(truth)),
                         list(itertools.chain.from_iterable(predicted)))

    best_acc.append(acc)

    pmr = pmr_right / len(truth)
    taus = np.mean(taus)
    f.close()
    acc = max(best_acc)
    return acc, pmr, taus


def decode(args, test_real, checkpoint):
    with torch.no_grad():
        model = natorderNet(args)
        model.cuda()

        print('load parameters')
        model.load_state_dict(checkpoint['model'])

        acc, pmr, ktau = valid_model(args, model, test_real)
        print('test acc:{:.2%} pmr:{:.2%} ktau:{:.2f}'.format(acc, pmr, ktau))
