import numpy as np
def data_load_dblp(data_path):
    f1 = open(data_path + '/paper_author.txt', 'r', encoding='utf-8')
    pa = np.zeros((4057, 14328))
    for line in f1.readlines():
        paper, author = line.strip('\n').split('\t')
        i = int(author)
        j = int(paper)
        pa[i][j] = 1
    pa=torch.from_numpy(pa)
    adj=pa
    apa = pa.dot(pa.T)
    f2 = open(data_path + '/paper_conf.txt', 'r', encoding='utf-8')
    pc = np.zeros((14328, 20))
    for line in f2.readlines():
        a, b = line.strip('\n').split('\t')
        i = int(a)
        j = int(b)
        pc[i][j] = 1
    uu=ap.dot(pc)
    apcpa=uu.dot(uu.T)
    f2 = open(data_path + '/paper_term.txt', 'r', encoding='utf-8')
    pt = np.zeros((14328, 8898))
    for line in f2.readlines():
        a, b = line.strip('\n').split('\t')
        i = int(a)
        j = int(b)
        pt[i][j] = 1
    uu = ap.dot(pt)
    aptpa = uu.dot(uu.T)
    label = np.zeros((4057, 4))
    f1 = open(data_path + '/author_label1.txt', "r")
    cont1 = f1.readlines()
    for i in range(len(cont1)):
        a, b = cont1[i].strip('\n').split('\t')
        b = np.asarray(b, dtype='float32')
        c = int(b)
        label[i][c] = 1
    dimension = 384
    neigh_f = open(data_path + '/paper_embeddings.txt', "r")
    embeddings = np.zeros((P_n, dimension))
    i = 0
    for line in neigh_f:
        id, sentence, embedding = line.strip('\n').split('\t')
        embedding = embedding[1:-2].replace(',', ' ')
        embedding = embedding.split()
        embeds = np.asarray(embedding, dtype='float32')
        embeddings[i] = embeds
        i = i + 1
    return apcpa,apa,aptpa,adj,embeddings,labels
def data_load_imdb(data_path):
    neigh_f = open(data_path + '/movie_abstract_embeddings.txt', "r", encoding='utf-8')
    embeddings = np.zeros((3293, 384))
    i = 0
    for line in neigh_f:
        id, embedding = line.strip('\n').split('\t')
        embedding = embedding[1:-2].replace(',', ' ')
        embedding = embedding.split()
        embeds = np.asarray(embedding, dtype='float32')
        embeddings[i] = embeds
        i = i + 1
    f1 = open(data_path + '/movie_directors.txt', 'r', encoding='utf-8')
    dm= np.zeros((1718, 3293))
    for line in f1.readlines():
        movie, director = line.strip('\n').split('\t')
        j = int(director)
        i = int(movie)
        dm[j][i] = 1
    dm = torch.from_numpy(dm)
    adj=dm
    dmd=dm.dot(dm.T)
    f = open(data_path + '/movie_actor1.txt', 'r', encoding='utf-8')
    ma = np.zeros((3293, 2130))
    for line in f.readlines():
        movie, actor = line.strip('\n').split('\t')
        j = int(actor)
        i = int(movie)
        ma[i][j] = 1
    uu = dm.dot(ma)
    dmamd = uu.dot(uu.T)
    label = np.zeros((1718, 3))
    f1 = open(data_path + '/director_labels.txt', "r")
    cont1 = f1.readlines()
    for i in range(len(cont1)):
        a, b = cont1[i].strip('\n').split('\t')
        b = np.asarray(b, dtype='float32')
        c = int(b)
        label[i][c] = 1
    return dmd,dmamd,embeddings,label,adj
def load_data_dblp():
    data_path='../data/dblp'
    apcpa, apa, aptpa, adj, embeddings, labels=data_load_dblp(data_path)
    embeddings = torch.from_numpy(embeddings)
    labels= torch.from_numpy(labels).long()
    num_classes = labels.shape[1]
    labels = labels.nonzero()[:, 1]
    apa = sparse.csr_matrix(apa)
    apcpa = sparse.csr_matrix(apcpa)
    apa = dgl.from_scipy(apa)
    apcpa = dgl.from_scipy(apcpa)
    aptpa = sparse.csr_matrix(aptpa)
    aptpa = dgl.from_scipy(aptpa)
    g = [apa,aptpa,apcpa]
    train_idx = range(811)
    val_idx = range(811,1622)
    test_idx = range(1622, 4057)
    num_nodes = apcpa.number_of_nodes()
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)
    return g, adj, embeddings,labels, num_classes, train_idx, val_idx, test_idx, train_mask, \
    val_mask, test_mask
def load_data_imdb():
    data_path='../data/imdb'
    dmd, dmamd, embeddings, label, adj=data_load_imdb(data_path)
    labels, features = torch.from_numpy(labels).long(), \
                       torch.from_numpy(embeddings).float()
    num_classes = labels.shape[1]
    labels = labels.nonzero()[:, 1]
    dmd = sparse.csr_matrix(dmd)
    dmd = dgl.from_scipy(dmd)
    dmamd = sparse.csr_matrix(dmamd)
    dmamd = dgl.from_scipy(dmamd)
    g = [dmd, dmamd]
    data_idx = []
    for i in range(1718):
        data_idx.append(i)
    np.random.shuffle(data_idx)
    train_idx = data_idx[0:1374]
    val_idx = data_idx[1374:1546]
    test_idx = data_idx[1546:1718]
    num_nodes = 1718  # 节点数量
    train_mask = get_binary_mask(num_nodes, train_idx)  # 对应位置上的节点设置为1，其余位置为0
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)
    return g, adj, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, \
           val_mask, test_mask
