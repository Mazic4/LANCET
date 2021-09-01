class mnist_config:
    dataset = 'mnist'
    image_size = 28 * 28
    num_label = 10

    gen_emb_size = 20
    noise_size = 100

    dis_lr = 3e-3
    enc_lr = 1e-3
    gen_lr = 1e-3

    eval_period = 600
    vis_period = 100

    data_root = 'data'

    size_labeled_data = 100

    train_batch_size = 100
    train_batch_size_2 = 100
    dev_batch_size = 20

    seed = 13

    feature_match = True
    top_k = 5
    top1_weight = 1.

    supervised_only = False
    feature_match = True
    p_loss_weight = 1e-4
    p_loss_prob = 0.1
    
    max_epochs = 2000

    load_model = 0
    savedir = 'model/'
    pixelcnn_path = 'model/mnist.True.3.best.pixel'

class svhn_config:
    dataset = 'svhn'
    image_size = 3 * 32 * 32
    num_label = 10

    gen_emb_size = 20
    noise_size = 100

    dis_lr = 1e-3
    enc_lr = 1e-3
    gen_lr = 1e-3
    min_lr = 1e-4

    eval_period = 1460
    vis_period =1460
    label_period = 36500
    sys_label = True
    data_root = 'data'
    log_path = "svhn_intermidiate_result"

    #this value is default as 100 when sys_label is True
    size_labeled_data = 1000

    train_batch_size = 100
    train_batch_size_2 = 100
    dev_batch_size = 200

    max_epochs = 150
    ent_weight = 0.1
    pt_weight = 0.8

    p_loss_weight = 1e-4
    p_loss_prob = 0.1

class cifar_config:
    dataset = 'cifar'
    image_size = 3 * 32 * 32
    num_label = 10

    gen_emb_size = 20
    noise_size = 100

    dis_lr = 6e-4 * 2
    enc_lr = 3e-4
    gen_lr = 3e-4

    eval_period = 1000
    vis_period = 1000
    label_period = 25000

    data_root = '../data'
    log_path = "cifar_intermidiate_result"

    size_labeled_data = 500

    train_batch_size = 100
    train_batch_size_2 = 100
    dev_batch_size = 2000

    max_epochs = 300
    vi_weight = 1e-2

class cifar100_config:
    dataset = 'cifar'
    image_size = 3 * 32 * 32
    num_label = 100

    gen_emb_size = 20
    noise_size = 100

    dis_lr = 6e-4
    enc_lr = 3e-4
    gen_lr = 3e-4

    eval_period = 250
    vis_period = 250
    label_period = 6250

    data_root = '../data'

    size_labeled_data = 2000

    train_batch_size = 400
    train_batch_size_2 = 400
    dev_batch_size = 400

    max_epochs = 900
    vi_weight = 1e-2

class speechcommand_config:
    
    dataset = 'speechcommand'
    train_path = 'data/speechcommand_dataset/train'
    test_path = 'data/speechcommand_dataset/test'
    valid_path = 'data/speechcommand_dataset/valid'

    # Training settings
    batch_size = 100
    test_batch_size = 100
    arc = 'LeNet'
    epochs = 500
    lr = 6e-4
    momentum = 0.9
    optimizer = 'adam'
    cuda = True
    seed = 1234
    log_interval = 160
    patience = 10

    # feature extraction options
    window_size = .02
    window_stride = .01
    window_type = 'hamming'
    normalize = True

class har_config:
    dataset = 'har'
    image_size = 3 * 32 * 32
    num_label = 10

    gen_emb_size = 20
    noise_size = 100

    dis_lr = 6e-4 * 2
    enc_lr = 3e-4
    gen_lr = 3e-4

    eval_period = 1000
    vis_period = 1000
    label_period = 25000

    data_root = '../data'
    log_path = "har_intermidiate_result"

    size_labeled_data = 100

    train_batch_size = 100
    train_batch_size_2 = 100
    dev_batch_size = 2000

    max_epochs = 200
    vi_weight = 1e-2
