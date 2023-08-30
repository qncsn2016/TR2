from argparse import ArgumentParser

BATCHNORM_MOMENTUM = 0.01

class Config(object):
    """Wrapper class for model hyperparameters."""

    def __init__(self):
        """
        Defaults
        """
        self.mode = None
        self.save_path = None
        self.model_path = None
        self.data_path = None
        self.datasize = None
        self.ckpt = None
        self.optimizer = None
        self.parser = self.setup_parser()
        self.args = vars(self.parser.parse_args())
        self.__dict__.update(self.args)

    def setup_parser(self):
        """
        Sets up an argument parser
        :return:
        """
        parser = ArgumentParser(description='training code')
        parser.add_argument('-mode', dest='mode', help='predcls/sgcls/sgdet', default='predcls', type=str)
        parser.add_argument('-data_path', default='', type=str)
        parser.add_argument('-pre_path', default='', help='precomputed feature', type=str)
        parser.add_argument('-model_path', default='',help='path of ckpt', type=str)
        parser.add_argument('-output_path', default='output/', type=str)

        parser.add_argument('-optimizer', help='adamw/adam/sgd', default='adamw', type=str)
        parser.add_argument('-lr', dest='lr', help='learning rate', default=1e-5, type=float)
        parser.add_argument('-nepoch', help='epoch number', default=15, type=float)
        parser.add_argument('-datasize', dest='datasize', default='large', type=str)
        parser.add_argument('-enc_layer', default=1, type=int)
        parser.add_argument('-dec_layer', default=3, type=int)
        parser.add_argument('-object_loss',default=10)

        return parser
