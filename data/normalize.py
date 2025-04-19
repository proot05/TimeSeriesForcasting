import torch


class Normalizer_ts():
    def __init__(self,params=[],method = '-11',dim=None):
        self.params = params
        self.method = method
        self.dim = dim

    def fit_normalize(self,data):
        assert type(data) == torch.Tensor
        if len(self.params) ==0:
            if self.method == '-11' or self.method == '01':
                if self.dim == None:
                    self.params = (torch.max(data),torch.min(data))
                else:
                    self.params = (torch.max(data,dim=self.dim, keepdim = True)[0],torch.min(data,dim=self.dim, keepdim = True)[0])
            elif self.method == 'ms':
                if self.dim == None:
                    self.params = (torch.mean(data,self.dim),torch.std(data,dim=self.dim))
                else:
                    self.params = (torch.mean(data,dim=self.dim, keepdim = True),torch.std(data,dim=self.dim, keepdim = True))
            elif self.method == 'none':
                self.params = None
        return self.fnormalize(data,self.params,self.method)

    def normalize(self, new_data):
        return self.fnormalize(new_data,self.params,self.method)

    def denormalize(self, new_data_norm):
        return self.fdenormalize(new_data_norm,self.params,self.method)

    def get_params(self):
        if self.method == 'ms':
            print('returning mean and std')
        elif self.method == '01':
            print('returning max and min')
        elif self.method == '-11':
            print('returning max and min')
        elif self.method == 'none':
            print('do nothing')
        return self.params

    @staticmethod
    def fnormalize(data, params, method):
        if method == '-11':
            return (data-params[1].to(data.device))/(params[0].to(data.device)-params[1].to(data.device))*2-1
        elif method == '01':
            return (data-params[1].to(data.device))/(params[0].to(data.device)-params[1].to(data.device))
        elif method == 'ms':
            return (data-params[0].to(data.device))/params[1].to(data.device)
        elif method == 'none':
            return data

    @staticmethod
    def fdenormalize(data_norm, params, method):
        if method == '-11':
            return (data_norm + 1) / 2 * (params[0].to(data_norm.device) - params[1].to(data_norm.device)) + params[
                1].to(data_norm.device)
        elif method == '01':
            return (data_norm) * (params[0].to(data_norm.device) - params[1].to(data_norm.device)) + params[1].to(
                data_norm.device)
        elif method == 'ms':
            return data_norm * params[1].to(data_norm.device) + params[0].to(data_norm.device)
        elif method == 'none':
            return data_norm