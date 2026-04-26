import numpy as np
import time
from FL_core.client.clientALA import *



class FedALA(object):
    def __init__(self, args):
        self.device = args.device
        self.global_rounds = args.global_rounds
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.join_clients = int(self.num_clients * self.join_ratio)

        self.clients = []
        self.selected_clients = []

        self.set_clients(args, clientALA)

        self.Budget = []


    def train(self, args):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models(args)   
            

            for client in self.selected_clients:    
                client.train(i)

                self.receive_models()
                self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

        print("\nBest global accuracy.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        for client in self.selected_clients:
            print(f'============================Client{client.cid}=========================================')
            print(f'Best loss: {client.best_loss}')
            print(f'Best dice: {client.best_dice}')
            print(f'Total loss: {client.losses}')
            print(f'Total dice: {client.dices}')
            print('=====================================================================')
            #print(f'args : {client.args}')
            print(f'Used datasets : {client.dataset}')
            print('=====================================================================')
            


    def set_clients(self, args, clientObj):
        for i in range(self.num_clients):
            client = clientObj(args, cid=i)
            self.clients.append(client)

    def select_clients(self):
        if self.random_join_ratio:
            join_clients = np.random.choice(range(self.join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            join_clients = self.join_clients
        selected_clients = list(np.random.choice(self.clients, join_clients, replace=False))

        return selected_clients
    
    def send_models(self, args):
        assert (len(self.clients) > 0)
        for client in self.clients:
            client.local_initialization(args, self.global_model)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_train_samples = 0
        for client in self.selected_clients:
            active_train_samples += client.c_weight   

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []
        
        for client in self.selected_clients:
            self.uploaded_weights.append(client.c_weight / active_train_samples) 
            self.uploaded_ids.append(client.cid)   
            self.uploaded_models.append(client.model)    

    def add_parameters(self, w, client_model):   
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    # Parameter aggregation weighted by data volume
    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data = torch.zeros_like(param.data) 
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)
