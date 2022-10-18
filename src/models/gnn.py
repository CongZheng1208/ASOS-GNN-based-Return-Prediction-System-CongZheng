import torch
import numpy as np
import os
from torch_geometric.nn import SAGEConv, to_hetero
from torch.nn import Linear
import torch_geometric.transforms as T
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader

from .model_template import ModelTemplate
from .utils import mse_loss, ce_loss, binary_ce_loss
from .sage_gnn import SAGEGNNEncoder

from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay


OPTIMIZERS = {
    "sparse-adam": torch.optim.SparseAdam,
    "adam": torch.optim.Adam
}

LOSS = {
    "mse": mse_loss,
    "cross-entropy": ce_loss,
    "binary-cross-entropy": binary_ce_loss
}

ENCODERS = {
    "sage-conv": SAGEGNNEncoder,
    # "sage-conv-with-linear": SAGEGNNEncoder_hetero,
    # "gat": GAT,
    # "hetero-gnn-1": HeteroGNN1
}

country_name = np.array([
    "Australia",
    "Austria",
    "Denmark",
    "France",
    "Germany",
    "Netherlands",
    "Sweden",
    "UK",
    "United States"
])

brand_name = np.array([
    "ASOS DESIGN",
    "ASOS Petite",
    "Topshop",
    "Stradivarius",
    "Bershka",
    "ASOS Curve",
    "New Look",
    "Collusion",
    "Nike",
    "other",
    "Pull&Bear"
    ]
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
class EdgeDecoder(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, dropout=0.0):
        super().__init__()
        self.dropout_fn = torch.nn.Dropout(dropout)

        self.lins = torch.nn.ModuleList()

        input_layer = Linear(2*input_channels, hidden_channels[0])
        self.lins.append(input_layer)

        if len(hidden_channels) > 1:
            for n in range(0, len(hidden_channels) - 1):
                lin = Linear(hidden_channels[n], hidden_channels[n+1])
                self.lins.append(lin)
            
        self.output_layer = Linear(hidden_channels[-1], 1)
        
    def forward(self, x_dict, edge_label_index):
        row, col = edge_label_index
        x = torch.cat([x_dict["customer"][row], x_dict["variant"][col]], dim=-1)
        
        for layer in self.lins:
            x = self.dropout_fn(layer(x))
            x = F.leaky_relu(x)

        x = self.output_layer(x).sigmoid()
        return torch.cat([x, torch.ones_like(x) - x], dim=1)


class GNNModel(torch.nn.Module):
    def __init__(self, data, model_args):
        super().__init__()
        self.data = data.data if data else None

        encoder_name = model_args.pop("encoder_name")
        encoder_args = model_args.pop("encoder_args")

        decoder_args = model_args.pop("decoder_args")
        decoder_args["input_channels"] = encoder_args["out_channels"]

        self.encoder = ENCODERS[encoder_name](data, **encoder_args)
        if encoder_name == "sage-conv" or encoder_name == "gat":
            self.encoder = to_hetero(self.encoder, self.data.metadata(), aggr="max")

        self.decoder = EdgeDecoder(**decoder_args)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        x = self.encoder(x_dict, edge_index_dict)
        return self.decoder(x, edge_label_index)


class GNNClf(ModelTemplate):
    def __init__(self, dataset, test_dataset, val_dataset=None, loss="mse", model_args=None, path=None):
        super().__init__("GNN Classifier")

        self.save_path = path
        self.data = dataset.data if dataset else None
        self.val_data = val_dataset.data if val_dataset else None
        self.test_data = test_dataset.data if test_dataset else None
        self.loss = LOSS[loss]

        optimizer_args = model_args.pop("optimizer")

        self.model = GNNModel(self.data, 
                              model_args).to(device)

        self.optimizer = OPTIMIZERS[optimizer_args["name"]](
            self.model.parameters(), **optimizer_args["args"])

        self.save_epochs = model_args.pop("save_epochs")
        self.batch_size = model_args.pop("batch_size")

        self.losses, self.val_losses = [], []
        self.accuracy, self.val_accuracy = [], []
        self.precision, self.val_precision = [], []
        self.recall, self.val_recall = [], []
        self.f1, self.val_f1 = [], []

        self.train_dataloader = NeighborLoader(
            self.data.data,
            directed=False,
            # Sample 30 neighbors for each node for 2 iterations
            num_neighbors={key: [10] * 2 for key in self.data.data.edge_types},
            # Use a batch size of 128 for sampling training nodes
            batch_size=self.batch_size,
            input_nodes=("customer", self.data.data["customer"].node_index)
        )
                                    

    def describe(self):
        return self.model

    def save(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, "model.pt"))

    def load(self, path):
        self.model.load_state_dict(torch.load(os.path.join(path, "model.pt"),  map_location=torch.device('cpu')))

    def get_train_results(self):
        scores = {
            "losses": self.losses,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1-score": self.f1
        }

        val_scores = {
            "losses": self.val_losses,
            "accuracy": self.val_accuracy,
            "precision": self.val_precision,
            "recall": self.val_recall,
            "f1-score": self.val_f1
        }

        return scores, val_scores

    def get_data(self):
        return self.train_data, self.val_data

    def decision_function(self, X):
        return self.model.decision_function(X)

    def train(self, epochs):
        for epoch in range(1, epochs + 1):
            train_loss, train_acc, train_prec, train_rec, train_f1 = 0, 0, 0, 0, 0
            for n, train_data in enumerate(self.train_dataloader, start=1):
                self.model.train()
                self.optimizer.zero_grad()
                train_data.to(device)
                pred = self.model.forward(train_data.x_dict, train_data.edge_index_dict,
                                    train_data["customer", "purchases", "variant"].edge_index)

                target = train_data["customer", "purchases", "variant"].edge_label

                train_loss += self.loss(pred, target)

                if epoch % self.save_epochs == 0:

                    train_results = self.get_scores(pred, target, loss=self.loss)

                    train_acc += train_results['accuracy']
                    train_prec += train_results['precision']
                    train_rec += train_results['recall']
                    train_f1 += train_results['f1-score']

            train_loss /= n
            train_loss.backward()
            self.optimizer.step()

            if epoch % self.save_epochs == 0:
                val_results = self.validation()

                train_loss = float(train_loss.detach().cpu())
                val_loss = float(val_results['loss'].cpu())
                val_acc = val_results['accuracy']
                val_prec = val_results['precision']
                val_rec = val_results['recall']
                val_f1 = val_results['f1-score']

                self.losses.append(train_loss)
                self.val_losses.append(val_loss)

                self.accuracy.append(train_acc / n)
                self.val_accuracy.append(val_acc)

                self.precision.append(train_prec / n)
                self.val_precision.append(val_prec)

                self.recall.append(train_rec / n)
                self.val_recall.append(val_rec)

                self.f1.append(train_f1 / n)
                self.val_f1.append(val_f1)

                print(f"""Epoch {epoch}, Train CE loss: {self.losses[-1]:.3f}, 
                Train Accuracy: {100*self.accuracy[-1]:.2f}%,
                Validation CE loss: {self.val_losses[-1]:.3f},
                Validation Accuracy: {100*self.val_accuracy[-1]:.2f}%""")

                if len(self.val_losses) > 1 and self.val_losses[-1] < min(self.val_losses[:-1]):
                    self.save()
                
            if len(self.val_losses) > 1 and self.val_losses[-1] < min(self.val_losses[:-1]):
                self.save()
                
             

    @torch.no_grad()
    def validation(self):
        self.model.eval()
        self.val_data.data.to(device)
        pred = self.model.forward(self.val_data.data.x_dict, self.val_data.data.edge_index_dict,
                            self.val_data.data["customer", "purchases", "variant"].edge_index)

        target = self.val_data.data["customer", "purchases", "variant"].edge_label

        scores = self.get_scores(pred, target, loss=self.loss)

        return scores

    @torch.no_grad()
    def test(self, data=False):
        self.model.eval()
        test_data = data if data else self.test_data[0]
        test_data.to(device)
        
        pred = self.model.forward(test_data.x_dict, test_data.edge_index_dict,
                            test_data["customer", "purchases", "variant"].edge_index)
        target = test_data["customer", "purchases", "variant"].edge_label

        scores = self.get_scores(pred, target, loss=self.loss)
        roc_scores = self.get_roc_scores(pred[:,1], target)
        scores["roc"] = roc_scores

        pred_processed = np.round(pred.detach().cpu().numpy()[:,1])
        target_processed = target.cpu().numpy()
        scores["confusion_mat"] = confusion_matrix(pred_processed, target_processed, normalize='all')

        #self.plotSingleCountryCM("all country", scores["confusion_mat"])

        self.test_by_country(test_data, pred, target, scores)

        return scores
    

    @torch.no_grad()
    def test_by_country(self, test_data, pred, target, scores):
        # test single country
        customer_id =  test_data["customer", "purchases", "variant"].edge_index[0]
        customer_id_t2n = customer_id.numpy()
   
        customer_co = test_data["customer", "is_from", "country"].edge_index
        customer_co_t2n = customer_co.numpy()

        data_size = test_data["customer", "purchases", "variant"].edge_index.size(dim=1)

        country = torch.empty(data_size)

        for i in range(data_size):
            index = np.argwhere( customer_co_t2n[0][:] == customer_id_t2n[i])   
            country[i] = customer_co[1][index[0][0]]
        country_t2n = country.numpy()

        for i in range(9):
            index_co = np.argwhere(country_t2n == i).T[0]







            pred_co = pred[index_co]
            target_co = target[index_co]

            pred_co_processed = np.round(pred_co.detach().cpu().numpy()[:,1])
            target_co_processed = target_co.cpu().numpy()

            confusion_mat = confusion_matrix(pred_co_processed, target_co_processed, normalize=None)

            scores_co = self.get_scores(pred_co, target_co, loss=self.loss)

            print("==========================================================================================================")
            print(f"The number of events of customers from {country_name[i]} is:{index_co.size}, {100*index_co.size/data_size}%")
            print(f"The number of customers with {country_name[i]} nationality is:{np.unique(customer_id_t2n[index_co]).size}")
            print(f"Score of {country_name[i]} is :{scores_co}")
            print(confusion_mat)
            print(" ")

            
            if(i==4):
                #self.plotSingleCountryCM(country_name[i], confusion_mat)
                self.plotSingleCountryROC(country_name[i], pred_co, target_co, "purple")
            
            if(i==5):
                #self.plotSingleCountryCM(country_name[i], confusion_mat)
                self.plotSingleCountryROC(country_name[i], pred_co, target_co, "deeppink")

            if(i==6):
                #self.plotSingleCountryCM(country_name[i], confusion_mat)
                self.plotSingleCountryROC(country_name[i], pred_co, target_co, "c")


        # Plot the mean ROC curve for all countries as well as the ROC curve for the random classifier
        fpr = scores["roc"]["fpr"]
        tpr = scores["roc"]["tpr"]
        roc_auc = scores["roc"]["auc"]

        plt.plot(fpr, tpr, color="red", label=f"ROC curve of all country (area = {roc_auc:.2f})")
        plt.plot([0, 1],[0, 1], color="navy", lw=1, linestyle="--", label="Random Classifier")

        plt.legend()
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.xlim(0, 1)
        plt.ylim(0, 1.05)
        plt.grid(alpha=0.3)
        plt.show()


    @torch.no_grad()
    def test_by_brand(self, test_data, pred, target):

        # test single brand. 
        variant_id =  test_data["customer", "purchases", "variant"].edge_index[1]
        variant_id_t2n = variant_id.numpy()
   
        variant_co = test_data["variant", "is_from", "brand"].edge_index
        variant_co_t2n = variant_co.numpy()

        data_size = test_data["customer", "purchases", "variant"].edge_index.size(dim=1)

        brand = torch.empty(data_size)

        for i in range(data_size):
            index = np.argwhere( variant_co_t2n[0][:] == variant_id_t2n[i])   
            brand[i] = variant_co[1][index[0][0]]

        brand_t2n = brand.numpy()

        for i in range(10):
            index_co = np.argwhere(brand_t2n == i).T[0]

            pred_co = pred[index_co]
            target_co = target[index_co]
            scores_co = self.get_scores(pred_co, target_co, loss=self.loss)
            print("Score is :")
            print(brand_name[i])
            print(scores_co)
            print(" ")       


    @torch.no_grad()
    def plotSingleCountryROC(self, country_name, pred_co, target_co, color):

        score = self.get_roc_scores(pred_co[:,1], target_co)
        fpr = score["fpr"]
        tpr = score["tpr"]
        roc_auc = score["auc"]

        plt.plot(fpr, tpr, color=color, label=f"ROC curve of {country_name} (area = {roc_auc:.2f})")

    @torch.no_grad()
    def plotSingleCountryCM(self, country_name, confusion_mat):
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat)
        disp.plot()
        plt.title(f"The confusion matrix of {country_name}")
        plt.show()