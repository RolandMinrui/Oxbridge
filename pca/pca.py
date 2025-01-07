from tqdm import tqdm
import torch
from torch_pca import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sentence_transformers import SentenceTransformer

class Data_Projection:
    def __init__(self, model_name_or_path, device, train_dataset, train_source, n_dim=2):
        self.device = device
        self.model = SentenceTransformer(model_name_or_path, device=device)
        self.train_dataset = train_dataset
        self.train_source = train_source
        self.pca = PCA(n_components=n_dim, svd_solver='auto')
        self.test_embeddings = {} # {name: embeddings}
        self.train()
    
    def train(self):
        print("----- Embedding the Dataset -----")
        train_embeddings = self.model.encode(self.train_dataset)
        train_embeddings = torch.from_numpy(train_embeddings).to(torch.float16).to(self.device) # not support bf16
        print("----- Fit and Run the PCA -----")
        train_embeddings = self.pca.fit_transform(train_embeddings)
        self.train_embeddings = train_embeddings
    
    def test(self, test_dataset, name):
        print("----- Embedding the Dataset -----")
        test_embeddings = self.model.encode(test_dataset)
        test_embeddings = torch.from_numpy(test_embeddings).to(torch.float16).to(self.device)
        print("----- Run the PCA -----")
        test_embeddings = self.pca.transform(test_embeddings)
        self.test_embeddings[name] = test_embeddings
        print(f"----- Add {name} to Embeddings -----")

        return test_embeddings

    def plot(self, test_embeddings): 
        train_np = self.train_embeddings.cpu().numpy()
        test_np = test_embeddings.cpu().numpy()
        c_dist = pairwise_distances(train_np, test_np, metric='euclidean').mean()

        plt.figure(figsize=(8, 6))
        plt.scatter(train_np[:, 0], train_np[:, 1], c='skyblue', label='Train Embeddings')
        plt.scatter(test_np[:, 0], test_np[:, 1], c='lightsalmon', label='Test Embeddings')
        plt.text(0.05, 0.05, f'Cluster Distance: {c_dist:.2f}', transform=plt.gca().transAxes, 
                 fontsize=12, verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.5))

        plt.legend()
        plt.show()
    
    def plot_source(self):
        train_np = self.train_embeddings.cpu().numpy()
        sources = set(self.train_source)
        colors = plt.cm.tab10.colors  # Use a colormap for distinct colors

        plt.figure(figsize=(8, 6))
        for i, source in enumerate(sources):
            # Filter embeddings for the current source
            indices = [idx for idx, src in enumerate(self.train_source) if src == source]
            source_embeddings = train_np[indices]
            plt.scatter(source_embeddings[:, 0], source_embeddings[:, 1], 
                        c=[colors[i]], label=source, alpha=0.7)

        plt.legend()
        plt.show()

    def plot_all(self):
        train_np = self.train_embeddings.cpu().numpy()
        num_tests = len(self.test_embeddings)
        
        fig, axes = plt.subplots(1, num_tests, figsize=(4 * num_tests, 6), squeeze=False)
        
        for i, (name, test_embeddings) in enumerate(self.test_embeddings.items()):
            test_np = test_embeddings.cpu().numpy()
            c_dist = pairwise_distances(train_np, test_np, metric='euclidean').mean()

            axes[0, i].scatter(train_np[:, 0], train_np[:, 1], c='skyblue', label='Train Embeddings')
            axes[0, i].scatter(test_np[:, 0], test_np[:, 1], c='lightsalmon', label=f'{name} Embeddings')
            axes[0, i].text(0.05, 0.05, f'Cluster Distance: {c_dist:.2f}', transform=axes[0, i].transAxes, 
                            fontsize=12, verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.5))

            # axes[0, i].legend()
            axes[0, i].set_title(name)
        
        plt.tight_layout()
        plt.show()
