"""Small CPU networks for a fixed-budget tabular and history-window benchmark.

Architectures are explicit local implementations, not reproductions of TabNet,
FT-Transformer, or another published benchmark's full training recipe.
"""
import math
import numpy as np
import torch
from torch import nn
from scipy.special import expit
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import Ridge, LogisticRegression


class ResidualBlock(nn.Module):
    def __init__(self, width, gated=False, ablation=None):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.a, self.b = nn.Linear(width, width), nn.Linear(width, width)
        self.drop = nn.Dropout(.1)
        self.gated = gated
        self.ablation = ablation

    def forward(self, x):
        z = self.norm(x)
        if self.gated:
            z = torch.relu(self.a(z)) * (1 if self.ablation == 'No gates' else torch.sigmoid(self.b(z)))
        else:
            z = self.b(torch.relu(self.a(z)))
        return torch.relu(self.drop(z) if self.ablation == 'No residuals' else x + self.drop(z))


class CrossNet(nn.Module):
    def __init__(self, inputs, width, ablation=None):
        super().__init__()
        self.cross_w = nn.ParameterList([nn.Parameter(torch.randn(inputs)/math.sqrt(inputs)) for _ in range(3)])
        self.cross_b = nn.ParameterList([nn.Parameter(torch.zeros(inputs)) for _ in range(3)])
        self.deep = nn.Sequential(nn.Linear(inputs, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())
        self.head = nn.Linear(inputs+width, 1)
        self.ablation = ablation

    def forward(self, x):
        crossed = x
        for w, b in zip(self.cross_w, self.cross_b):
            crossed = x * (crossed @ w).unsqueeze(-1) + b + crossed
        if self.ablation == 'No cross branch':
            crossed = torch.zeros_like(crossed)
        return self.head(torch.cat([crossed, self.deep(x)], dim=1))


class FieldTransformer(nn.Module):
    def __init__(self, groups, width, ablation=None):
        super().__init__()
        self.groups = groups
        self.tokens = nn.ModuleList([nn.Linear(len(g), width) for g in groups])
        layer = nn.TransformerEncoderLayer(width, 2, dim_feedforward=width*2, dropout=.1,
                                            batch_first=True, activation='gelu', norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.ablation = ablation

    def forward(self, x):
        tokens = torch.stack([layer(x[:, list(group)]) for layer, group in zip(self.tokens, self.groups)], dim=1)
        return self.head((tokens if self.ablation == 'No attention' else self.encoder(tokens)).mean(dim=1))


class CausalConv(nn.Module):
    def __init__(self, inputs, outputs, dilation):
        super().__init__()
        self.dilation = dilation
        self.conv = nn.Conv1d(inputs, outputs, 2, dilation=dilation)

    def forward(self, x):
        return self.conv(nn.functional.pad(x, (self.dilation, 0)))


class HistoryNet(nn.Module):
    def __init__(self, kind, channels, width, steps):
        super().__init__()
        self.kind = kind
        if kind == 'TCN':
            self.body = nn.Sequential(CausalConv(channels, width, 1), nn.ReLU(), CausalConv(width, width, 2), nn.ReLU())
        elif kind == 'SeqTransformer':
            self.project = nn.Linear(channels, width)
            self.position = nn.Parameter(torch.randn(1, steps, width)*.02)
            layer = nn.TransformerEncoderLayer(width, 2, width*2, dropout=.1, batch_first=True, norm_first=True)
            self.body = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        else:
            cls = nn.LSTM if 'LSTM' in kind else nn.GRU if 'GRU' in kind else nn.RNN
            self.body = cls(channels, width, batch_first=True, bidirectional=kind.startswith('Bi'))
        self.head = nn.Sequential(nn.Dropout(.1), nn.Linear(width*(2 if kind.startswith('Bi') else 1), 1))

    def forward(self, x):
        if self.kind == 'TCN':
            last = self.body(x.transpose(1, 2))[:, :, -1]
        elif self.kind == 'SeqTransformer':
            last = self.body(self.project(x)+self.position)[:, -1]
        else:
            _, hidden = self.body(x)
            if 'LSTM' in self.kind:
                hidden = hidden[0]
            last = torch.cat([hidden[-2], hidden[-1]], dim=1) if self.kind.startswith('Bi') else hidden[-1]
        return self.head(last)


class TorchModel:
    def __init__(self, kind, params, groups, cls, seed, channels=0, steps=3, epochs=40, ablation=None):
        self.kind, self.params, self.groups = kind, params, groups
        self.cls, self.seed, self.channels, self.steps, self.epochs = cls, seed, channels, steps, epochs
        self.ablation = ablation

    def fit(self, X, y, validation=None):
        torch.set_num_threads(1)
        torch.manual_seed(self.seed)
        torch.use_deterministic_algorithms(True)
        width = self.params['width']
        n_inputs = X.shape[1]
        if self.channels:
            self.net = HistoryNet(self.kind, self.channels, width, self.steps)
        elif self.kind == 'FeatureTransformer':
            self.net = FieldTransformer(self.groups, width, self.ablation)
        elif self.kind == 'DCN':
            self.net = CrossNet(n_inputs, width, self.ablation)
        elif self.kind in ['ResMLP', 'GatedMLP']:
            self.net = nn.Sequential(nn.Linear(n_inputs, width), nn.ReLU(),
                                     ResidualBlock(width, self.kind == 'GatedMLP', self.ablation),
                                     ResidualBlock(width, self.kind == 'GatedMLP', self.ablation), nn.Linear(width, 1))
        else:
            activation = nn.Identity if self.ablation == 'No activations' else nn.ReLU
            self.net = nn.Sequential(nn.Linear(n_inputs, width), activation(), nn.Dropout(.1),
                                     nn.Linear(width, width), activation(), nn.Dropout(.1), nn.Linear(width, 1))
        xx = torch.from_numpy(np.asarray(X, dtype=np.float32))
        if self.channels:
            xx = xx.reshape(len(xx), self.steps, self.channels)
        yy = torch.from_numpy(np.asarray(y, dtype=np.float32)).reshape(-1, 1)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.params['lr'], weight_decay=self.params['weight_decay'])
        criterion = nn.BCEWithLogitsLoss() if self.cls else nn.MSELoss()
        self.history = []
        self.parameter_count = sum(p.numel() for p in self.net.parameters())
        for epoch in range(self.epochs):
            self.net.train()
            order = torch.randperm(len(xx))
            total = 0.
            for ids in order.split(2048):
                opt.zero_grad(set_to_none=True)
                loss = criterion(self.net(xx[ids]), yy[ids])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'{self.kind}: nonfinite training loss')
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 5.)
                opt.step()
                total += float(loss.detach())*len(ids)
            self.history.append({'epoch': epoch+1, 'training_objective': total/len(xx)})
            if validation is not None and epoch+1 in [1,5,10,20,30,40]:
                # Curves only: neither weights, epoch count nor parameters use these losses.
                self.net.eval()
                vv = torch.from_numpy(np.asarray(validation[0],dtype=np.float32))
                if self.channels:
                    vv = vv.reshape(len(vv),self.steps,self.channels)
                vy = torch.from_numpy(np.asarray(validation[1],dtype=np.float32)).reshape(-1,1)
                with torch.inference_mode():
                    for label, inputs, actual in [('train_eval_loss',xx,yy),('validation_eval_loss',vv,vy)]:
                        loss_sum = 0.
                        for part,target in zip(inputs.split(4096),actual.split(4096)):
                            loss_sum += float(criterion(self.net(part),target))*len(part)
                        self.history[-1][label] = loss_sum/len(inputs)
        self.net.eval()
        return self

    def predict(self, X):
        self.net.eval()
        xx = torch.from_numpy(np.asarray(X, dtype=np.float32))
        if self.channels:
            xx = xx.reshape(len(xx), self.steps, self.channels)
        with torch.inference_mode():
            output = torch.cat([self.net(part).reshape(-1) for part in xx.split(4096)]).numpy()
        return expit(output) if self.cls else output


class BasisNetwork:
    """ELM random hidden units or learned RBF centres, with a fitted output head."""
    def __init__(self, kind, params, cls, seed):
        self.kind, self.params, self.cls, self.seed = kind, params, cls, seed

    def transform(self, X):
        if self.kind == 'ELM':
            z = X @ self.weights + self.bias
            return np.tanh(z) if self.params['activation'] == 'tanh' else np.maximum(z, 0)
        return np.exp(-self.params['gamma']*cdist(X, self.centres, metric='sqeuclidean')/X.shape[1])

    def fit(self, X, y):
        if self.kind == 'ELM':
            rng = np.random.default_rng(self.seed)
            self.weights = rng.normal(size=(X.shape[1], self.params['width']))/math.sqrt(X.shape[1])
            self.bias = rng.normal(size=self.params['width'])
        else:
            km = MiniBatchKMeans(n_clusters=self.params['centres'], random_state=self.seed,
                                  batch_size=2048, n_init=3, max_iter=60)
            km.fit(X)
            self.centres = km.cluster_centers_
        z = self.transform(X)
        self.output = LogisticRegression(C=1/self.params['alpha'], max_iter=2000, random_state=self.seed) if self.cls else Ridge(alpha=self.params['alpha'])
        self.output.fit(z, y)
        return self

    def predict(self, X):
        z = self.transform(X)
        return self.output.predict_proba(z)[:, 1] if self.cls else self.output.predict(z)
