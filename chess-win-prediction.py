# using peewee to open the file since it is a .db file (from lichess)
from peewee import *
import base64
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from torchmetrics import Accuracy
from torchsummary import summary
import pytorch_lightning as pl
from random import randrange
from collections import OrderedDict
import time

db = SqliteDatabase('./database/chess_games.db')

class Evaluations(Model):
  id = IntegerField()
  fen = TextField()
  binary = BlobField()
  eval = FloatField()

  class Meta:
    database = db

  def binary_base64(self):
    return base64.b64encode(self.binary)
  
db.connect()

# LABEL_COUNT reprensts number of rows in the database
LABEL_COUNT = 37164639
print(LABEL_COUNT)

# Global settings to use with nvidia cuda gpu
torch.set_float32_matmul_precision('high')

class EvaluationDataset(IterableDataset):
  def __init__(self, count):
    self.count = count

  def __iter__(self):
    return self
  
  def __next__(self):
    idx = randrange(self.count)
    return self[idx]
  
  def __len__(self):
    return self.count
  
  def __getitem__(self, index):
    eval = Evaluations.get(Evaluations.id == index+1)
    bin = np.frombuffer(eval.binary, dtype=np.uint8)
    bin = np.unpackbits(bin, axis=0).astype(np.single)
    eval.eval = max(eval.eval, -15)
    eval.eval = min(eval.eval, 15)
    ev = np.array([eval.eval]).astype(np.single)
    return { 'binary': bin, 'eval': ev }
  
dataset = EvaluationDataset(count=LABEL_COUNT)

class EvaluationModel(pl.LightningModule):
  def __init__(self, learning_rate=1e-3, batch_size=512, layer_count=10):
    super().__init__()
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    layers = []
    for i in range(layer_count-1):
      layers.append((f'linear-{i}', nn.Linear(808, 808)))
      layers.append((f'relu-{i}', nn.ReLU()))
    layers.append((f'linear-{layer_count-1}', nn.Linear(808, 1)))
    self.seq = nn.Sequential(OrderedDict(layers))

  def forward(self, x):
    return self.seq(x)
  
  def training_step(self, batch):
    x, y = batch['binary'], batch['eval']
    y_hat = self(x)
    loss = F.l1_loss(y_hat, y)
    self.log('train_loss', loss)
    return loss

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
  
  def train_dataloader(self):
    dataset = EvaluationDataset(count=LABEL_COUNT)
    return DataLoader(dataset, batch_size=self.batch_size, pin_memory=True, num_workers=15, persistent_workers=True)

if __name__ == '__main__':
  version_name = f'{int(time.time())}-batch_size-512-layer_count-10'
  logger = pl.loggers.TensorBoardLogger("lightning_logs", name="chessml", version=version_name)
  trainer = pl.Trainer(precision='16-mixed', max_epochs=3, accelerator='gpu', logger=logger)
  model = EvaluationModel(layer_count=16, batch_size=512, learning_rate=1e-3)
  print(model)
  summary(model, (808,), device='cpu')
  trainer.fit(model)

  from IPython.display import display, SVG
  from random import randrange

  SVG_BASE_URL = "https://us-central1-spearsx.cloudfunctions.net/chesspic-fen-image/" 

  def svg_url(fen):
    fen_board = fen.split()[0]
    return SVG_BASE_URL + fen_board

  def show_index(idx):
    eval = Evaluations.select().where(Evaluations.id == idx+1).get()
    batch = dataset[idx]
    x, y = torch.tensor(batch['binary']), torch.tensor(batch['eval'])
    y_hat = model(x)
    loss = F.l1_loss(y_hat, y)
    print(f'Idx {idx} Eval {y.data[0]:.2f} Prediction {y_hat.data[0]:.2f} Loss {loss:.2f}')
    print(f'FEN {eval.fen}')
    print(url=svg_url(eval.fen))

  for i in range(15):
    idx = randrange(LABEL_COUNT)
    show_index(idx)

  import chess

  MATERIAL_LOOKUP = { chess.KING: 0, chess.QUEEN: 9, chess.ROOK: 5, chess.BISHOP: 3, chess.KNIGHT: 3, chess.PAWN: 1 }

  def avg(lst):
    return sum(lst) / len(lst)

  def material_for_board(board):
    eval = 0.0
    for sq, piece in board.piece_map().items():
      mat = MATERIAL_LOOKUP[piece.piece_type] 
      if piece.color == chess.BLACK:
        mat = mat * -1
      eval += mat
    return eval
    
  def guess_zero_loss(idx):
    eval = Evaluations.select().where(Evaluations.id == idx+1).get()
    y = torch.tensor(eval.eval)
    y_hat = torch.zeros_like(y)
    loss = F.l1_loss(y_hat, y)
    return loss

  def guess_material_loss(idx):
    eval = Evaluations.select().where(Evaluations.id == idx+1).get()
    board = chess.Board(eval.fen)
    y = torch.tensor(eval.eval)
    y_hat = torch.tensor(material_for_board(board))
    loss = F.l1_loss(y_hat, y)
    return loss

  def guess_model_loss(idx):
    batch = dataset[idx]
    x, y = torch.tensor(batch['binary']), torch.tensor(batch['eval'])
    y_hat = model(x)
    loss = F.l1_loss(y_hat, y)
    return loss

  zero_losses = []
  mat_losses = []
  model_losses = []
  for i in range(100):
    idx = randrange(LABEL_COUNT)
    zero_losses.append(guess_zero_loss(idx))
    mat_losses.append(guess_material_loss(idx))
    model_losses.append(guess_model_loss(idx))

  print(f'Guess Zero Avg Loss {avg(zero_losses)}')
  print(f'Guess Material Avg Loss {avg(mat_losses)}')
  print(f'Guess Model Avg Loss {avg(model_losses)}')

  from sklearn.metrics import  mean_squared_error, r2_score

  y_list = []
  y_pred = []

  for i in range(1000000, 1010001):
    batch = dataset[i]
    x, y = torch.tensor(batch['binary']), torch.tensor(batch['eval'])
    y_hat = model(x)
    y_list.append(torch.Tensor.detach(y).numpy())
    y_pred.append(torch.Tensor.detach(y_hat).numpy())

  print(mean_squared_error(y_list, y_pred))
  print(r2_score(y_list, y_pred))