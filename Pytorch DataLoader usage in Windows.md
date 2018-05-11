# Pytorch DataLoader usage in Windows

The main difference in Python between Windows and Unix systems is multiprocessing . 
Here is an example

## Data set
```python
import torch
import torch.utils.data as Data
torch_dataset = Data.TensorDataset( x, y)
```
## Loader
```python
loader = Data.DataLoader(
    dataset=torch_dataset, # torch TensorDataset format
    batch_size=BATCH_SIZE, # mini batch size
    shuffle=True, 
    num_workers=2, 
)
```
**Note**: `num_workers` should be `0` if using CUDA

## Multiprocessing in Windows
This is a major difference
```python
def main():
    for step, (batch_x, batch_y) in enumerate(loader):
        # do something here
if __name__ == '__main__':
    # __spec__ = None # uncomment if using Spyder
    main()
```
